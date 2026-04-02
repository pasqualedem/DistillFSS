import torch
import torch.nn as nn
import torch.nn.functional as F

from distillfss.data.utils import BatchKeys
from distillfss.models.insid3.utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from distillfss.utils.utils import ResultDict
from huggingface_hub import PyTorchModelHubMixin
import einops

from .insid3 import INSID3

class ClassDistiller(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Predict candidate_mask
        self.candidate_pred = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
        )
        self.ref_prototype = nn.Parameter(torch.randn(embed_dim))

    def forward(self, query_feat):
        candidate_mask = self.candidate_pred(query_feat)
        return candidate_mask, self.ref_prototype

class DistilledINSID3(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes: int, insid3: INSID3 = None):
        super().__init__()
        self.num_classes = num_classes
        self.teacher = insid3
        
        # Assuming we use ViT-Base which has embed_dim=768
        # We can dynamically get it by checking the encoder
        embed_dim = self.teacher.encoder.embed_dim
        
        self.student = nn.ModuleList()
        for i in range(num_classes):
            self.student.append(ClassDistiller(embed_dim=embed_dim))

    def extract_features(self, images):
        return self.teacher.encoder.get_intermediate_layers(images, n=1, reshape=True)[0]
    
    def _class_predict(self, tgt_image, feat_tgt, feat_tgt_deb, class_idx):
        candidate_mask_float, ref_prototype = self.student[class_idx](feat_tgt)
        
        # Make candidate_mask binary with a threshold (e.g., 0.5)
        candidate_mask = torch.sigmoid(candidate_mask_float)
        candidate_mask = (candidate_mask > 0.5).squeeze()
        
        if candidate_mask.sum() == 0:
            final_mask = self.teacher._finalize_mask(candidate_mask, tgt_image)
            return {
                "pred_mask": final_mask,
                "prob_mask": torch.zeros_like(candidate_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                "candidate_mask": candidate_mask_float.unsqueeze(0).unsqueeze(0),
            }
        C = feat_tgt.shape[1]
        h, w = feat_tgt.shape[2], feat_tgt.shape[3]

        # Fine-grained clustering
        feat_tgt_flat = feat_tgt[0].reshape(C, -1).permute(1, 0)
        cluster_labels = agglomerative_clustering(feat_tgt_flat, self.teacher.tau).reshape(h, w)
        K = int(cluster_labels.max().item()) + 1

        feat_tgt_deb_flat = feat_tgt_deb[0].reshape(C, -1).permute(1, 0)
        cluster_protos = compute_cluster_prototypes(
            feat_tgt_deb_flat, cluster_labels.view(-1), K
        )

        # Seed selection and cluster aggregation
        pred_mask_unrefined, prob_mask = self.teacher._seed_and_aggregate(
            candidate_mask, cluster_labels, cluster_protos, K,
            ref_prototype.unsqueeze(1), feat_tgt, feat_tgt_deb, h, w
        )
        
        final_mask = self.teacher._finalize_mask(pred_mask_unrefined, tgt_image)
        return {
            "pred_mask": final_mask,
            "candidate_mask": candidate_mask_float.unsqueeze(0).unsqueeze(0),
            "prob_mask": prob_mask.float().unsqueeze(0).unsqueeze(0)
        }

    def forward(self, x):
        if self.training:
            teacher_result = self.teacher(x)
        else:
            teacher_result = {ResultDict.COARSE_MASKS: None}

        query_img = x[BatchKeys.IMAGES][:, 0]
        bsz, _, qry_h, qry_w = query_img.shape
        
        if "query_feat" in teacher_result:
            query_feat = teacher_result["query_feat"]
            query_feat_deb = teacher_result["query_feat_deb"]
        else:
            with torch.no_grad():
                query_feat = self.extract_features(query_img)
                query_feat = F.normalize(query_feat, p=2, dim=1)
                query_feat_deb = self.teacher._debias_features(query_feat.unsqueeze(0))[0]            
            
        masks = self.teacher._preprocess_masks(x[BatchKeys.PROMPT_MASKS], x[BatchKeys.DIMS])                                                                            
        decisions = []
        distilled_coarse_masks = []
        
        # We generate logits per class
        for c in range(masks.size(2)):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            n_shots = int(class_examples.sum().item())

            if n_shots == 0 and self.training == False:
                decisions.append(torch.zeros((bsz, 1, qry_h, qry_w), device=query_img.device, dtype=query_img.dtype))
                distilled_coarse_masks.append(None)
                continue

            # --- Student predictions ---
            student_res = self._class_predict(query_img, query_feat, query_feat_deb, c)
            decisions.append(student_res["pred_mask"].unsqueeze(0))
            distilled_coarse_masks.append((student_res["prob_mask"], student_res["candidate_mask"]))

        decisions = torch.stack(decisions, dim=1)
        probs = torch.cat([cm[0] for cm in distilled_coarse_masks], dim=1)
        if decisions.size(1) == 1:
            logits = torch.cat([1 - decisions.float(), decisions.float()], dim=1)
        else:
            # Take probs to decisions size
            probs = F.interpolate(probs, size=decisions.shape[2:], mode="bilinear", align_corners=False)
            
            # # BG probability is 1 - max_fg_prob masked by the union of all fg decisions
            max_prob = probs.max(dim=1, keepdim=True)[0]
            bg_prob = (1 - max_prob) * (1 - decisions.any(dim=1, keepdim=True).float())
            logits = torch.cat([bg_prob, probs], dim=1)
            
        logits = self.teacher.postprocess_masks(logits, x[BatchKeys.DIMS])
        key = ResultDict.DISTILLED_LOGITS if self.training else ResultDict.LOGITS

        return {
            key: logits,
            ResultDict.DISTILLED_COARSE: distilled_coarse_masks,
            **teacher_result
        }

def build_insid3_distiller(teacher, num_classes):
    return DistilledINSID3(num_classes=num_classes, insid3=teacher)
