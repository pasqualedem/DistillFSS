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

    def forward(self, query_feat):
        candidate_mask = self.candidate_pred(query_feat)
        return candidate_mask

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
        candidate_mask = self.student[class_idx](feat_tgt)
        
        return {
            "pred_mask": candidate_mask,
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
            decisions.append(student_res["pred_mask"])
            distilled_coarse_masks.append(None)
            
        # Check if decision have the same spatial size, if not interpolate to the largest one
        max_h = max(decision.shape[1] for decision in decisions)
        max_w = max(decision.shape[2] for decision in decisions)
        if any(decision.shape[1] != max_h or decision.shape[2] != max_w for decision in decisions):
            decisions = [
                F.interpolate(decision, size=(max_h, max_w), mode="bilinear", align_corners=False)
                for decision in decisions
            ]

        probs = torch.cat(decisions, dim=1)
        # log-sum-exp trick for stability
        logsumexp = torch.logsumexp(probs, dim=1, keepdim=True)  # [B, 1, H, W]

        # denominator = 1 + sum(exp(logits))
        # work in log-space:
        # log(den) = log(1 + sum(exp(logits))) = logaddexp(0, logsumexp)
        log_den = torch.logaddexp(torch.zeros_like(logsumexp), logsumexp)

        # foreground probabilities
        p_fg = torch.exp(probs - log_den)

        # background probability
        p_bg = torch.exp(-log_den)

        # concatenate
        logits = torch.cat([p_bg, p_fg], dim=1)  # [B, C+1, H, W]
            
        logits = self.teacher.postprocess_masks(logits, x[BatchKeys.DIMS])
        key = ResultDict.DISTILLED_LOGITS if self.training else ResultDict.LOGITS

        return {
            key: logits,
            ResultDict.DISTILLED_COARSE: distilled_coarse_masks,
            **teacher_result
        }

def build_insid3_distiller(teacher, num_classes):
    return DistilledINSID3(num_classes=num_classes, insid3=teacher)
