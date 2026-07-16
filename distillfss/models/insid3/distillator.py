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

        # Predict candidate_mask (foreground logit at feature resolution)
        self.candidate_pred = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
        )

        # Predict the reference prototype the teacher derives from the support set,
        # so the query-only student internalises the class (support) summary.
        self.prototype_pred = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )

    def forward(self, query_feat):
        candidate_mask = self.candidate_pred(query_feat)
        prototype = F.normalize(self.prototype_pred(query_feat), p=2, dim=1)
        return candidate_mask, prototype

class DistilledINSID3(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        num_classes: int,
        insid3: INSID3 = None,
        coarse: bool = True,
        logit_mode: str = "logits",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.teacher = insid3
        # coarse: distil the teacher's intermediate candidate_mask + prototype.
        #   Helps datasets where classes co-occur & are separable (WeedMap); HURTS
        #   single-class-per-image *diagnosis* datasets (ISIC) — see the class collapse
        #   documented in the ablation. Configure per-dataset in the model yaml.
        # logit_mode: "logits" (real logits, principled) or "double_softmax" (legacy
        #   probability-then-softmax; regularizes ill-posed diagnosis competitions).
        assert logit_mode in ("logits", "double_softmax"), f"bad logit_mode {logit_mode}"
        self.coarse = coarse
        self.logit_mode = logit_mode
        
        # Assuming we use ViT-Base which has embed_dim=768
        # We can dynamically get it by checking the encoder
        embed_dim = self.teacher.encoder.embed_dim
        
        self.student = nn.ModuleList()
        for i in range(num_classes):
            self.student.append(ClassDistiller(embed_dim=embed_dim))

    def extract_features(self, images):
        return self.teacher.encoder.get_intermediate_layers(images, n=1, reshape=True)[0]
    
    def _class_predict(self, tgt_image, feat_tgt, feat_tgt_deb, class_idx):
        candidate_mask, prototype = self.student[class_idx](feat_tgt)

        return {
            "pred_mask": candidate_mask,
            # Coarse targets mirroring the teacher intermediates (see COARSE_MASKS):
            "candidate_mask": torch.sigmoid(candidate_mask),  # match teacher's soft [0,1] map
            "prototype": prototype,                           # [B, C, 1, 1]
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

            # Build the coarse targets aligned with the teacher's COARSE_MASKS layout.
            teacher_coarse = teacher_result.get(ResultDict.COARSE_MASKS)
            teacher_coarse_c = teacher_coarse[c] if teacher_coarse is not None else None
            if self.coarse and teacher_coarse_c is not None:
                t_cand = teacher_coarse_c[0]
                s_cand = student_res["candidate_mask"]
                if s_cand.shape[-2:] != t_cand.shape[-2:]:
                    s_cand = F.interpolate(s_cand, size=t_cand.shape[-2:], mode="bilinear", align_corners=False)
                distilled = [s_cand]
                if len(teacher_coarse_c) > 1:
                    distilled.append(student_res["prototype"])
                distilled_coarse_masks.append(distilled)
            else:
                distilled_coarse_masks.append(None)
            
        # Check if decision have the same spatial size, if not interpolate to the largest one
        max_h = max(decision.shape[1] for decision in decisions)
        max_w = max(decision.shape[2] for decision in decisions)
        if any(decision.shape[1] != max_h or decision.shape[2] != max_w for decision in decisions):
            decisions = [
                F.interpolate(decision, size=(max_h, max_w), mode="bilinear", align_corners=False)
                for decision in decisions
            ]

        # Per-class foreground logits from each student head.
        fg_logits = torch.cat(decisions, dim=1)  # [B, C, H, W]

        # Assemble real multi-class logits with an implicit background logit of 0
        # (equivalently the "denominator = 1 + sum exp(fg)" reference). These are
        # raw logits: the FocalLoss / cross_entropy applies softmax itself, so we
        # must NOT pre-softmax into probabilities here (doing so double-softmaxes,
        # caps confidence, and starves the rarer class -> class collapse).
        if self.logit_mode == "double_softmax":
            # OLD (regressing) formulation: pre-softmax into probabilities, which
            # the FocalLoss then softmaxes AGAIN. Kept behind the toggle for ablation.
            logsumexp = torch.logsumexp(fg_logits, dim=1, keepdim=True)
            log_den = torch.logaddexp(torch.zeros_like(logsumexp), logsumexp)
            p_fg = torch.exp(fg_logits - log_den)
            p_bg = torch.exp(-log_den)
            logits = torch.cat([p_bg, p_fg], dim=1)  # [B, C+1, H, W] (probabilities)
        else:
            bg_logit = torch.zeros_like(fg_logits[:, :1])
            logits = torch.cat([bg_logit, fg_logits], dim=1)  # [B, C+1, H, W]

        logits = self.teacher.postprocess_masks(logits, x[BatchKeys.DIMS])
        key = ResultDict.DISTILLED_LOGITS if self.training else ResultDict.LOGITS

        return {
            key: logits,
            ResultDict.DISTILLED_COARSE: distilled_coarse_masks,
            **teacher_result
        }

def build_insid3_distiller(teacher, num_classes, coarse=True, logit_mode="logits"):
    return DistilledINSID3(
        num_classes=num_classes,
        insid3=teacher,
        coarse=coarse,
        logit_mode=logit_mode,
    )
