import torch
import torch.nn as nn
import torch.nn.functional as F

from distillfss.data.utils import BatchKeys
from distillfss.utils.utils import ResultDict
from huggingface_hub import PyTorchModelHubMixin


class SegICClassDistiller(nn.Module):
    """Lightweight student that learns refined correspondence features."""

    def __init__(self, vision_dim=768, out_dim=256):
        super().__init__()
        self.vision_dim = vision_dim
        self.out_dim = out_dim

        # Refine correspondence features
        self.refine = nn.Sequential(
            nn.Conv2d(vision_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_dim, kernel_size=1),
        )

    def forward(self, query_feat):
        return self.refine(query_feat)


class DistilledSegIC(nn.Module, PyTorchModelHubMixin):
    """SEGIC distillator with shared feature refinement + per-class projectors."""

    def __init__(self, num_classes, teacher_segic=None):
        super().__init__()
        self.num_classes = num_classes
        self.teacher = teacher_segic

        # Shared feature refiner for all classes
        vision_dim = 768 if teacher_segic is None else self._get_vision_dim()
        self.student = SegICClassDistiller(vision_dim, 256)

        # Per-class lightweight projectors for specialization
        self.class_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=1),
            )
            for _ in range(num_classes)
        ])

    def _get_vision_dim(self):
        if hasattr(self.teacher, "vision_dim"):
            return self.teacher.vision_dim
        if hasattr(self.teacher, "encoder") and hasattr(self.teacher.encoder, "embed_dim"):
            return self.teacher.encoder.embed_dim
        raise AttributeError("Unable to determine teacher SEGIC vision dimension.")

    def forward(self, x):
        # Teacher performs full correspondence discovery with support set
        teacher_result = self.teacher(x)

        if not self.training:
            return teacher_result

        query_img = x[BatchKeys.IMAGES][:, 0]
        dims = x[BatchKeys.DIMS]
        bsz = query_img.shape[0]
        query_ori_sizes = dims[:, 0, :]

        # Student only learns from query features, no support set info
        image_feat, _ = self.teacher.extract_dift_feature(
            query_img, query_ori_sizes, None, inference=True,
            is_inst_list=[False] * bsz,
        )

        # Shared student refines features
        refined_feat = self.student(image_feat)

        # Per-class projectors specialize the refined features
        distilled_coarse_feats = []
        for c in range(self.num_classes):
            class_refined = self.class_projectors[c](refined_feat)
            distilled_coarse_feats.append([class_refined])

        return {
            **teacher_result,
            ResultDict.DISTILLED_LOGITS: teacher_result[ResultDict.LOGITS],
            ResultDict.DISTILLED_COARSE: distilled_coarse_feats,
        }


def build_segic_distiller(teacher, num_classes):
    return DistilledSegIC(num_classes=num_classes, teacher_segic=teacher)
