import torch
import torch.nn as nn
import torch.nn.functional as F

from distillfss.data.utils import BatchKeys
from distillfss.utils.utils import ResultDict

from . import PAHNetModel, build_pahnet


_FTS_SIZE = (64, 64)
_REDUCE_DIM = 256

# ResNet50 channel sizes (the only backbone supported by PAHNet checkpoints)
_IN_CHANNELS_4 = 2048   # qry_bcb_fts['3'] — layer4 with dilated convs
_IN_CHANNELS_5 = 512    # qry_bcb_fts['4'] — cls[0:2](ppm(layer4))


class PAHNetClassDistiller(nn.Module):
    """Per-class student that distills PAHNet's two-level similarity matching.

    The teacher computes four correlation maps from query × support:
      corr_fg_4, corr_4   (from layer4 features)
      corr_fg_5, corr_5   (from ppm+cls features)
    and merges them into corr_query_mask → init_merge_query → Swin transformer.

    This student learns to produce the same four maps from query features only
    (no support).  They flow directly into the teacher's frozen init_merge_query,
    replacing the support-dependent matching mechanism.  The Swin transformer is
    skipped (it needs cross-attention to support); a learnable per-class support
    prototype replaces supp_pro.
    """

    def __init__(self):
        super().__init__()

        # Level 4: layer4 (2048ch) → (corr_fg_4, corr_4)
        self.corr_extractor_4 = nn.Sequential(
            nn.Conv2d(_IN_CHANNELS_4, _REDUCE_DIM, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(_REDUCE_DIM, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        # Level 5: ppm+cls (512ch) → (corr_fg_5, corr_5)
        self.corr_extractor_5 = nn.Sequential(
            nn.Conv2d(_IN_CHANNELS_5, _REDUCE_DIM, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(_REDUCE_DIM, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        # Distilled support prototype — replaces supp_pro in init_merge_query
        self.supp_proto = nn.Parameter(torch.zeros(1, _REDUCE_DIM, 1, 1))

    def forward(self, query_feat_high_4, query_feat_high_5):
        """
        Args:
            query_feat_high_4: (B, 2048, H, W) — layer4 query features
            query_feat_high_5: (B,  512, H, W) — ppm+cls query features
        Returns:
            coarse_masks: [corr_fg_4, corr_4, corr_fg_5, corr_5]  each (B,1,64,64)
            corr_query_mask: (B, 2, 64, 64) — merged map for init_merge_query
            supp_proto_exp:  (B, 256, 64, 64) — expanded class prototype
        """
        B = query_feat_high_4.shape[0]

        # ---- level 4 ----
        corr4 = self.corr_extractor_4(query_feat_high_4)
        corr4 = F.interpolate(corr4, size=_FTS_SIZE, mode='bilinear', align_corners=True)
        corr_fg_4 = corr4[:, 0:1]
        corr_4    = corr4[:, 1:2]

        # ---- level 5 ----
        corr5 = self.corr_extractor_5(query_feat_high_5)
        corr5 = F.interpolate(corr5, size=_FTS_SIZE, mode='bilinear', align_corners=True)
        corr_fg_5 = corr5[:, 0:1]
        corr_5    = corr5[:, 1:2]

        # ---- merge: pixel-wise max over levels (differentiable proxy for
        #      teacher's per-sample best-selection + shot averaging) ----
        corr_fg = torch.max(corr_fg_4, corr_fg_5)
        corr    = torch.max(corr_4,    corr_5)
        corr_query_mask = torch.cat([corr_fg, corr], dim=1)  # (B, 2, 64, 64)

        supp_proto_exp = self.supp_proto.expand(B, _REDUCE_DIM, *_FTS_SIZE)

        return [corr_fg_4, corr_4, corr_fg_5, corr_5], corr_query_mask, supp_proto_exp


class DistilledPAHNet(nn.Module):
    """Distilled PAHNet: per-class convolutions replace the FSS matching mechanism.

    Result-dict keys (mirrors DistilledDCAMA):
      Training:  LOGITS (teacher), COARSE_MASKS (teacher corr maps),
                 DISTILLED_LOGITS (student), DISTILLED_COARSE (student corr maps)
      Inference: LOGITS (student)
    """

    def __init__(self, num_classes: int, pahnet: PAHNetModel = None, pahnet_args: dict = None):
        super().__init__()

        if pahnet is not None and pahnet_args is not None:
            raise ValueError("Provide pahnet or pahnet_args, not both.")
        if pahnet is None and pahnet_args is not None:
            pahnet = build_pahnet(**pahnet_args)
        if pahnet is None and pahnet_args is None:
            raise ValueError("Provide pahnet or pahnet_args.")

        self.num_classes = num_classes
        self.teacher = pahnet
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.student = nn.ModuleList([
            PAHNetClassDistiller()
            for _ in range(num_classes)
        ])

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()   # teacher always frozen
        return self

    # ------------------------------------------------------------------ helpers

    def _extract_query_feat(self, batch):
        """Return backbone features and the down-projected query feature map.

        Returns:
            qry_bcb_fts: raw backbone dict (keys '1'–'4')
            query_feat:  (B, 256, 64, 64) — down-projected, shared across classes
            query_feat_high_4: (B, 2048, H, W) — layer4 features
            query_feat_high_5: (B,  512, H, W) — ppm+cls features
        """
        query = batch[BatchKeys.IMAGES][:, 0]
        with torch.no_grad():
            qry_bcb_fts = self.teacher.extract_feats(query)
            query_feat = torch.cat([qry_bcb_fts['1'], qry_bcb_fts['2']], dim=1)
            query_feat = self.teacher.down_query(query_feat)
            query_feat = F.interpolate(query_feat, size=_FTS_SIZE, mode='bilinear', align_corners=True)
        return qry_bcb_fts, query_feat, qry_bcb_fts['3'], qry_bcb_fts['4']

    def _teacher_downstream(self, query_feat, corr_query_mask, supp_proto_exp, qry_bcb_fts, h, w):
        """Run teacher pipeline from init_merge_query onward (no Swin transformer).

        The corr_query_mask (student output) and supp_proto_exp (learned param)
        replace the support-dependent inputs.  Gradients flow back to the student
        through the frozen teacher modules.
        """
        # init_merge_query expects cat([query_feat, supp_pro, corr_query_mask]) = 256+256+2 = 514ch
        query_cat = torch.cat([query_feat, supp_proto_exp, corr_query_mask], dim=1)
        query_feat_merged = self.teacher.init_merge_query(query_cat)   # (B, 256, 64, 64)

        # Skip Swin transformer (needs support cross-attention).
        # Apply ReLU matching the teacher's post-transformer relu.
        merge_feat = self.teacher.relu(query_feat_merged)              # (B, 256, 64, 64)

        # Meta path
        query_meta = self.teacher.ASPP_meta(merge_feat)
        query_meta = self.teacher.res1_meta(query_meta)
        query_meta = self.teacher.res2_meta(query_meta) + query_meta
        meta_out = self.teacher.cls_meta(query_meta)                   # (B, 2, 64, 64)

        # Base path (query-only, unchanged)
        base_out = F.interpolate(qry_bcb_fts['4'], size=_FTS_SIZE, mode='bilinear', align_corners=True)
        base_out = self.teacher.base_learner(base_out)

        # Ensemble (no gram weighting in distilled mode: est_val = 0)
        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        meta_map_bg = meta_out_soft[:, 0:1]
        meta_map_fg = meta_out_soft[:, 1:]
        base_map    = base_out_soft[:, 1:].sum(1, keepdim=True)

        est_map = torch.zeros_like(meta_map_fg)
        meta_map_bg = self.teacher.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
        meta_map_fg = self.teacher.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

        merge_bg  = self.teacher.cls_merge(torch.cat([meta_map_bg, base_map], dim=1))
        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)           # (B, 2, 64, 64)

        return F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

    # ------------------------------------------------------------------ forward

    def forward(self, batch: dict):
        query  = batch[BatchKeys.IMAGES][:, 0]
        x_size = query.size()
        h = int((x_size[2] - 1) / 8 * self.teacher.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.teacher.zoom_factor + 1)

        if self.training:
            with torch.no_grad():
                teacher_result = self.teacher(batch)   # provides LOGITS + COARSE_MASKS
        else:
            teacher_result = {}

        qry_bcb_fts, query_feat, query_feat_high_4, query_feat_high_5 = \
            self._extract_query_feat(batch)

        distilled_coarse  = []
        distilled_logits  = []

        for c, distiller in enumerate(self.student):
            class_examples = batch[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            if class_examples.sum() == 0:
                distilled_coarse.append(None)
                distilled_logits.append(
                    torch.full((query.shape[0], 2, h, w), float('-inf'), device=query.device)
                )
                continue

            coarse_masks, corr_query_mask, supp_proto_exp = distiller(
                query_feat_high_4, query_feat_high_5
            )
            final_out = self._teacher_downstream(
                query_feat, corr_query_mask, supp_proto_exp, qry_bcb_fts, h, w
            )

            distilled_coarse.append(coarse_masks)          # [c_fg4, c4, c_fg5, c5]
            distilled_logits.append(final_out)

        # Multi-class ensemble (same as PAHNetModel.forward)
        logits       = torch.stack(distilled_logits, dim=1)   # (B, C, 2, H, W)
        fg_logits    = logits[:, :, 1, ::]
        bg_logits    = logits[:, :, 0, ::]
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits    = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
        logits       = torch.cat([bg_logits, fg_logits], dim=1)
        logits       = self.teacher.postprocess_masks(logits, batch["dims"])

        key = ResultDict.DISTILLED_LOGITS if self.training else ResultDict.LOGITS

        return {
            **teacher_result,
            ResultDict.DISTILLED_COARSE: distilled_coarse,
            key: logits,
        }
