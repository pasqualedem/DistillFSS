import torch
import torch.nn as nn
import torch.nn.functional as F

from distillfss.data.utils import BatchKeys
from distillfss.utils.utils import ResultDict
from huggingface_hub import PyTorchModelHubMixin

from .msd import MSDNet

class ClassDistiller(nn.Module):
    def __init__(self, reduce_dim=64):
        super().__init__()
        
        # Predict 1-channel coarse corr_query_mask
        self.coarse_mask_pred = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(reduce_dim, 1, kernel_size=1)
        )
        
        # Predict global support vector (reduce_dim, 1, 1)
        self.supp_pp_pred = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1)
        )

    def forward(self, query_feat):
        corr_query_mask = self.coarse_mask_pred(query_feat)
        global_supp_pp = self.supp_pp_pred(query_feat)
        return corr_query_mask, global_supp_pp


class DistilledMSDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes, msdnet: MSDNet = None):
        super().__init__()
        self.num_classes = num_classes
        self.teacher = msdnet
        self.reduce_dim = msdnet.reduce_dim
        
        self.student = nn.ModuleList()
        for i in range(num_classes):
            self.student.append(ClassDistiller(reduce_dim=self.reduce_dim))

    def extract_features(self, query_img):
        qry_bcb_fts = self.teacher.backbone(query_img)
        query_feat = torch.cat([qry_bcb_fts['1'], qry_bcb_fts['2']], dim=1)
        query_feat = self.teacher.adjust_feature_qry(query_feat)
        return query_feat

    def forward(self, x):
        if self.training:
            teacher_result = self.teacher(x)
        else:
            teacher_result = {ResultDict.COARSE_MASKS: None}

        query_img = x[BatchKeys.IMAGES][:, 0]
        bsz, _, qry_h, qry_w = query_img.shape
        
        with torch.no_grad():
            query_feat = self.extract_features(query_img)
            
        masks = self.teacher._preprocess_masks(x[BatchKeys.PROMPT_MASKS], x[BatchKeys.DIMS])
        
        class_probs = []
        distilled_coarse_masks = []
        
        # Similar to typical loop, we generate logits per class
        for c in range(masks.size(2)):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            n_shots = int(class_examples.sum().item())

            if n_shots == 0 and self.training == False:
                class_probs.append(torch.zeros((bsz, 1, qry_h, qry_w), device=query_img.device, dtype=query_img.dtype))
                distilled_coarse_masks.append(None)
                continue

            # --- Student predictions ---
            corr_query_mask, global_supp_pp = self.student[c](query_feat)
            
            # Form dummy output for loss: coarse mask + GAP predicted support 
            # We bundle both for the MSE loss in distillfss.models.loss
            distilled_coarse_masks.append([corr_query_mask, global_supp_pp])
            
            # --- Decoder ---
            # 1. Run Teacher's Transformer
            query_embed = global_supp_pp.squeeze(-1)
            query_pos = self.teacher.query_embed.weight
            key_pos = self.teacher.pe_layer(query_feat)
            key_embed = query_feat
            masking = None
            fg_embed = self.teacher.transformer(key_embed, masking, query_embed, query_pos, key_pos)

            # 2. Run Teacher's CMGM merging operations
            pyramid_feat_list = []
            for idx, tmp_bin in enumerate(self.teacher.pyramid_bins):
                if tmp_bin <= 1.0:
                    bin_val = int(query_feat.shape[2] * tmp_bin)
                    query_feat_bin = nn.AdaptiveAvgPool2d(bin_val)(query_feat)
                else:
                    bin_val = tmp_bin
                    query_feat_bin = self.teacher.avgpool_list[idx](query_feat)
                
                supp_feat_bin = global_supp_pp.expand(-1, -1, bin_val, bin_val)
                corr_mask_bin = F.interpolate(corr_query_mask, size=(bin_val, bin_val), mode='bilinear', align_corners=True)
                
                merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)
                merge_feat_bin = self.teacher.init_merge[idx](merge_feat_bin)

                if idx >= 1:
                    pre_feat_bin = pyramid_feat_list[idx-1].clone()
                    pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin_val, bin_val), mode='bilinear', align_corners=True)
                    rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                    merge_feat_bin = self.teacher.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin

                merge_feat_bin = self.teacher.beta_conv[idx](merge_feat_bin) + merge_feat_bin
                merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
                pyramid_feat_list.append(merge_feat_bin)
                     
            query_feat_merged = torch.cat(pyramid_feat_list, 1)        
            query_feat_merged = self.teacher.res1(query_feat_merged)

            # 3. Output formatting
            out1 = torch.sigmoid(torch.einsum("bchw,bcl->blhw", query_feat_merged, fg_embed))
            out = F.interpolate(out1, size=(qry_h, qry_w), mode='bilinear', align_corners=True)
            class_probs.append(out)

        fg_probs = torch.cat(class_probs, dim=1)
        if fg_probs.size(1) == 1:
            logits = torch.cat([1 - fg_probs, fg_probs], dim=1)
        else:
            fg_logits = F.softmax(fg_probs, dim=1)
            bg_logits = (1 - fg_probs.clamp(0, 1)).prod(dim=1, keepdim=True)
            logits = torch.cat([bg_logits, fg_logits], dim=1)

        logits = self.teacher.postprocess_masks(logits, x[BatchKeys.DIMS])
        
        key = ResultDict.DISTILLED_LOGITS if self.training else ResultDict.LOGITS

        return {
            **teacher_result,
            key: logits,
            ResultDict.DISTILLED_COARSE: distilled_coarse_masks,
        }

def build_msdnet_distiller(teacher, num_classes):
    return DistilledMSDNet(num_classes=num_classes, msdnet=teacher)
