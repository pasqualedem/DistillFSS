import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from distillfss.data.utils import BatchKeys
from distillfss.utils.utils import ResultDict
from huggingface_hub import PyTorchModelHubMixin

from .dmtnet import DMTNetwork
from einops import rearrange, repeat

class ClassDistiller(nn.Module):
    def __init__(self, dmtnet: DMTNetwork, num_conv_layers=1):
        super().__init__()
        self.stack_ids = dmtnet.stack_ids + 1
        self.num_conv_layers = num_conv_layers

        # dmtnet backbone channels depending on type
        if "vgg" in dmtnet.backbone_type:
            feat_channels = [256, 512, 512]
        elif "resnet" in dmtnet.backbone_type:
            feat_channels = [512, 1024, 2048]
        else:
            feat_channels = [512, 1024, 2048]  # Default
            
        self.coarse_extractor = nn.ModuleList([])
        for inch in feat_channels:
            # We want to map each feature straight to 128 channels (the input of hpn_learner.decoder1)
            conv_modules = [
                self.build_conv_block(inch, inch) for _ in range(num_conv_layers-1)
            ]
            self.coarse_extractor.append(nn.Sequential(*conv_modules, self.build_conv_block(inch, 128)))
            
    def build_conv_block(self, inch, outch=128):
        return nn.Sequential(
            nn.Conv2d(inch, inch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(inch, outch, kernel_size=1),
        )

    def forward(self, query_feats):
        # query_feats are from all layers. 
        # Stack IDs define where each block belongs.
        # We collect them and create the 128-channel encoded representation.
        
        corrs = []
        for idx, query_feat in enumerate(query_feats):
            if idx < self.stack_ids[0]:
                # mapped to first element (e.g. l4)
                coarse = self.coarse_extractor[0](query_feat)
            elif idx < self.stack_ids[1]:
                coarse = self.coarse_extractor[1](query_feat)
            elif idx < self.stack_ids[2]:
                coarse = self.coarse_extractor[2](query_feat)
            else:
                raise ValueError(f"Unexpected layer index {idx} for stack ids {self.stack_ids}")

            corrs.append(coarse)
            
        # We need to get one single representation from corrs.
        # In DMTNet, corrs from different scales have different spatial resolutions. 
        # They are interpolated to the resolution of the largest feature (sqz2) and added/averaged.
        
        # separate by block
        corr_l4 = torch.stack(corrs[:self.stack_ids[0]]).mean(dim=0)
        corr_l3 = torch.stack(corrs[self.stack_ids[0]:self.stack_ids[1]]).mean(dim=0)
        corr_l2 = torch.stack(corrs[self.stack_ids[1]:self.stack_ids[2]]).mean(dim=0)
        
        # interpolate to the largest size (corr_l4)
        spatial_size = corr_l4.size()[2:]
        corr_l3 = F.interpolate(corr_l3, spatial_size, mode='bilinear', align_corners=True)
        corr_l2 = F.interpolate(corr_l2, spatial_size, mode='bilinear', align_corners=True)
        
        hypercorr_encoded = corr_l4 + corr_l3 + corr_l2
        return hypercorr_encoded

class DistilledDMTNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes, dmtnet: DMTNetwork = None, num_conv_layers=1):
        super().__init__()
        self.num_classes = num_classes
        self.teacher = dmtnet
        
        self.student = nn.ModuleList()
        for i in range(num_classes):
            self.student.append(ClassDistiller(self.teacher, num_conv_layers))
            
    def _preprocess_masks(self, masks, dims):
        B, N, C, H, W = masks.size()
        masks = masks[:, :, 1:, ::]
        support_dims = dims[:, 1:]
        repeated_dims = repeat(support_dims, "b n d -> (b n c) d", c=C)
        masks = rearrange(masks, "b n c h w -> (b n c) h w")
        return rearrange(masks, "(b n c) h w -> b n c h w", b=B, n=N)

    def extract_features(self, query):
        query_feats = self.teacher.extract_feats(query, self.teacher.backbone, self.teacher.feat_ids, self.teacher.bottleneck_ids, self.teacher.lids)
        return query_feats
        
    def forward(self, x):
        if self.training:
            teacher_result = self.teacher(x)
        else:
            teacher_result = {ResultDict.COARSE_MASKS: None}

        query = x[BatchKeys.IMAGES][:, 0]
        with torch.no_grad():
            query_feats = self.extract_features(query)
            
        masks = self._preprocess_masks(x[BatchKeys.PROMPT_MASKS], x[BatchKeys.DIMS])
        
        # We loop over the number of classes.
        fg_logits_masks = []
        distilled_coarse_masks = []
        for c in range(masks.size(2)):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            n_shots = class_examples.sum().item()
            
            # Predict correlation encoded with student
            hypercorr_encoded = self.student[c](query_feats)
            distilled_coarse_masks.append([hypercorr_encoded])
            
            # Use teacher's decoder
            hypercorr_decoded = self.teacher.hpn_learner.decoder1(hypercorr_encoded)
            upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
            hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
            logit_mask = self.teacher.hpn_learner.decoder2(hypercorr_decoded)
            logit_mask = F.interpolate(logit_mask, query.size()[2:], mode='bilinear', align_corners=True)
            
            fg_logits_masks.append(logit_mask)
            
        raw_logits = torch.stack(fg_logits_masks, dim=1)
        raw_logits = F.softmax(raw_logits, dim=2)
        fg_logits = raw_logits[:, :, 1, ::]
        bg_logits = raw_logits[:, :, 0, ::]
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
        logits = torch.cat([bg_logits, fg_logits], dim=1)

        logits = self.teacher.postprocess_masks(logits, x["dims"])
        
        key = ResultDict.DISTILLED_LOGITS if self.training else ResultDict.LOGITS

        return {
            **teacher_result,
            key: logits,
            ResultDict.DISTILLED_COARSE: distilled_coarse_masks,
        }

