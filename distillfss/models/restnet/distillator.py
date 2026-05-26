import torch
import torch.nn as nn
import torch.nn.functional as F

from distillfss.data.utils import BatchKeys
from distillfss.utils.utils import ResultDict
from huggingface_hub import PyTorchModelHubMixin

from .restnet import RestNetwork
from einops import rearrange, repeat


class ClassDistiller(nn.Module):
    def __init__(self, restnet: RestNetwork, num_conv_layers=1):
        super().__init__()
        self.stack_ids = restnet.stack_ids + 1
        self.num_conv_layers = num_conv_layers

        if "vgg" in restnet.backbone_type:
            feat_channels = [512, 512, 512]
        elif "resnet" in restnet.backbone_type:
            feat_channels = [512, 1024, 2048]
        else:
            feat_channels = [512, 1024, 2048]

        self.coarse_extractor = nn.ModuleList([])
        for inch in feat_channels:
            conv_modules = [
                self.build_conv_block(inch, inch) for _ in range(num_conv_layers - 1)
            ]
            self.coarse_extractor.append(nn.Sequential(*conv_modules, self.build_conv_block(inch, 128)))

    def build_conv_block(self, inch, outch=128):
        return nn.Sequential(
            nn.Conv2d(inch, inch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(inch, outch, kernel_size=1),
        )

    def forward(self, query_feats):
        corrs = []
        for idx, query_feat in enumerate(query_feats):
            if idx < self.stack_ids[0]:
                coarse = self.coarse_extractor[0](query_feat)
            elif idx < self.stack_ids[1]:
                coarse = self.coarse_extractor[1](query_feat)
            elif idx < self.stack_ids[2]:
                coarse = self.coarse_extractor[2](query_feat)
            else:
                raise ValueError(f"Unexpected layer index {idx} for stack ids {self.stack_ids}")
            corrs.append(coarse)

        corr_l4 = torch.stack(corrs[:self.stack_ids[0]]).mean(dim=0)
        corr_l3 = torch.stack(corrs[self.stack_ids[0]:self.stack_ids[1]]).mean(dim=0)
        corr_l2 = torch.stack(corrs[self.stack_ids[1]:self.stack_ids[2]]).mean(dim=0)

        spatial_size = corr_l4.size()[2:]
        corr_l3 = F.interpolate(corr_l3, spatial_size, mode='bilinear', align_corners=True)
        corr_l2 = F.interpolate(corr_l2, spatial_size, mode='bilinear', align_corners=True)

        hypercorr_encoded = corr_l4 + corr_l3 + corr_l2
        return hypercorr_encoded


class DistilledRestNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes, restnet: RestNetwork = None, num_conv_layers=1):
        super().__init__()
        self.num_classes = num_classes
        self.teacher = restnet

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
        query_feats = self.teacher.extract_feats(
            query, self.teacher.backbone, self.teacher.feat_ids,
            self.teacher.bottleneck_ids, self.teacher.lids
        )
        return query_feats

    def forward(self, x):
        if self.training:
            teacher_result = self.teacher(x)
        else:
            teacher_result = {}

        query = x[BatchKeys.IMAGES][:, 0]
        with torch.no_grad():
            query_feats = self.extract_features(query)

        masks = self._preprocess_masks(x[BatchKeys.PROMPT_MASKS], x[BatchKeys.DIMS])

        fg_logits_masks = []
        distilled_coarse_masks = []
        for c in range(masks.size(2)):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            n_shots = class_examples.sum().item()

            hypercorr_encoded = self.student[c](query_feats)
            distilled_coarse_masks.append([hypercorr_encoded])

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
