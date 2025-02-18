import torch
import torch.nn as nn
import torch.nn.functional as F

from fssweed.data.utils import BatchKeys
from fssweed.utils.utils import ResultDict

from .dcama import DCAMAMultiClass, DCAMA_model, boost_coarse_map, postprocess_masks, refine_coarse_maps, reshape_and_prepare_features, stack_and_reshape_features


class ClassDistiller(nn.Module):
    def __init__(self, dcama: DCAMAMultiClass):
        super().__init__()
        self.coarse_extractor = nn.ModuleList([])
        for inch in dcama.feat_channels[1:]:
            self.coarse_extractor.append(self.build_conv_block(inch))
        self.stack_ids = dcama.stack_ids
            
    def build_conv_block(self, inch):
        return nn.Sequential(
            nn.Conv2d(inch, inch, kernel_size=3, padding=1),
            # nn.BatchNorm2d(inch),
            nn.ReLU(),
            nn.Conv2d(inch, 1, kernel_size=1)
        )
    
    def forward(self, query_feats):
        coarse_masks = []
        for idx, query_feat in enumerate(query_feats):
            if idx < self.stack_ids[0]: continue

            if idx < self.stack_ids[1]:
                coarse = self.coarse_extractor[0](query_feat)
            elif idx < self.stack_ids[2]:
                coarse = self.coarse_extractor[1](query_feat)
            else:
                coarse = self.coarse_extractor[2](query_feat)
                
            coarse_masks.append(coarse)
                
        coarse_masks1 = stack_and_reshape_features(coarse_masks, self.stack_ids, self.stack_ids[2], self.stack_ids[3])
        coarse_masks2 = stack_and_reshape_features(coarse_masks, self.stack_ids, self.stack_ids[1], self.stack_ids[2])
        coarse_masks3 = stack_and_reshape_features(coarse_masks, self.stack_ids, self.stack_ids[0], self.stack_ids[1])
        return coarse_masks1, coarse_masks2, coarse_masks3


class DistilledDCAMA(nn.Module):
    def __init__(self, num_classes, dcama: DCAMAMultiClass):
        super(DistilledDCAMA, self).__init__()
        self.num_classes = num_classes
        self.teacher = dcama
        
        self.student = nn.ModuleList()
        
        for i in range(num_classes):
            self.student.append(
                ClassDistiller(dcama)
            )
            
    def extract_features(self, x):
        query = x[BatchKeys.IMAGES][:, 0]
        query_features = self.teacher.extract_feats(query)
        return {
            ResultDict.QUERY_FEATS: query_features,
            ResultDict.SUPPORT_FEATS: None
        }
                    
    def forward(self, x):
        if self.training:
            dcama_result = self.teacher(x)
        else:
            dcama_result = self.extract_features(x)
        
        dcama_result[ResultDict.DISTILLED_COARSE] = []
        dcama_result[ResultDict.DISTILLED_LOGITS] = []
        for i, distiller in enumerate(self.student):
            query_feats = dcama_result[ResultDict.QUERY_FEATS]
            support_feats = dcama_result[ResultDict.SUPPORT_FEATS]
            if query_feats[i] is None:
                query_img = x[BatchKeys.IMAGES][:, 0]
                dcama_result[ResultDict.DISTILLED_COARSE].append(None)
                dcama_result[ResultDict.DISTILLED_LOGITS].append(torch.full((1, 2, *query_img.shape[-2:]), -torch.inf, device=query_img.device))   
                continue   
            if isinstance(query_feats[i], list): # If the element is still a list and not a tensor
                query_feats = query_feats[i]
                support_feats = support_feats[i]

            
            coarse_masks1, coarse_masks2, coarse_masks3 = distiller(query_feats)
            mix = self.teacher.model.mix_maps(coarse_masks1, coarse_masks2, coarse_masks3)
            mix, _, _ = self.teacher.model.skip_concat_features(mix, query_feats, support_feats, None)
            logit_mask, _, _ = self.teacher.model.upsample_and_classify(mix)
            dcama_result[ResultDict.DISTILLED_COARSE].append([coarse_masks1, coarse_masks2, coarse_masks3])
            dcama_result[ResultDict.DISTILLED_LOGITS].append(logit_mask)
            
        logits = dcama_result[ResultDict.DISTILLED_LOGITS]
        logits = torch.stack(logits, dim=1)
        fg_logits = logits[:, :, 1, ::]
        bg_logits = logits[:, :, 0, ::]
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
        logits = torch.cat([bg_logits, fg_logits], dim=1)

        logits = postprocess_masks(logits, x["dims"])

        key = ResultDict.DISTILLED_LOGITS if self.training else ResultDict.LOGITS

        return {
            **dcama_result,
            key: logits,
        }