import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from distillfss.data.utils import BatchKeys
from distillfss.utils.utils import ResultDict

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
        
        
class SupportDistillerBlock(nn.Module):
    def __init__(self, inch, heads=32, use_support=False):
        super().__init__()
        
        self.keys = nn.Embedding(heads, inch)
        self.heads = heads
        self.use_support = use_support
        if use_support:
            self.conv_mappings = nn.ModuleList([self.build_conv_block(inch) for _ in range(heads)])
        self.attn = nn.MultiheadAttention(inch, num_heads=8)
        self.out = nn.Conv2d(inch, 1, kernel_size=1)
        
    def build_conv_block(self, inch):
        return nn.Sequential(
            nn.Conv2d(inch, inch, kernel_size=1, padding=1)
        )
        
    def forward(self, query_feats, support_feats):
        if self.use_support:
            mapped_supports = [self.conv_mappings[i](support_feats) for i in range(self.heads)]
            mapped_support = torch.stack(mapped_supports, dim=1)
            
            # Average pooling to remove H and W dimension
            mapped_support = einops.reduce(mapped_support, 'b heads c h w -> heads b c', 'mean')
            keys = einops.repeat(self.keys.weight, 'heads c -> heads b c', b=query_feats.shape[0])
            keys = keys + mapped_support
        else:
            keys = einops.repeat(self.keys.weight, 'heads c -> heads b c', b=query_feats.shape[0])
        
        h, w = query_feats.shape[-2:]
        query_feats = einops.rearrange(query_feats, 'b c h w -> (h w) b c')
        
        coarse_map = self.attn(query_feats, keys, keys)[0]
        coarse_map = einops.rearrange(coarse_map, '(h w) b c -> b c h w', h=h, w=w)
        coarse_map = self.out(coarse_map)
        return coarse_map
    
        
class SupportDistiller(nn.Module):
    def __init__(self, dcama: DCAMAMultiClass, use_support=False):
        super().__init__()
        self.coarse_extractor = nn.ModuleList([])
        for inch in dcama.feat_channels[1:]:
            self.coarse_extractor.append(SupportDistillerBlock(inch, use_support=use_support))
        self.stack_ids = dcama.stack_ids
            
    def forward(self, query_feats, support_feats):
        coarse_masks = []
        for idx, query_feat in enumerate(query_feats):
            if support_feats is None: 
                support_feat = None
            elif isinstance(support_feats[0], list):
                support_feat = torch.cat([support_example[idx] for support_example in support_feats], dim=2)
            else:
                support_feat = support_feats[idx]
                
            if idx < self.stack_ids[0]:
                continue
            if idx < self.stack_ids[1]:
                coarse = self.coarse_extractor[0](query_feat, support_feat)
            elif idx < self.stack_ids[2]:
                coarse = self.coarse_extractor[1](query_feat, support_feat)
            else:
                coarse = self.coarse_extractor[2](query_feat, support_feat)
                
            coarse_masks.append(coarse)
                
        coarse_masks1 = stack_and_reshape_features(coarse_masks, self.stack_ids, self.stack_ids[2], self.stack_ids[3])
        coarse_masks2 = stack_and_reshape_features(coarse_masks, self.stack_ids, self.stack_ids[1], self.stack_ids[2])
        coarse_masks3 = stack_and_reshape_features(coarse_masks, self.stack_ids, self.stack_ids[0], self.stack_ids[1])
        return coarse_masks1, coarse_masks2, coarse_masks3
        
        
class AttentionDistilledDCAMA(nn.Module):
    def __init__(self, num_classes, dcama: DCAMAMultiClass, use_support=False):
        super().__init__()
        self.num_classes = num_classes
        self.teacher = dcama
        self.use_support = use_support
        self.support_feats = None
        
        self.student = nn.ModuleList()
        
        for i in range(num_classes):
            self.student.append(
                SupportDistiller(dcama, use_support=use_support)
            )
            
    def extract_features(self, x):
        query = x[BatchKeys.IMAGES][:, 0]
        query_features = self.teacher.extract_feats(query)
        if not self.training and self.support_feats is None:
            support = x[BatchKeys.IMAGES][:, 1:]
            support_features = [self.teacher.extract_feats(support[:, i]) for i in range(support.shape[1])]
            self.support_feats = support_features
        return {
            ResultDict.QUERY_FEATS: query_features,
            ResultDict.SUPPORT_FEATS: self.support_feats
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
            
            coarse_masks1, coarse_masks2, coarse_masks3 = distiller(query_feats, support_feats)
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