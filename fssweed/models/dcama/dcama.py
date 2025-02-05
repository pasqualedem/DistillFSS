r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
from functools import reduce
from operator import add

from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

from fssweed.models.dcama.swin_transformer import SwinTransformer
from fssweed.models.dcama.transformer import MultiHeadedAttention, PositionalEncoding, get_attn_fn
from fssweed.data.utils import BatchKeys
from fssweed.utils.utils import ResultDict


def stack_and_reshape_features(features, stack_ids, start_idx, end_idx):
    bsz, ch, ha, wa = features[end_idx - 1 - stack_ids[0]].size()
    stacked_features = torch.stack(features[start_idx - stack_ids[0]:end_idx - stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
    return stacked_features


def reshape_and_prepare_features(query_feat, support_feats, support_mask, nshot, idx):
    bsz, ch, ha, wa = query_feat.size()
    query = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()
    
    if nshot == 1:
        support_feat = support_feats[idx]
        mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear', align_corners=True)
        mask = mask.view(support_feat.size()[0], -1)
        support_feat = support_feat.view(support_feat.size()[0], support_feat.size()[1], -1).permute(0, 2, 1).contiguous()
    else:
        support_feat = torch.stack([support_feats[k][idx] for k in range(nshot)])
        support_feat = support_feat.view(-1, ch, ha * wa).permute(0, 2, 1).contiguous()
        mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True) for k in support_mask])
        mask = mask.view(bsz, -1)
    return query, support_feat, mask

def refine_coarse_maps(coarse_maps, hyperparameters):
    """
    Refines the coarse maps based on hyperparameters, resizing other maps for compatibility.
    The mean across channels is used when resizing maps with different channel counts.

    Args:
        coarse_maps (list of torch.Tensor): List of tensors representing coarse maps.
                                            Each tensor has shape (B, C, H, W).
        hyperparameters (list of float): List of hyperparameters corresponding to each coarse map.

    Returns:
        list of torch.Tensor: Refined coarse maps.
    """
    refined_maps = []
    num_maps = len(coarse_maps)

    for i in range(num_maps):
        # Get the target map and its hyperparameter
        target_map = coarse_maps[i]
        hyperparam = hyperparameters[i]
        
        # Compute contributions from other maps
        other_maps_contribution = 0
        for j in range(num_maps):
            if i != j:
                # Resize the mean of the other map's channels to match the target map's size
                mean_map = coarse_maps[j].mean(dim=1, keepdim=True)  # Take mean across channels
                resized_map = F.interpolate(mean_map, size=target_map.shape[2:], mode='bilinear', align_corners=False)
                other_maps_contribution += (1 - hyperparam) / (num_maps - 1) * resized_map
        
        # Compute the refined map
        refined_map = hyperparam * target_map + other_maps_contribution
        refined_maps.append(refined_map)
    
    return refined_maps

def boost_coarse_map(coarse_maps, boost_alpha, boost_index):
    boost_map = coarse_maps[boost_index].unsqueeze(0)
    boosted_maps = []
    for i in range(len(coarse_maps)):
        cur_map = coarse_maps[i].unsqueeze(0)
        if i == boost_index:
            boosted_maps.append(cur_map)
            continue
        boosted_map_resized = F.interpolate(boost_map, size=cur_map.shape[2:], mode='bilinear', align_corners=False)
        boosted_map = boost_alpha * boosted_map_resized + (1 - boost_alpha) * cur_map
        boosted_maps.append(boosted_map)
    return boosted_maps


def separate_coarse_maps(coarse_maps, mask_lenghts):
    B, C, H, W = coarse_maps.shape
    result = []
    
    # Initialize the start index for slicing
    start_idx = 0
    
    # Iterate over the separation points to slice the tensor into chunks
    for lenght in mask_lenghts:
        # Slice the tensor from start_idx to end_idx along the channel dimension
        result.append(coarse_maps[:, start_idx:start_idx+lenght, :, :])
        # Update start_idx for the next chunk
        start_idx = start_idx+lenght
    
    # Handle the last chunk, from the last separation point to the end of the channel dimension
    if start_idx < C:
        result.append(coarse_maps[:, start_idx:, :, :])
    
    return result


class DCAMA(nn.Module):

    def __init__(self, backbone, pretrained_path, use_original_imgsize, concat_support=True, train_backbone=False):
        super(DCAMA, self).__init__()

        self.backbone = backbone
        self.use_original_imgsize = use_original_imgsize
        self.train_backbone = train_backbone

        # feature extractor initialization
        if backbone == 'resnet50':
            self.feature_extractor = resnet.resnet50()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
        elif backbone == 'resnet101':
            self.feature_extractor = resnet.resnet101()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.feat_ids = list(range(0, 34))
        elif backbone == 'swin':
            self.feature_extractor = SwinTransformer(img_size=384, patch_size=4, window_size=12, embed_dim=128,
                                            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.feature_extractor.load_state_dict(torch.load(pretrained_path)['model'])
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
        self.model = DCAMA_model(in_channels=self.feat_channels, stack_ids=self.stack_ids, concat_support=concat_support)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.raw_loss = nn.BCELoss()
    
    def forward(self, query_img, support_img, support_mask, query_mask=None):
        result = self.forward_1shot(query_img, support_img, support_mask)
        if query_mask is not None:
            result[ResultDict.LOSS] = self.compute_objective(result, query_mask)
        return result

    def forward_1shot(self, query_img, support_img, support_mask):
        if self.train_backbone:
            query_feats = self.extract_feats(query_img)
            support_feats = self.extract_feats(support_img)
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img)
                support_feats = self.extract_feats(support_img)

        return self.model(query_feats, support_feats, support_mask.clone())

    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []

        if self.backbone == 'swin':
            _ = self.feature_extractor.forward_features(img)
            for feat in self.feature_extractor.feat_maps:
                bsz, hw, c = feat.size()
                h = int(hw ** 0.5)
                feat = feat.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feats.append(feat)
        elif self.backbone == 'resnet50' or self.backbone == 'resnet101':
            bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
            # Layer 0
            feat = self.feature_extractor.conv1.forward(img)
            feat = self.feature_extractor.bn1.forward(feat)
            feat = self.feature_extractor.relu.forward(feat)
            feat = self.feature_extractor.maxpool.forward(feat)

            # Layer 1-4
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                res = feat
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

                if bid == 0:
                    res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

                feat += res

                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())

                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

    def predict_mask_nshot(self, batch, nshot):
        r""" n-shot inference """
        query_img = batch[BatchKeys.IMAGES][:, 0]
        support_imgs = batch[BatchKeys.IMAGES][:, 1:]
        support_masks = batch[BatchKeys.PROMPT_MASKS]

        if nshot == 1:
            result = self.forward_1shot(query_img, support_imgs[:, 0], support_masks[:, 0])
        else:
            query_feats = self.extract_feats(query_img)
            n_support_feats = []
            for k in range(nshot):
                support_feats = self.extract_feats(support_imgs[:, k])
                n_support_feats.append(support_feats)
            result = self.model(query_feats, n_support_feats, support_masks.clone(), nshot)
        logit_mask = result[ResultDict.LOGITS]

        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch[BatchKeys.DIMS][1].item(), batch[BatchKeys.DIMS][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        else:
            logit_mask = F.interpolate(logit_mask, support_imgs[0].size()[2:], mode='bilinear', align_corners=True)

        result[ResultDict.LOGITS] = logit_mask
        return result

    # def compute_objective(self, result, gt_mask): # Cross entropy
    #     logit_mask = result[ResultDict.LOGITS]
    #     bsz = logit_mask.size(0)
    #     logit_mask = logit_mask.view(bsz, 2, -1)
        
    #     fg_raw_attn1, fg_raw_attn2, fg_raw_attn3 = result[ResultDict.FG_RAW_ATTN_OUTS]
    #     fg_raw_attn1 = F.interpolate(fg_raw_attn1, gt_mask.size()[-2:], mode='bilinear', align_corners=True)
    #     fg_raw_attn2 = F.interpolate(fg_raw_attn2, gt_mask.size()[-2:], mode='bilinear', align_corners=True)
    #     fg_raw_attn3 = F.interpolate(fg_raw_attn3, gt_mask.size()[-2:], mode='bilinear', align_corners=True)
    #     fg_raw = torch.cat([fg_raw_attn1, fg_raw_attn2, fg_raw_attn3], dim=1).mean(dim=1).unsqueeze(1)
        
    #     bg_raw_attn1, bg_raw_attn2, bg_raw_attn3 = result[ResultDict.BG_RAW_ATTN_OUTS]
    #     bg_raw_attn1 = F.interpolate(bg_raw_attn1, gt_mask.size()[-2:], mode='bilinear', align_corners=True)
    #     bg_raw_attn2 = F.interpolate(bg_raw_attn2, gt_mask.size()[-2:], mode='bilinear', align_corners=True)
    #     bg_raw_attn3 = F.interpolate(bg_raw_attn3, gt_mask.size()[-2:], mode='bilinear', align_corners=True)
    #     bg_raw = torch.cat([bg_raw_attn1, bg_raw_attn2, bg_raw_attn3], dim=1).mean(dim=1).unsqueeze(1)
        
    #     raw = torch.stack([bg_raw, fg_raw], dim=1)
    #     raw = raw.view(bsz, 2, -1) / 100

    #     gt_mask = gt_mask.view(bsz, -1).long()
        
    #     loss_mask = self.cross_entropy_loss(logit_mask, gt_mask)
        
    #     loss_attn = self.raw_loss(raw, gt_mask)
    #     return (loss_attn / 1 * 2) + (loss_mask / 2 * 1)
    
    def compute_objective(self, result, gt_mask):
        logit_mask = result[ResultDict.LOGITS]
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        
        coarse_mask1, coarse_mask2, coarse_mask3 = result[ResultDict.COARSE_MASKS]
        coarse_mask1 = F.interpolate(coarse_mask1, gt_mask.size()[-2:], mode='bilinear', align_corners=True)
        coarse_mask2 = F.interpolate(coarse_mask2, gt_mask.size()[-2:], mode='bilinear', align_corners=True)
        coarse_mask3 = F.interpolate(coarse_mask3, gt_mask.size()[-2:], mode='bilinear', align_corners=True)
        coarse = torch.cat([coarse_mask1, coarse_mask2, coarse_mask3], dim=1).mean(dim=1).unsqueeze(1)
                    
        coarse = coarse.view(bsz, -1)
        # Clamp the values between 0 and 1
        coarse = torch.clamp(coarse, min=0, max=1)
        
        gt_mask = gt_mask.view(bsz, -1)
        
        loss_mask = self.cross_entropy_loss(logit_mask, gt_mask.long())
        loss_attn = self.raw_loss(coarse, gt_mask)
        return (loss_attn / 1 * 2) + (loss_mask / 2 * 1)


    def train_mode(self):
        self.train()
        self.feature_extractor.eval()
        
    def set_importance_levels(self, alpha=1, beta=1, gamma=1, boost_alpha=0.0, boost_index=0):
        self.model.alpha = alpha
        self.model.beta = beta
        self.model.gamma = gamma
        self.model.boost_alpha = boost_alpha
        self.model.boost_index = boost_index
        
    def set_attn_fn(self, **kwargs):
        for dcama_module in self.model.DCAMA_blocks:
            dcama_module.attn_fn = get_attn_fn(**kwargs)


class DCAMA_model(nn.Module):
    def __init__(self, in_channels, stack_ids, concat_support=True):
        super(DCAMA_model, self).__init__()

        self.stack_ids = stack_ids
        self.alpha = 1
        self.beta = 1
        self.gamma = 1
        self.boost_alpha = 0.0
        self.boost_index = 0
        self.concat_support = concat_support

        # DCAMA blocks
        self.DCAMA_blocks = nn.ModuleList()
        self.pe = nn.ModuleList()
        for inch in in_channels[1:]:
            self.DCAMA_blocks.append(MultiHeadedAttention(h=8, d_model=inch, dropout=0.5))
            self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))

        outch1, outch2, outch3 = 16, 64, 128

        # conv blocks
        self.conv1 = self.build_conv_block(stack_ids[3]-stack_ids[2], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1]) # 1/32
        self.conv2 = self.build_conv_block(stack_ids[2]-stack_ids[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1]) # 1/16
        self.conv3 = self.build_conv_block(stack_ids[1]-stack_ids[0], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1]) # 1/8

        self.conv4 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/32 + 1/16
        self.conv5 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8

        # mixer blocks
        in_mixer = outch3+2*in_channels[1]+2*in_channels[0] if concat_support else outch3+in_channels[1]+in_channels[0]
        self.mixer1 = nn.Sequential(nn.Conv2d(in_mixer, outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer3 = nn.Sequential(nn.Conv2d(outch1, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))

    def process_dcama_blocks(self, query, support_feat, mask, idx):
        if idx < self.stack_ids[1]:
            return self.DCAMA_blocks[0](self.pe[0](query), self.pe[0](support_feat), mask, return_attn=True)
        elif idx < self.stack_ids[2]:
            return self.DCAMA_blocks[1](self.pe[1](query), self.pe[1](support_feat), mask, return_attn=True)
        else:
            return self.DCAMA_blocks[2](self.pe[2](query), self.pe[2](support_feat), mask, return_attn=True)
        
    def mix_maps(self, coarse_masks1, coarse_masks2, coarse_masks3):
        coarse_masks1_conv = self.conv1(coarse_masks1) 
        coarse_masks2_conv = self.conv2(coarse_masks2)
        coarse_masks3_conv = self.conv3(coarse_masks3)

        # multi-scale cascade (pixel-wise addition)
        coarse_masks1_conv = F.interpolate(coarse_masks3_conv, coarse_masks2_conv.size()[-2:], mode='bilinear', align_corners=True)
        mix = coarse_masks1_conv + coarse_masks2_conv
        mix = self.conv4(mix)

        mix = F.interpolate(mix, coarse_masks3_conv.size()[-2:], mode='bilinear', align_corners=True)
        mix = mix + coarse_masks3_conv
        mix = self.conv5(mix)
        return mix
        
    def upsample_and_classify(self, feature_mix):
        # mixer blocks forward
        mix1 = self.mixer1(feature_mix)
        upsample_size = (mix1.size(-1) * 2,) * 2
        mix1 = F.interpolate(mix1, upsample_size, mode='bilinear', align_corners=True)
        mix2 = self.mixer2(mix1)
        upsample_size = (mix2.size(-1) * 2,) * 2
        mix2 = F.interpolate(mix2, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.mixer3(mix2)
        return logit_mask, mix2, mix1
        
    def skip_concat_features(self, mix, query_feats, support_feats, nshot):
        sf1, sf0 = None, None
        
        if self.concat_support:
            # skip connect 1/8 and 1/4 features (concatenation)
            if nshot == 1:
                support_feat = support_feats[self.stack_ids[1] - 1]
            else:
                support_feat = torch.stack([support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]).max(dim=0).values
                sf1 = support_feat.clone()
                mix = torch.cat((mix, query_feats[self.stack_ids[1] - 1], support_feat), 1)
        else:
            mix = torch.cat((mix, query_feats[self.stack_ids[1] - 1]), 1)

        upsample_size = (mix.size(-1) * 2,) * 2
        mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True)
        
        if self.concat_support:
            if nshot == 1:
                support_feat = support_feats[self.stack_ids[0] - 1]
            else:
                support_feat = torch.stack([support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]).max(dim=0).values
            sf0 = support_feat.clone()
            mix = torch.cat((mix, query_feats[self.stack_ids[0] - 1], support_feat), 1)
        else:
            mix = torch.cat((mix, query_feats[self.stack_ids[0] - 1]), 1)
        return mix, sf0, sf1

    def forward(self, query_feats, support_feats, support_mask, nshot=1):       
        # Main logic
        coarse_masks, attns, fg_raw_outs, bg_raw_outs = [], [], [], []
        for idx, query_feat in enumerate(query_feats):
            if idx < self.stack_ids[0]: continue

            bsz, ch, ha, wa = query_feat.size()
            query, support_feat, mask = reshape_and_prepare_features(query_feat, support_feats, support_mask, nshot, idx)
            coarse_mask, attn, fg_raw_out, bg_raw_out = self.process_dcama_blocks(query, support_feat, mask, idx)
            
            coarse_masks.append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, 1, ha, wa))
            attns.append(attn)
            fg_raw_outs.append(fg_raw_out)
            bg_raw_outs.append(bg_raw_out)

        coarse_masks1 = stack_and_reshape_features(coarse_masks, self.stack_ids, self.stack_ids[2], self.stack_ids[3])
        fg_raw_out1 = stack_and_reshape_features(fg_raw_outs, self.stack_ids, self.stack_ids[2], self.stack_ids[3])
        bg_raw_out1 = stack_and_reshape_features(bg_raw_outs, self.stack_ids, self.stack_ids[2], self.stack_ids[3])
        
        coarse_masks2 = stack_and_reshape_features(coarse_masks, self.stack_ids, self.stack_ids[1], self.stack_ids[2])
        fg_raw_out2 = stack_and_reshape_features(fg_raw_outs, self.stack_ids, self.stack_ids[1], self.stack_ids[2])
        bg_raw_out2 = stack_and_reshape_features(bg_raw_outs, self.stack_ids, self.stack_ids[1], self.stack_ids[2])
        
        coarse_masks3 = stack_and_reshape_features(coarse_masks, self.stack_ids, self.stack_ids[0], self.stack_ids[1])
        fg_raw_out3 = stack_and_reshape_features(fg_raw_outs, self.stack_ids, self.stack_ids[0], self.stack_ids[1])
        bg_raw_out3 = stack_and_reshape_features(bg_raw_outs, self.stack_ids, self.stack_ids[0], self.stack_ids[1])

        coarse_masks1, coarse_masks2, coarse_masks3 = refine_coarse_maps(
            [coarse_masks1, coarse_masks2, coarse_masks3],
            [self.alpha, self.beta, self.gamma]
        )
        
        coarse_masks = boost_coarse_map(
            [*[c.unsqueeze(0) for c in coarse_masks1[0]], *[c.unsqueeze(0) for c in coarse_masks2[0]], *[c.unsqueeze(0) for c in coarse_masks3[0]]],
            self.boost_alpha, self.boost_index
        )
        
        coarse_masks1 = torch.cat(coarse_masks[:coarse_masks1.shape[1]], dim=1).view(coarse_masks1.size())
        coarse_masks2 = torch.cat(coarse_masks[coarse_masks1.shape[1]:coarse_masks1.shape[1]+coarse_masks2.shape[1]], dim=1).view(coarse_masks2.size())
        coarse_masks3 = torch.cat(coarse_masks[coarse_masks1.shape[1]+coarse_masks2.shape[1]:], dim=1).view(coarse_masks3.size())

        mix = self.mix_maps(coarse_masks1, coarse_masks2, coarse_masks3)
        pre_mix = mix.clone()
        mix, sf0, sf1 = self.skip_concat_features(mix, query_feats, support_feats, nshot)
        logit_mask, mix2, mix1 = self.upsample_and_classify(mix)

        return {
            ResultDict.LOGITS: logit_mask,
            ResultDict.ATTENTIONS: attns,
            ResultDict.FG_RAW_ATTN_OUTS: [fg_raw_out1, fg_raw_out2, fg_raw_out3],
            ResultDict.BG_RAW_ATTN_OUTS: [bg_raw_out1, bg_raw_out2, bg_raw_out3],
            ResultDict.PRE_MIX: pre_mix,
            ResultDict.MIX: mix,
            ResultDict.MIX_1: mix1,
            ResultDict.MIX_2: mix2,
            ResultDict.COARSE_MASKS: [coarse_masks1, coarse_masks2, coarse_masks3],
            ResultDict.SUPPORT_FEAT_0: sf0,
            ResultDict.SUPPORT_FEAT_1: sf1,
            ResultDict.QUERY_FEAT_0: query_feats[self.stack_ids[0] - 1],
            ResultDict.QUERY_FEAT_1: query_feats[self.stack_ids[1] - 1],
            ResultDict.QUERY_FEATS: query_feats,
            ResultDict.SUPPORT_FEATS: support_feats,
            ResultDict.NSHOT: nshot
        }

    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        r""" bulid conv blocks """
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch,
                                                   kernel_size=ksz, stride=stride, padding=pad))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)


class DCAMAMultiClass(DCAMA):
    def __init__(self, backbone, pretrained_path, use_original_imgsize, image_size, concat_support=True, train_backbone=False):
        self.predict = None
        self.generate_class_embeddings = None
        self.image_size = image_size
        super().__init__(backbone, pretrained_path, use_original_imgsize, concat_support=concat_support, train_backbone=train_backbone)

    def _preprocess_masks(self, masks, dims):
        B, N, C, H, W = masks.size()
        # remove bg from masks
        masks = masks[:, :, 1:, ::]
        mask_size = 256

        # Repeat dims along class dimension
        support_dims = dims[:, 1:]
        repeated_dims = repeat(support_dims, "b n d -> (b n c) d", c=C)
        masks = rearrange(masks, "b n c h w -> (b n c) h w")

        # Remove padding from masks
        # pad_dims = [get_preprocess_shape(h, w, mask_size) for h, w in repeated_dims]
        # masks = [mask[:h, :w] for mask, (h, w) in zip(masks, pad_dims)]
        # masks = torch.cat(
        #     [
        #         F.interpolate(
        #             torch.unsqueeze(mask, 0).unsqueeze(0),
        #             size=(self.image_size, self.image_size),
        #             mode="nearest",
        #         )[0]
        #         for mask in masks
        #     ]
        # )
        return rearrange(masks, "(b n c) h w -> b n c h w", b=B, n=N)

    def forward(self, x):

        masks = self._preprocess_masks(
            x[BatchKeys.PROMPT_MASKS], x[BatchKeys.DIMS]
        )
        assert (
            masks.shape[0] == 1
        ), "Only tested with batch size = 1"
        results = []
        query = x[BatchKeys.IMAGES][:, :1]
        support = x[BatchKeys.IMAGES][:, 1:]
        # get logits for each class
        for c in range(masks.size(2)):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            n_shots = class_examples.sum().item()
            if n_shots > 0:
                class_input_dict = {
                    BatchKeys.IMAGES: torch.cat([query, support[class_examples].unsqueeze(0)], dim=1),
                    BatchKeys.PROMPT_MASKS: masks[:, :, c, ::][
                        class_examples
                    ].unsqueeze(0),
                }
                result = self.predict_mask_nshot(class_input_dict, n_shots)
            else:
                result = {
                    ResultDict.LOGITS: torch.full((1, 2, *query.shape[-2:]), -torch.inf, device=query.device)
                }
            results.append(result)

        results = {k: [d[k] for d in results] for k in results[0]}
        logits = results[ResultDict.LOGITS]
        logits = torch.stack(logits, dim=1)
        fg_logits = logits[:, :, 1, ::]
        bg_logits = logits[:, :, 0, ::]
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
        logits = torch.cat([bg_logits, fg_logits], dim=1)

        logits = self.postprocess_masks(logits, x["dims"])

        return {
            **results,
            ResultDict.LOGITS: logits,
        }

    def postprocess_masks(self, logits, dims):
        max_dims = torch.max(dims.view(-1, 2), 0).values.tolist()
        dims = dims[:, 0, :]  # get real sizes of the query images
        logits = [
            F.interpolate(
                torch.unsqueeze(logit, 0),
                size=dim.tolist(),
                mode="bilinear",
                align_corners=False,
            )
            for logit, dim in zip(logits, dims)
        ]

        logits = torch.cat(
            [
                F.pad(
                    mask,
                    (
                        0,
                        max_dims[1] - dims[i, 1],
                        0,
                        max_dims[0] - dims[i, 0],
                    ),
                    mode="constant",
                    value=float("-inf"),
                )
                for i, mask in enumerate(logits)
            ]
        )
        return logits

    def get_learnable_params(self, train_params):
        return self.parameters()
