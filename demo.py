import lovely_tensors as lt
import streamlit as st

import pickle
from einops import rearrange, repeat
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TvT
import torch.nn as nn
import numpy as np
from PIL import Image
from torchmetrics import F1Score, MetricCollection
from torchvision.transforms.functional import resize
from sklearn.decomposition import PCA


import numpy as np
import torch

from fssweed.utils.segmentation import create_rgb_segmentation, unnormalize
from fssweed.utils.utils import (
    ResultDict,
    StrEnum,
    torch_dict_load,
    torch_dict_save,
    to_device,
)
from fssweed.models import build_model
from fssweed.data import get_preprocessing, get_testloaders
from fssweed.data.utils import (
    AnnFileKeys,
    PromptType,
    BatchKeys,
    merge_dicts,
    min_max_scale,
    sum_scale,
)
from fssweed.substitution import Substitutor
import matplotlib.pyplot as plt
from tqdm import tqdm


lt.monkey_patch()

PROMPT_IMAGES = [
    # 'frame0009_2.png',
    'frame0021_2.png',
    "frame0033_3.png",
    'frame0034_1.png',
    'frame0048_0.png',
]


parameters = {
    "dataloader": {"num_workers": 0},
    "dataset": {
        "preprocess": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "image_size": 384,
        },
        "datasets": {
            "test_weedmap": {
                "train_root": "../Datasets/WeedMap/0_rotations_processed_003_test/RedEdge/000",
                "test_root": "../Datasets/WeedMap/0_rotations_processed_003_test/RedEdge/003",
                "prompt_images": None,
                "remove_black_images": False,
            }
        },
    },
    "model": {
        "name": "dcama",
        "backbone": "swin",
        "backbone_checkpoint": "checkpoints/swin_base_patch4_window12_384.pth",
        "model_checkpoint": "checkpoints/swin_fold0_pascal_modcross_soft.pt",
        # 'model_checkpoint': "checkpoints/f4z7ghu7.pt",
        "concat_support": False,
        "image_size": 384,
        "train_backbone": True,
    }
}

@st.cache_data
def get_data(support_images):
    parameters["dataset"]["datasets"]["test_weedmap"]["prompt_images"] = support_images
    preprocess = get_preprocessing(parameters["dataset"])
    test_loaders = get_testloaders(
        parameters["dataset"],
        parameters["dataloader"]
    )
    weedmap = test_loaders["test_weedmap"]
    return weedmap
    
    
@st.cache_data
def get_model(use_pe, device):
    parameters["model"]["pe"] = use_pe
    model = build_model(parameters["model"])
    model.to(device)
    model.eval()
    return model

def pred_layer(model, coarse_masks1, coarse_masks2, coarse_masks3, query_feats):
    mix = model.model.mix_maps(coarse_masks1, coarse_masks2, coarse_masks3)
    mix, _, _ = model.model.skip_concat_features(mix, query_feats, None, None)
    logit_mask, _, _ = model.model.upsample_and_classify(mix)
    return logit_mask


def feature_ablation(model, result, chosen_class, selected_x, selected_y):
    query_feats = result[ResultDict.QUERY_FEATS][chosen_class]
    coarse_masks = tuple(result[ResultDict.COARSE_MASKS][chosen_class])
    with torch.no_grad():
        orig_out = pred_layer(model, *coarse_masks, query_feats)[:, :, selected_x, selected_y]
    diffs = []
    for i in range(len(coarse_masks)):
        for j in range(coarse_masks[i].shape[1]):
            new_input = coarse_masks[i].clone()
            new_input[:, j] = 0
            new_expl_input = [*coarse_masks[0:i], *[new_input], *coarse_masks[i+1:]]
            with torch.no_grad():
                new_out = pred_layer(model, *new_expl_input, query_feats)[:, :, selected_x, selected_y]
            diffs.append(orig_out - new_out)
    
    abl_attr = sum_scale(torch.stack([torch.abs(diff[0, 1]) for diff in diffs]))
    return abl_attr

def explain(model, input_dict, result):
    num_classes = 2
    selected_x = 56
    selected_y = 65
    
    
    attns = result[ResultDict.ATTENTIONS]
    masks = input_dict[BatchKeys.PROMPT_MASKS]
    flag_examples = input_dict[BatchKeys.FLAG_EXAMPLES]
    target_shape = input_dict[BatchKeys.IMAGES][:, 0].shape[2:]
    
    for chosen_class in range(num_classes):

        st.write(f"#### Class {chosen_class+1} interpretation")
        class_attns = attns[chosen_class]
        class_examples = flag_examples[:, :, chosen_class + 1]
        mask = masks[:, :, chosen_class+1, ::][class_examples]
        support_mask = resize((result[ResultDict.LOGITS].argmax(dim=1) == chosen_class+1), target_shape, interpolation=TvT.InterpolationMode.NEAREST).float()
        support_mask = 2*support_mask - 1
        support_mask = TvT.gaussian_blur(support_mask, (9,9), sigma=5)

        level_contributions = []
        for level_attn in class_attns:

            level_attn = level_attn.mean(dim=1)

            hw = level_attn.shape[-2]
            h = w = int(hw ** 0.5)
            mask_current = resize(mask, (h, w), interpolation=TvT.InterpolationMode.NEAREST)
            mask_current = rearrange(mask_current, "n h w -> 1 1 n (h w)")
            level_attn = F.softmax(level_attn, dim=-1)
            level_attn = rearrange(level_attn, "b (h1 w1) (n h2 w2) -> (b h2 w2 n) h1 w1", n=mask_current.shape[2], h1=h, w1=w, h2=h, w2=w)
            level_attn = resize(level_attn, target_shape)
            level_attn = level_attn[:, selected_y, selected_x]
            level_attn = rearrange(level_attn, "(b h2 w2 n) -> (b n) h2 w2", h2=h, w2=w, n=mask_current.shape[2])
            resized_level_attn = resize(level_attn, target_shape, interpolation=TvT.InterpolationMode.BILINEAR, antialias=False)
            level_contributions.append(resized_level_attn)

        contrib_seq = torch.stack(level_contributions)
        contrib = contrib_seq.mean(dim=0)
        
        with st.spinner("Doing Feature Ablation..."):
            cmask_contrib = feature_ablation(model, result, chosen_class, selected_x, selected_y)
        st.write(contrib_seq)
        with st.expander("Full masks"):
            st.write(min_max_scale(contrib_seq).chans(cmap="viridis").fig)
            
        col1, col2, col3 = st.columns(3)
    
        col1.write(f"Mean contribution")
        contrib = min_max_scale(contrib)
        col1.write(contrib)
        col1.write((contrib*support_mask).chans(cmap="viridis").fig)
        
        
        col2.write(f"Weighted contribution")
        cmask_contrib = rearrange(cmask_contrib, "c -> c 1 1 1")
        weighted_contrib = min_max_scale((contrib_seq * cmask_contrib).sum(dim=0))
        weighted_contrib = (weighted_contrib * support_mask)
        col2.write(weighted_contrib)
        col2.write(weighted_contrib.chans(cmap="viridis").fig)
        
        col3.write(f"Support mask")
        col3.write(support_mask)
        col3.write(support_mask.chans(cmap="viridis").fig)
        
        
def feature_map_pca_heatmap(feature_map):
    """
    Given a feature map of shape (D, H, W), performs PCA along the feature dimension
    and returns a heatmap based on the first principal component.
    
    Args:
        feature_map (torch.Tensor): Input tensor of shape (D, H, W).
    
    Returns:
        heatmap (torch.Tensor): Heatmap of shape (H, W) based on the first principal component.
    """
    # Check the input dimensions
    D, H, W = feature_map.shape
    feature_map_reshaped = feature_map.view(D, -1).T  # Reshape to (H*W, D)

    # Convert to numpy for PCA
    feature_map_np = feature_map_reshaped.cpu().numpy()
    
    # Perform PCA
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(feature_map_np)
    
    # Reshape the result back to (H, W) and convert it to a tensor
    heatmap = torch.tensor(principal_component).view(H, W)
    
    # Normalize the heatmap for better visualization
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    return heatmap
        
def attention_summary(result, masks, flag_examples):
    attns_class = result[ResultDict.ATTENTIONS]
    pre_mix = result[ResultDict.PRE_MIX]
    mix = result[ResultDict.MIX]
    mix1 = result[ResultDict.MIX_1]
    mix2 = result[ResultDict.MIX_2]
    sf1  = result[ResultDict.SUPPORT_FEAT_1]
    sf0  = result[ResultDict.SUPPORT_FEAT_0]
    qf0  = result[ResultDict.QUERY_FEAT_0]
    qf1  = result[ResultDict.QUERY_FEAT_1]
    coarse_masks = result[ResultDict.COARSE_MASKS]
    
    masks = masks[:, :, 1:, ::]
    st.write("## Model Summary")
    target_size = 48
    for j, attns in enumerate(attns_class):
        attns = [
            attn.mean(dim=1) for attn in attns
        ]
        class_examples = flag_examples[:, :, j + 1]
        mask = masks[:, :, j, ::][class_examples]
        outs = []
        for attn in attns:
            hw = attn.shape[-1]
            h = w = int(hw ** 0.5)
            # resize mask to attn
            mask = resize(mask, (h, w), interpolation=TvT.InterpolationMode.NEAREST)
            mask = rearrange(mask, "1 h w -> 1 1 (h w)")
            attn = attn * mask
            attn = attn.sum(dim=-1)
            # attn = torch.matmul(attn, mask)
            attn = rearrange(attn, "1 (h w) -> 1 h w", h=h, w=w)
            attn = resize(attn, (target_size, target_size))
            outs.append(attn)
        out = torch.cat(outs).mean(dim=0)
        out = (out - out.min()) / (out.max() - out.min())
        
        cols = st.columns(4)
        with cols[0]:
            st.write("### Coarse Mask 1")
            st.write(coarse_masks[j][0][0])
            coarse1 = coarse_masks[j][0][0].mean(dim=0)
            st.write(coarse1.chans(scale=4).fig)
        with cols[1]:
            st.write("### Coarse Mask 2")
            st.write(coarse_masks[j][1][0])
            coarse2 = coarse_masks[j][1][0].mean(dim=0)
            st.write(coarse2.chans(scale=4).fig)
        with cols[2]:
            st.write("### Coarse Mask 3")
            st.write(coarse_masks[j][2][0])
            coarse3 = coarse_masks[j][2][0].mean(dim=0)
            st.write(coarse3.chans(scale=4).fig)
        with cols[3]:
            st.write("### Coarse Mean")
            coarse3 = coarse_masks[j][2]
            coarse2 = F.interpolate(coarse_masks[j][1], coarse3.size()[-2:], mode='bilinear', align_corners=True)
            coarse1 = F.interpolate(coarse_masks[j][2], coarse3.size()[-2:], mode='bilinear', align_corners=True)
            coarse = torch.cat([coarse1, coarse2, coarse3], dim=1)
            coarse_mean = coarse.mean(dim=1)
            st.write(coarse_mean)
            st.write(coarse_mean.chans(scale=4).fig)
            
        with st.expander("Full Coarse Maps"):
            st.write(coarse.chans(scale=4).fig)
        
        cols = st.columns(4)
        with cols[0]:
            st.write("### Pre-mix")
            st.write(pre_mix[j][0])
            pre_mix_pca = feature_map_pca_heatmap(pre_mix[j][0])
            st.write(pre_mix_pca.chans(scale=4).fig)
        with cols[1]:
            st.write("### Mix")
            st.write(mix[j][0])
            coarse1 = feature_map_pca_heatmap(mix[j][0])
            st.write(coarse1.chans(scale=4).fig)
        with cols[2]:
            st.write("### Mix Out 1")
            st.write(mix1[j][0])
            coarse1 = feature_map_pca_heatmap(mix1[j][0])
            st.write(coarse1.chans(scale=4).fig)
        with cols[3]:
            st.write("### Mix Out 2")
            st.write(mix2[j][0])
            coarse1 = feature_map_pca_heatmap(mix2[j][0])
            st.write(coarse1.chans(scale=4).fig)
                
        with st.expander("Query Features"):
            cols = st.columns(2)
            with cols[0]:
                st.write("### Query Feature 0")
                qf0_pca = feature_map_pca_heatmap(qf0[j][0])
                st.write(qf0_pca.chans(scale=4).fig)
            with cols[1]:
                st.write("### Query Feature 1")
                qf1_pca = feature_map_pca_heatmap(qf1[j][0])
                st.write(qf1_pca.chans(scale=4).fig)

        with st.expander("Support Features"):
            cols = st.columns(2)
            with cols[0]:
                st.write("### Support Feature 0")
                sf0_pca = feature_map_pca_heatmap(sf0[j][0])
                st.write(sf0_pca.chans(scale=4).fig)
            with cols[1]:
                st.write("### Support Feature 1")
                sf1_pca = feature_map_pca_heatmap(sf1[j][0])
                st.write(sf1_pca.chans(scale=4).fig)
    
def main():
    with st.sidebar:
        st.write("### Parameters")
        device = st.selectbox("Device", ["cpu", "cuda"], index=0)
        support_images = st.multiselect(
            "Support images", PROMPT_IMAGES, default=PROMPT_IMAGES[:1]
        )
        use_pe = st.checkbox("Use PE", value=True)
    
    model = get_model(use_pe, device)
    data = get_data(support_images)
    
    examples = data.dataset.extract_prompts()
    examples = to_device(examples, device)
    
    st.write("### Support images")
    st.write(unnormalize(examples[BatchKeys.IMAGES]).rgb.fig)
    
    if "iterator" not in st.session_state:
        st.session_state["iterator"] = iter(data)
    if "batch" not in st.session_state:
        st.session_state["batch"] = next(st.session_state["iterator"])
    if st.button("Next"):
        st.session_state["batch"] = next(st.session_state["iterator"])
    batch = st.session_state["batch"]
    
    image_dict, gt = batch
    
    st.write("Query Image")
    st.write(unnormalize(image_dict[BatchKeys.IMAGES][:, 0]).rgb.fig)
    
    input_dict = to_device(merge_dicts(prompts=examples, imgs=image_dict), device)
    gt = to_device(gt, device)
    
    if st.button("Predict"):
        with torch.no_grad():
            result = model(input_dict)
            
        st.session_state["result"] = result
        
    if "result" in st.session_state:
        result = st.session_state["result"]
        outputs = torch.argmax(result[ResultDict.LOGITS], dim=1)
        pred_col, gt_col = st.columns(2)
        
        if st.checkbox("Show attention summary"):
            attention_summary(result, input_dict[BatchKeys.PROMPT_MASKS], input_dict[BatchKeys.FLAG_EXAMPLES])
        
        pred_col.write("Predictions")
        pred_col.write(create_rgb_segmentation(outputs, num_classes=3).rgb.fig)
        
        gt_col.write("Ground Truth")
        gt_col.write(create_rgb_segmentation(gt, num_classes=3).rgb.fig)
        
        explain(model, input_dict, result)
    
if __name__ == "__main__":
    main()