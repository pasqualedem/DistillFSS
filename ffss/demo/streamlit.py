import itertools
import pickle
from einops import rearrange
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

import numpy as np
import torch

import lovely_tensors as lt

from ffss.demo.preprocess import denormalize, preprocess_support_set, preprocess_to_batch
from ffss.demo.utils import (
    COLORS,
    TEXT_COLORS,
    SupportExample,
    get_color_from_class,
    open_rgb_image,
    debug_write,
    retrieve_models,
    take_elem_from_batch,
)
from ffss.utils.utils import ResultDict, StrEnum, torch_dict_load, torch_dict_save
from ffss.models import MODEL_REGISTRY

lt.monkey_patch()

from ffss.data import get_preprocessing
from ffss.data.utils import (
    AnnFileKeys,
    PromptType,
    BatchKeys,
)

from ffss.demo.visualize import (
    crop_seg,
    draw_all,
    draw_masks,
    feature_map_pca_heatmap,
    get_image,
    plot_seg,
)
from ffss.demo.builtin import built_in_dataset, predict

from torchvision.transforms.functional import resize
import torchvision.transforms as TvT


IMG_DIR = "data/coco/train2017"
ANNOTATIONS_DIR = "data/annotations/instances_val2017.json"
EMBEDDINGS_DIR = "data/coco/embeddings"
MAX_EXAMPLES = 30
VIT_B_SAM_PATH = "checkpoints/sam_vit_b_01ec64.pth"

SIZE = 384
PROMPT_SIZE = 512
CUSTOM_PREPROCESS = False

dataset_params = {
    "preprocess": {
        "image_size": SIZE,
    }
}

preprocess = get_preprocessing(dataset_params)


dcama_versions = {
    "coco fold0": {
        "model_checkpoint":"checkpoints/swin_fold0.pt"
    },
    "coco fold3": {
        "model_checkpoint":"checkpoints/swin_fold3.pt"
    },
    "pascal fold0": {
        "model_checkpoint":"checkpoints/swin_fold0_pascal.pt"
    },
    "pascal fold0 modified": {
        "model_checkpoint":"checkpoints/swin_fold0_pascal_modified.pt",
        "concat_support": False
    },
    "pascal fold0 loss": {
        "model_checkpoint":"checkpoints/swin_fold0_pascal_modloss.pt",
        "concat_support": True
    },
    "pascal fold0 loss sigmoid": {
        "model_checkpoint":"checkpoints/swin_fold0_pascal_modsigmoid.pt",
        "concat_support": False
    },
    "pascal fold0 loss cross entropy": {
        "model_checkpoint":"checkpoints/swin_fold0_pascal_modcross.pt",
        "concat_support": False
    },
    "pascal fold0 loss cross entropy soft": {
        "model_checkpoint":"checkpoints/swin_fold0_pascal_modcross_soft.pt",
        "concat_support": False
    },
}

def load_dcama(version):
    params = dcama_versions[version]
    return MODEL_REGISTRY['dcama'](
    backbone_checkpoint="checkpoints/swin_base_patch4_window12_384.pth",
    **params
)


class SS(StrEnum):
    SUPPORT_SET = "support_set"
    CLASSES = "classes"
    RESULT = "result"


@st.cache_resource
def load_model(device, version):
    return load_dcama(version).to(device).eval()


def reset_support(idx):
    if idx is None:
        st.session_state[SS.SUPPORT_SET] = []
        st.session_state[SS.CLASSES] = []
        return
    st.session_state[SS.SUPPORT_SET].pop(idx)   
    
    
def type_classes():
    st.write("Choose the classes you want to segment in the image")
    cols = st.columns(2)
    with cols[0]:
        new_class = st.text_input("Type and press enter to add a class")
        classes = st.session_state.get(SS.CLASSES, [])
        if new_class not in classes and new_class != "":
            classes.append(new_class)
        st.session_state[SS.CLASSES] = classes
    with cols[1]:
        if st.button("Reset"):
            reset_support(None)
            classes = []
    st.write("Classes:", ", ".join(classes))


def build_support_set():
    if st.session_state.get(SS.SUPPORT_SET, None) is None:
        st.session_state[SS.SUPPORT_SET] = []
    st.write("## Upload and annotate the support images")

    support_image = st.file_uploader(
        "If you want, you can upload and annotate another support image",
        type=["png", "jpg", "webp"],
        key="support_image",
    )
    if support_image is not None:
        add_support_image(support_image)


def add_support_image(support_image):
    support_image = open_rgb_image(support_image)
    st.write(
        "Use the annotation tool to annotate the image with bounding boxes, click Complete when you are done"
    )
    tab1, tab2 = st.tabs(["Annotate", "Load mask"])
    with tab1:
        cols = st.columns(3)
        with cols[0]:
            selected_class = st.selectbox(
                "Select the class you want to annotate",
                st.session_state[SS.CLASSES],
                key="selectbox_class",
            )
            prompt_type = "polygon"
        with cols[1]:
            edit_mode = st.checkbox("Edit annotations", key="edit_mode")
        with cols[2]:
            focused = st.checkbox("Focused")
        edit_mode = "transform" if edit_mode else prompt_type
        selected_class_color_f, selected_class_color_st, selected_class_color_focused = get_color_from_class(
            st.session_state[SS.CLASSES], selected_class
        )
        shape = PROMPT_SIZE, PROMPT_SIZE
        selected_class_color_f = selected_class_color_focused if focused else selected_class_color_f
        selected_class_color_st = "white" if focused else selected_class_color_st
        results = st_canvas(
            fill_color=selected_class_color_f,  # Fixed fill color with some opacity
            stroke_color=selected_class_color_st,  # Fixed stroke color with full opacity
            background_image=support_image,
            drawing_mode=edit_mode,
            key="input_prompt_detection",
            width=shape[1],
            height=shape[0],
            stroke_width=2,
            update_streamlit=False,
        )
    with tab2:
        st.write("Load a mask to segment the image")
        st.write("Select the color for each class (background is always black)")
        color_map = {}
        color_cols = st.columns(len(st.session_state[SS.CLASSES]))
        for i, cls in enumerate(st.session_state[SS.CLASSES]):
            with color_cols[i]:
                color = st.selectbox(
                    f"Select color for {cls}",
                    TEXT_COLORS,
                    key=f"color_{cls}",
                    index=i,
                )
                color_map[i] = np.array(COLORS[TEXT_COLORS.index(color)])
        mask = st.file_uploader(
            "Upload the mask",
            type=["png", "jpg"],
            accept_multiple_files=False,
            key="mask_support",
        )
        mask = np.array(open_rgb_image(mask)) if mask is not None else None
        (
            st.image(mask, caption="Mask", use_column_width=True)
            if mask is not None
            else None
        )
        if mask is not None:
            results = {
                "mask": mask,
                "color_map": color_map,
            }
    if results is not None and st.button("Add Support Image"):
        add_annotated_support_image(support_image, results, shape)


def add_annotated_support_image(support_image, results, shape):
    example = SupportExample(support_image=support_image)
    if hasattr(results, "json_data") and results.json_data is not None:
        st.write("Extracting prompts from canvas")
        example.prompts = results.json_data
        example.reshape = shape
    if isinstance(results, dict) and "mask" in results:
        st.write("Extracting prompts from mask")
        example.prompts = results
        example.reshape = shape
    st.session_state[SS.SUPPORT_SET].append(example)
    st.session_state.pop("input_prompt_detection", None)
    st.session_state.pop("mask_support", None)
    st.session_state.pop("support_image", None)
    st.write("Support image added")


def preview_support_set(batch):
    num_images = len(batch[BatchKeys.IMAGES][0])
    preview_cols = st.columns(num_images)
    for i in range(num_images):
        img = batch[BatchKeys.IMAGES][0][i]
        masks = batch[BatchKeys.PROMPT_MASKS][0][i]
        img = get_image(img)
        img = draw_masks(img, masks=masks, colors=COLORS)
        with preview_cols[i]:
            if st.button(f"Remove Image {i+1}"):
                reset_support(i)
            st.image(img, caption=f"Support Image {i+1}", use_column_width=True)
            
            
def rect_explanation(query_image_pt, attns_class, masks, flag_examples, num_sample):
    st.write("## Rectangle Explanation")
    
    masks = masks[:, :, 1:, ::]
    shape = query_image_pt.shape[-2:]
    query_image_pt = denormalize(query_image_pt)
    query_image_pt = query_image_pt.squeeze(0).permute(1, 2, 0).cpu().numpy()
    query_image_pt = Image.fromarray((query_image_pt * 255).astype(np.uint8))

    results = st_canvas(
        fill_color="white",  # Fixed fill color with some opacity
        stroke_color="black",  # Fixed stroke color with full opacity
        background_image=query_image_pt,
        drawing_mode="rect",
        key=f"explain_input_{num_sample}",
        width=shape[1],
        height=shape[0],
        stroke_width=2,
        update_streamlit=False,
    )
    if results is not None and results.json_data is not None:
        if rects := [
        (rect["left"], rect["top"], rect['width'], rect["height"]) for rect in results.json_data["objects"]
        ]:
            rect = rects[0]
            rect = torch.tensor(rect)

            target_size = 48
            st.write(rect)

            for j, attns in enumerate(attns_class):                    
                attns = [
                    attn.mean(dim=1) for attn in attns
                ]
                class_examples = flag_examples[:, :, j + 1]
                mask = masks[:, :, j, ::][class_examples]

                rect_attns = []
                for attn in attns:
                    hw = attn.shape[-1]
                    h = w = int(hw ** 0.5)
                    scaled_rect = (rect * h / shape[0]).int()
                    mask_current = resize(mask, (h, w), interpolation=TvT.InterpolationMode.NEAREST)
                    mask_current = rearrange(mask_current, "1 h w -> 1 1 (h w)")
                    attn = attn * mask_current
                    attn = rearrange(attn, "b (h1 w1) (h2 w2) -> b h1 w1 h2 w2", h1=h, w1=w, h2=h, w2=w)
                    x0 = scaled_rect[0]
                    x1 = scaled_rect[0] + max(scaled_rect[2], 1)
                    y0 = scaled_rect[1]
                    y1 = scaled_rect[1] + max(scaled_rect[3], 1)
                    rect_attn = attn[0, x0:x1, y0:y1]
                    rect_attn = rect_attn.mean(dim=(0, 1))
                    rect_attn_norm = (rect_attn - rect_attn.min()) / (rect_attn.max() - rect_attn.min())
                    rect_attns.append((rect_attn, rect_attn_norm))
                st.write(f"Attention for class {j}")
                
                st.write(f"Attention summary for rect")
                mean_attn = torch.cat([
                    resize(attn[0].unsqueeze(0), (target_size, target_size)) for attn in rect_attns
                ]).mean(dim=0)
                mean_attn = (mean_attn - mean_attn.min()) / (mean_attn.max() - mean_attn.min())
                st.write(mean_attn.chans.fig)
                
                st.write("Each channel attention")
                n_cols = 4
                cols = st.columns(n_cols)
                for k, (_, attn) in enumerate(rect_attns):
                    n_col = k % n_cols
                    cols[n_col].write(attn.chans.fig)
                    

def get_raw_attn(result):
    fg_raws = result.get(ResultDict.FG_RAW_ATTN_OUTS, None)
    bg_raws = result.get(ResultDict.BG_RAW_ATTN_OUTS, None)
    if fg_raws is None:
        return None
    fg_raw_list = []
    bg_raw_list = []
    for raw_attn1, raw_attn2, raw_attn3 in fg_raws:
        raw_attn1 = F.interpolate(raw_attn1, raw_attn3.size()[-2:], mode='bilinear', align_corners=True)
        raw_attn2 = F.interpolate(raw_attn2, raw_attn3.size()[-2:], mode='bilinear', align_corners=True)
        raw_attn3 = F.interpolate(raw_attn3, raw_attn3.size()[-2:], mode='bilinear', align_corners=True)
        raw = torch.cat([raw_attn1, raw_attn2, raw_attn3], dim=1).mean(dim=1).unsqueeze(1)
        fg_raw_list.append(raw)
    for raw_attn1, raw_attn2, raw_attn3 in bg_raws:
        raw_attn1 = F.interpolate(raw_attn1, raw_attn3.size()[-2:], mode='bilinear', align_corners=True)
        raw_attn2 = F.interpolate(raw_attn2, raw_attn3.size()[-2:], mode='bilinear', align_corners=True)
        raw_attn3 = F.interpolate(raw_attn3, raw_attn3.size()[-2:], mode='bilinear', align_corners=True)
        raw = torch.cat([raw_attn1, raw_attn2, raw_attn3], dim=1).mean(dim=1).unsqueeze(1)
        bg_raw_list.append(raw)
    raw = [torch.cat((bg, fg), dim=1) for fg, bg in zip(fg_raw_list, bg_raw_list)]
    raw = [F.softmax(elem, dim=1) for elem in raw]
    fg_raw = [elem[:, 1] for elem in raw]
    bg_raw = [elem[:, 0] for elem in raw]
    return fg_raw, bg_raw
                    
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
    fg_raw_masks, bg_raw_masks = get_raw_attn(result)
    
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
            st.write("### FG Coarse Raw Mask Mean")
            st.write(fg_raw_masks[j])
            st.write(fg_raw_masks[j].chans(scale=4).fig)
        with cols[1]:
            st.write("### BG Coarse Raw Mask Mean")
            st.write(bg_raw_masks[j])
            st.write(bg_raw_masks[j].chans(scale=4).fig)
        with cols[2]:  
            st.write("### Attention Scores")
            st.write(out)
            st.write(out.chans(scale=4).fig)
        with cols[3]:
            st.write("### Pre-mix")
            st.write(pre_mix[j][0])
            pre_mix_pca = feature_map_pca_heatmap(pre_mix[j][0])
            st.write(pre_mix_pca.chans(scale=4).fig)
        cols = st.columns(3)
        with cols[0]:
            st.write("### Mix")
            st.write(mix[j][0])
            coarse1 = feature_map_pca_heatmap(mix[j][0])
            st.write(coarse1.chans(scale=4).fig)
        with cols[1]:
            st.write("### Mix Out 1")
            st.write(mix1[j][0])
            coarse1 = feature_map_pca_heatmap(mix1[j][0])
            st.write(coarse1.chans(scale=4).fig)
        with cols[2]:
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
        
def show_logits(logits):
    st.write("## Logits")
    cols = st.columns(logits.shape[1])
    logits = logits.softmax(dim=1)[0]
    for i, logit in enumerate(logits):
        with cols[i]:
            # logit = torch.clip(logit - 0.98, min=0.0, max=1.0)
            # logit = (logit - logit.min()) / (logit.max() - logit.min())
            st.write(logit)
            st.write(logit.chans.fig)
        
        
def dcama_personalization(model):
    cols = st.columns(5)
    with cols[0]:
        alpha = st.number_input("alpha", 0.0, value=1.0, format="%0.4f")
    with cols[1]:
        beta = st.number_input("beta", 0.0, value=1.0, format="%0.4f")
    with cols[2]:
        gamma = st.number_input("gamma", 0.0, value=1.0, format="%0.4f")
    with cols[3]:
        boost_alpha = st.number_input("boost_alpha", 0.0, value=0.0, format="%0.4f")
    with cols[4]:
        boost_index = st.number_input("boost_beta", 0, value=0)
    
    model.set_importance_levels(alpha, beta, gamma, boost_alpha, boost_index)
    
    
    # Select aggregation method
    aggregation_methods = ['sum', 'max', 'threshold', 'power', 'lse', 'sigmoid', 'hard']
    aggregation = st.selectbox("Select Aggregation Method", aggregation_methods)

    # Show inputs dynamically for hyperparameters
    kwargs = {}

    if aggregation == 'threshold':
        kwargs['threshold'] = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    if aggregation == 'power':
        kwargs['gamma'] = st.slider("Gamma", min_value=1.0, max_value=10.0, value=2.0, step=0.1)

    if aggregation == 'lse':
        kwargs['lambda_param'] = st.slider("Lambda Parameter", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    if aggregation == 'sigmoid':
        kwargs['tau'] = st.slider("Tau (Threshold)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        kwargs['k'] = st.slider("K (Steepness)", min_value=1, max_value=100, value=10, step=1)
        
    if st.checkbox("Smoothing"):
        kwargs['alpha'] = st.slider("Alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        
    kwargs["temperature"] = st.number_input("Temperature", 0.0, value=1.0, format="%0.4f")     
    model.set_attn_fn(aggregation=aggregation, **kwargs)
        
        
def try_it_yourself(model):
    st.write("Upload the image the you want to segment")
    query_images = st.file_uploader(
        "Choose an image", type=["png", "jpg", "webp"], accept_multiple_files=True
    )
    if len(query_images) > 0:
        images = [open_rgb_image(query_image) for query_image in query_images]
        with st.expander("Query Images"):
            cols = st.columns(len(query_images))
            for i, query_image in enumerate(query_images):
                image = open_rgb_image(query_image)
                with cols[i]:
                    # Save image in a temp file
                    st.image(image, caption=f"Query Image {i+1}", width=300)
                    
    support_batch = None
    type_classes()
    support_batch = st.file_uploader("Load support set from file") 
    if support_batch:
        support_batch = pickle.loads(support_batch.read())
    elif st.session_state.get(SS.CLASSES):
        build_support_set()
        if SS.SUPPORT_SET in st.session_state and len(st.session_state[SS.SUPPORT_SET]) > 0:
            focusing_factor = st.number_input("Focusing Factor", min_value=1, max_value=100, value=5)

            support_batch = preprocess_support_set(
                st.session_state[SS.SUPPORT_SET],
                list(range(len(st.session_state[SS.CLASSES]))),
                preprocess=preprocess,
                device=st.session_state.get("device", "cpu"),
                custom_preprocess=CUSTOM_PREPROCESS,
                focusing_factor=focusing_factor
            )
    if support_batch:
        preview_support_set(support_batch)
        cols = st.columns(2)
        with cols[0]:
            filename = st.text_input("Filename", "data.p")
        with cols[1]:
            st.download_button("Save Support Set", file_name=filename, data=pickle.dumps(support_batch))
        dcama_personalization(model)
    if (
        support_batch
        and SS.CLASSES in st.session_state
        and len(query_images) > 0
    ):
        batches = [
            preprocess_to_batch(
                image,
                support_batch.copy(),
                preprocess,
                device=st.session_state.get("device", "cpu"),
            )
            for image in images
        ]
        st.write(batches)
        st.write("## Predictions")
        if st.button("Predict"):
            st.session_state[SS.RESULT] = []
            progress = st.progress(0)
            for support_batch in batches:
                result = predict(model, support_batch)
                st.session_state[SS.RESULT].append(result)
                progress.progress((i + 1) / len(batches))
        if SS.RESULT in st.session_state:
            tabs = st.tabs([f"Query Image {i+1}" for i in range(len(batches))])
            for i, (support_batch, result) in enumerate(
                zip(batches, st.session_state[SS.RESULT])
            ):
                with tabs[i]:
                    pred = result[ResultDict.LOGITS].argmax(dim=1)
                    st.json(result, expanded=False)
                    plots, titles = plot_seg(
                        open_rgb_image(query_image),
                        pred,
                        COLORS,
                        dims=support_batch[BatchKeys.DIMS],
                        classes=st.session_state[SS.CLASSES],
                    )
                    cols = st.columns(2)
                    cols[0].image(
                        plots[0], caption=titles[0], use_column_width=True
                    )
                    cols[1].image(
                        plots[1], caption=titles[1], use_column_width=True
                    )
                attns_class = result.get(ResultDict.ATTENTIONS, None)
                pre_mix = result.get(ResultDict.PRE_MIX, None)
                if attns_class is not None:
                    query_image_pt = batches[i][BatchKeys.IMAGES][0, 0]
                    logits = crop_seg(result[ResultDict.LOGITS], support_batch[BatchKeys.DIMS])
                    show_logits(logits)
                    attention_summary(result, batches[i][BatchKeys.PROMPT_MASKS], batches[i][BatchKeys.FLAG_EXAMPLES])
                    rect_explanation(query_image_pt, attns_class, batches[i][BatchKeys.PROMPT_MASKS], batches[i][BatchKeys.FLAG_EXAMPLES], i)


def handle_gpu_memory(device):
    # Display GPU memory
    if device == "cuda":
        allocated = f"{torch.cuda.memory_allocated() / 1e9:.2f}"
        total = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}"
        st.progress(
            torch.cuda.memory_allocated()
            / torch.cuda.get_device_properties(0).total_memory,
            text=f"GPU memory allocated: {allocated} GB / {total} GB",
        )
    if st.button("Clear GPU cache") and device == "cuda":
        torch.cuda.empty_cache()


def main():
    st.set_page_config(layout="wide", page_title="Focused FSS")
    st.title("Focused FSS")
    st.sidebar.title("Settings")
    with st.sidebar:
        if cuda := torch.cuda.is_available():
            use_gpu = st.checkbox("Use GPU", True)
        device = "cuda" if cuda and use_gpu else "cpu"
        st.session_state["device"] = device
        # load model
        st.write("Working on device:", device)
        version = st.selectbox(label="Version", options=dcama_versions.keys())
        model = load_model(device, version)
        
        st.divider()
        st.json(st.session_state, expanded=False)
        handle_gpu_memory(device)
    tiy_tab, dataset_tab = st.tabs(["Try it yourself", "Built-in dataset"])
    with tiy_tab:
        try_it_yourself(model)
    with dataset_tab:
        built_in_dataset(model)


if __name__ == "__main__":
    main()
