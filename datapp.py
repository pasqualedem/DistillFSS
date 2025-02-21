import streamlit as st

from fssweed.data import get_preprocessing, get_testloaders
from fssweed.data.utils import BatchKeys

import lovely_tensors as lt

from fssweed.utils.segmentation import create_rgb_segmentation, unnormalize;

lt.monkey_patch()

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
            },
            "test_isic": {"datapath": "data", "prompt_images": []},
            "test_deepglobe": {"datapath": "data", "prompt_images": []},
            "test_evican": {"datapath": "data", "prompt_images": []},
            "test_nucleus": {"datapath": "data", "prompt_images": []},
            "test_pothole": {"datapath": "data", "prompt_images": []},
            "test_lab2wild": {"datapath": "data", "prompt_images": []},
            "test_lungcancer": {"datapath": "data", "prompt_images": []},
            "test_kvasir": {"datapath": "data", "prompt_images": []},
        },
    },
}

@st.cache_data
def get_data(support_images):
    preprocess = get_preprocessing(parameters["dataset"])
    test_loaders = get_testloaders(
        parameters["dataset"],
        parameters["dataloader"]
    )
    return test_loaders


test_datasets = get_data(support_images=None)

test_datasets_list = parameters["dataset"]["datasets"].keys()
test_dataset_selected = st.sidebar.selectbox(
    "Test dataset",
    list(test_datasets_list),
    index=0,
)

test_dataset = test_datasets[test_dataset_selected].dataset

train_len = test_dataset.train_len()
test_len = len(test_dataset)

split = st.sidebar.selectbox(
    "Split",
    ["train", "test"],
    index=0,
)

selected_idx = st.sidebar.slider(
    "Index",
    min_value=0,
    max_value=train_len if split == "train" else test_len,
    value=0,
    step=1,
)
num_samples = st.sidebar.slider(
    "Number of samples",
    min_value=1,
    max_value=100,
    value=1,
    step=1,
)


for i in range(num_samples):
    image_dict, gt = test_dataset.get_sample(selected_idx + i, split)

    image = unnormalize(image_dict[BatchKeys.IMAGES])
    gt = create_rgb_segmentation(gt.unsqueeze(0), num_classes=test_dataset.num_classes)
    name = image_dict[BatchKeys.IMAGE_IDS][0]

    st.write(name)
    col1, col2 = st.columns(2)
    col1.write(image.rgb.fig)
    col2.write(gt.rgb.fig)


