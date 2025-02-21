import torch

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from fssweed.data.deepglobe import DatasetDeepglobe
from fssweed.data.evican import EVICAN
from fssweed.data.isic import DatasetISIC
from fssweed.data.lab2wild import Lab2Wild
from fssweed.data.lung import LungCancer
from fssweed.data.nucleus import Nucleus
from fssweed.data.phenobench import PhenoBenchTestDataset
from fssweed.data.pothole import Pothole
from fssweed.data.transforms import Normalize, Resize

from fssweed.data.coco import CocoLVISDataset
from fssweed.data.coco_crop import CocoLVISCrop
from fssweed.data.utils import get_mean_std

from fssweed.data.weedmap import WeedMapTestDataset


TEST_DATASETS = {
    "test_weedmap": WeedMapTestDataset,
    "test_phenobench": PhenoBenchTestDataset,
    "test_deepglobe": DatasetDeepglobe,
    "test_isic": DatasetISIC,
    "test_evican": EVICAN,
    "test_nucleus": Nucleus,
    "test_pothole": Pothole,
    "test_lab2wild": Lab2Wild,
    "test_lungcancer": LungCancer,
    # "test_dram": DramTestDataset,
    # "test_brain": BrainTestDataset,
    # "test_kvaris": KvarisTestDataset,
}


def map_collate(dataset):
    return dataset.collate_fn if hasattr(dataset, "collate_fn") else None


def get_preprocessing(params):
    
    preprocess_params = params.get("preprocess", {})
    size = preprocess_params["image_size"]
    mean = preprocess_params.get("mean", "default")
    std = preprocess_params.get("std", "default")
    mean, std = get_mean_std(mean, std)
    return Compose(
            [
                Resize(size=(size, size)),
                ToTensor(),
                Normalize(mean, std),
            ]
        )

def get_testloaders(dataset_args, dataloader_args):
    preprocess = get_preprocessing(dataset_args)

    datasets_params = dataset_args.get("datasets")

    test_datasets_params = {
        k: v for k, v in datasets_params.items() if k.startswith("test_")
    }

    if test_datasets_params:
        test_datasets = {
            dataset: TEST_DATASETS[dataset](**params, preprocess=preprocess)
            for dataset, params in test_datasets_params.items()
        }
        test_dataloaders = {
            name: DataLoader(
                dataset=data,
                **dataloader_args,
                collate_fn=map_collate(data),
            )
            for name, data in test_datasets.items()
        }
    else:
        test_dataloaders = None
    return test_dataloaders
