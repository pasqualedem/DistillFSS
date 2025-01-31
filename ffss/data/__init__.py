import torch

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from ffss.data.transforms import Normalize, Resize

from ffss.data.coco import CocoLVISDataset
from ffss.data.coco_crop import CocoLVISCrop
from ffss.data.utils import get_mean_std

from ffss.data.weedmap import WeedMapTestDataset


TEST_DATASETS = {
    "test_weedmap": WeedMapTestDataset,
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
