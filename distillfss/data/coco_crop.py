import itertools
import os
import random
import warnings
from io import BytesIO
from typing import Any, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor, ToTensor
from distillfss.utils.logger import get_logger

import distillfss.data.utils as utils

from distillfss.data.utils import (
    AnnFileKeys,
    DataDict,
    convert_mask,
    xywh2xyxy,
)

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


class CocoLVISDataset(Dataset):
    """Dataset class for COCO and LVIS datasets."""

    def __init__(
        self,
        instances_path: str,
        img_dir: Optional[str] = None,
        preprocess=ToTensor(),
    ):
        """Initialize the dataset.

        Args:
            instances_path (str): Path to the instances json file.
            img_dir (Optional[str], optional): Path to the directory containing the images. Defaults to None.
            preprocess (_type_, optional): A preprocessing step to apply to the images. Defaults to ToTensor().
            image_size (int, optional): The size of the images. Defaults to 1024.
        """
        super().__init__()
        print(f"Loading dataset annotations from {instances_path}...")

        self.instances_path = instances_path

        self.img_dir = img_dir

        # load instances
        instances = utils.load_instances(self.instances_path)
        self.annotations = {
            x[AnnFileKeys.ID]: x for x in instances[AnnFileKeys.ANNOTATIONS]
        }
        self.categories = {
            x[AnnFileKeys.ID]: x for x in instances[AnnFileKeys.CATEGORIES]
        }

        # useful dicts
        (
            self.img_annotations,
            self.img2cat,
            self.img2cat_annotations,
            self.cat2img,
            self.cat2img_annotations,
        ) = self._load_annotation_dicts()

        # load image ids and info
        img2cat_keys = set(self.img2cat.keys())
        self.images = {
            x[AnnFileKeys.ID]: x
            for x in instances[AnnFileKeys.IMAGES]
            if x[AnnFileKeys.ID] in img2cat_keys
        }
        self.image_ids = list(self.images.keys())
        # processing
        self.preprocess = preprocess

    def _load_annotation_dicts(self) -> tuple[dict, dict, dict, dict, dict]:
        """Load useful annotation dicts.

        Returns:
            (dict, dict, dict, dict, dict): Returns four dictionaries:
                0. img_annotations: A dictionary mapping image ids to lists of annotations.
                1. img2cat: A dictionary mapping image ids to sets of category ids.
                2. img2cat_annotations: A dictionary mapping image ids to dictionaries mapping category ids to annotations.
                3. cat2img: A dictionary mapping category ids to sets of image ids.
                4. cat2img_annotations: A dictionary mapping category ids to dictionaries mapping image ids to annotations.
        """
        img_annotations = {}
        img2cat_annotations = {}
        cat2img_annotations = {}

        img2cat = {}
        cat2img = {}

        category_ids = set(self.categories.keys())

        for ann in self.annotations.values():
            if AnnFileKeys.ISCROWD in ann and ann[AnnFileKeys.ISCROWD] == 1:
                continue

            if ann[AnnFileKeys.CATEGORY_ID] not in category_ids:
                continue

            if ann[AnnFileKeys.IMAGE_ID] not in img_annotations:
                img_annotations[ann[AnnFileKeys.IMAGE_ID]] = []
            img_annotations[ann[AnnFileKeys.IMAGE_ID]].append(ann)

            if ann[AnnFileKeys.IMAGE_ID] not in img2cat_annotations:
                img2cat_annotations[ann[AnnFileKeys.IMAGE_ID]] = {}
                img2cat[ann[AnnFileKeys.IMAGE_ID]] = set()

            if (
                ann[AnnFileKeys.CATEGORY_ID]
                not in img2cat_annotations[ann[AnnFileKeys.IMAGE_ID]]
            ):
                img2cat_annotations[ann[AnnFileKeys.IMAGE_ID]][
                    ann[AnnFileKeys.CATEGORY_ID]
                ] = []
                img2cat[ann[AnnFileKeys.IMAGE_ID]].add(ann[AnnFileKeys.CATEGORY_ID])

            img2cat_annotations[ann[AnnFileKeys.IMAGE_ID]][
                ann[AnnFileKeys.CATEGORY_ID]
            ].append(ann)

            if ann[AnnFileKeys.CATEGORY_ID] not in cat2img_annotations:
                cat2img_annotations[ann[AnnFileKeys.CATEGORY_ID]] = {}
                cat2img[ann[AnnFileKeys.CATEGORY_ID]] = set()

            if ann["image_id"] not in cat2img_annotations[ann[AnnFileKeys.CATEGORY_ID]]:
                cat2img_annotations[ann[AnnFileKeys.CATEGORY_ID]][
                    ann[AnnFileKeys.IMAGE_ID]
                ] = []
                cat2img[ann[AnnFileKeys.CATEGORY_ID]].add(ann[AnnFileKeys.IMAGE_ID])

            cat2img_annotations[ann[AnnFileKeys.CATEGORY_ID]][
                ann[AnnFileKeys.IMAGE_ID]
            ].append(ann)
        return (
            img_annotations,
            img2cat,
            img2cat_annotations,
            cat2img,
            cat2img_annotations,
        )

    def _load_image(self, img_data: dict) -> Image:
        """Load an image from disk or from url.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            PIL.Image: The loaded image.
        """
        if self.img_dir is not None:
            return Image.open(f'{self.img_dir}/{img_data["file_name"]}').convert("RGB")
        return Image.open(BytesIO(requests.get(img_data["coco_url"]).content)).convert(
            "RGB"
        )

    def _load_and_preprocess_image(self, img_data: dict) -> torch.Tensor:
        """Load and preprocess an image.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            torch.Tensor: The preprocessed image.
        """
        image = self._load_image(img_data)
        return image if not self.preprocess else self.preprocess(image)

    def load_and_preprocess_images(self, img_ids: list[int]) -> torch.Tensor:
        """Load and preprocess images.

        Args:
            img_ids (list[int]): A list of image ids.
        Returns:
            torch.Tensor: The preprocessed images.
        """
        return [
            self._load_and_preprocess_image(self.images[img_id]) for img_id in img_ids
        ]

    def _get_image(
        self, image_id: list[int]
    ) -> torch.tensor:
        """Load, stack and preprocess the image.

        Args:
            image_id [int]: Image id.

        Returns:
            torch.Tensor: Returns a tensor containing the image.
        """

        return self._load_and_preprocess_image(self.images[image_id])

    def compute_ground_truth(
        self, image_id: int, cat_ids: list[int]
    ) -> list[torch.Tensor]:
        """Compute the ground truths for the given image ids and category ids.

        Args:
            image_id (int): Image ids.
            cat_ids (list[int]): Category ids.

        Returns:
            list[torch.Tensor]: A list of tensors containing the ground truths (per image).
        """
        ground_truths = []

        # generate masks
        img_size = (self.images[image_id]["height"], self.images[image_id]["width"])
        ground_truths = np.zeros(img_size, dtype=np.int64)

        for ann in self.img_annotations[image_id]:
            ann_cat = ann[AnnFileKeys.CATEGORY_ID]
            if ann_cat not in cat_ids:
                continue
            cat_idx = cat_ids.index(ann_cat)

            ann_mask = convert_mask(
                ann[AnnFileKeys.SEGMENTATION], *img_size
            )
            ground_truths[ann_mask == 1] = cat_idx

        return torch.tensor(ground_truths)

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Args:
            idx (int): The index of the item to get.

        Returns:
            dict: A dictionary containing the data.
        """
        base_image_data = self.images[self.image_ids[idx]]
        image_id = base_image_data["id"]
        cat_ids = list(self.img2cat[image_id])
        
        # load, stack and preprocess the images
        image = self._load_and_preprocess_image(base_image_data)
        ground_truths = self.compute_ground_truth(image_id, cat_ids)

        # stack ground truths
        dims = torch.tensor([self.images[image_id]["height"], self.images[image_id]["width"]])
        max_dims = torch.max(dims, 0).values.tolist()

        return DataDict(
            image=image,
            dims=dims,
            image_id=image_id,
            target=ground_truths,
        )

    def __len__(self):
        return len(self.images)


class CocoLVISCrop(CocoLVISDataset):
    """
    COCO dataset that gives for each instancein the image its crop.
    """
    def __init__(
        self,
        *args, **kwargs
    ):
        """Initialize the dataset.

        Args:
            instances_path (str): Path to the instances json file.
            img_dir (Optional[str], optional): Path to the directory containing the images. Defaults to None.
            preprocess (_type_, optional): A preprocessing step to apply to the images. Defaults to ToTensor().
            image_size (int, optional): The size of the images. Defaults to 1024.
        """
        super().__init__(*args, **kwargs)
        self.idx2annotation = {i: ann for i, ann in enumerate(self.annotations.values())}
        

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Args:
            idx (int): The index of the item to get.

        Returns:
            dict: A dictionary containing the data.
        """
        annotation = self.idx2annotation[idx]
        image_id = annotation[AnnFileKeys.IMAGE_ID]
        cat_id = annotation[AnnFileKeys.CATEGORY_ID]
        image: Image.Image = self._load_image(self.images[image_id])
        
        # get the crop
        bbox = xywh2xyxy([annotation[AnnFileKeys.BBOX]])[0]
        crop = image.crop(bbox)
        crop = self.preprocess(crop)
        return DataDict(
            image=crop,
            image_id=image_id,
            target=cat_id,
        )
            