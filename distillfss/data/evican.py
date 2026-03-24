import itertools
import os
import random
import warnings
from io import BytesIO
from typing import Any, Optional

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor, ToTensor
from distillfss.utils.logger import get_logger

import distillfss.data.utils as utils
from distillfss.data.transforms import (
    PromptsProcessor,
)
from distillfss.data.utils import (
    AnnFileKeys,
    BatchKeys,
)

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


def load_annotation_dicts(
    images_ids: list, annotations: dict, categories: dict
) -> tuple[dict, dict, dict, dict, dict]:
    """Load useful annotation dicts.

    Returns:
        mg_annotations: A dictionary mapping image ids to lists of annotations.
    """
    img_annotations = {
        k: [] for k in images_ids
    }

    category_ids = set(categories.keys())

    for ann in annotations.values():

        if AnnFileKeys.ISCROWD in ann and ann[AnnFileKeys.ISCROWD] == 1:
            continue

        if ann[AnnFileKeys.CATEGORY_ID] not in category_ids:
            continue

        img_annotations[ann[AnnFileKeys.IMAGE_ID]].append(ann)
        
    return img_annotations


class EVICAN(Dataset):
    """Dataset class for EVICAN Dataset"""

    id2class = {
        0: "Background",
        1: "Cell",
        2: "Nucleus",
    }
    num_classes = len(id2class)

    def __init__(
        self,
        datapath: str,
        preprocess=ToTensor(),
        prompt_images=None
    ):
        """Initialize the dataset.

        Args:
            name (str): A name for the dataset (e.g. "coco", "lvis").
            root (str): Path to the root folder.
            preprocess (_type_, optional): A preprocessing step to apply to the images. Defaults to ToTensor().
        """
        super().__init__()
        print(f"Loading dataset annotations from {datapath}...")

        self.prompt_images = prompt_images

        self.root = os.path.join(datapath, "EVICAN")
        self.img_dir = os.path.join(self.root, "Images")

        # load instances
        train_instances_path = os.path.join(
            self.root, "annotations_EVICAN2/instances_train2019_EVICAN2.json"
        )
        self.train_instances = utils.load_instances(train_instances_path)

        val_instances_path = os.path.join(
            self.root, "annotations_EVICAN2/instances_val2019_EVICAN2.json"
        )
        self.val_instances = utils.load_instances(val_instances_path)

        self.train_annotations = {
            x[AnnFileKeys.ID]: x for x in self.train_instances[AnnFileKeys.ANNOTATIONS]
        }
        self.val_annotations = {
            x[AnnFileKeys.ID]: x for x in self.val_instances[AnnFileKeys.ANNOTATIONS]
        }
        self.categories = {
            x[AnnFileKeys.ID]: x for x in self.train_instances[AnnFileKeys.CATEGORIES]
        }

        self.train_images = {
            x[AnnFileKeys.ID]: x for x in self.train_instances[AnnFileKeys.IMAGES]
        }
        self.val_images = {
            x[AnnFileKeys.ID]: x for x in self.val_instances[AnnFileKeys.IMAGES]
        }

        # useful dicts
        self.train_img_annotations = load_annotation_dicts(self.train_images.keys(), self.train_annotations, self.categories)
        self.val_img_annotations = load_annotation_dicts(self.val_images.keys(), self.val_annotations, self.categories)

        # processing
        self.preprocess = preprocess
        self.prompts_processor = PromptsProcessor(
            long_side_length=None,
            masks_side_length=256,
            custom_preprocess=False,
        )

    def _load_image(self, img_data: dict, split) -> Image:
        """Load an image from disk or from url.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            PIL.Image: The loaded image.
        """
        if self.img_dir is not None:
            return Image.open(f'{self.img_dir}/{split}/{img_data["file_name"]}').convert("RGB")
        return Image.open(BytesIO(requests.get(img_data["coco_url"]).content)).convert(
            "RGB"
        )

    def _load_and_preprocess_image(self, img_data: dict, split: str) -> torch.Tensor:
        """Load and preprocess an image.

        Args:
            img_data (dict): A dictionary containing the image data, as in the coco dataset.

        Returns:
            torch.Tensor: The preprocessed image.
        """
        image = self._load_image(img_data, split)
        return image if not self.preprocess else self.preprocess(image)

    def load_and_preprocess_images(self, imgs_data: list[int], split: str) -> torch.Tensor:
        """Load and preprocess images.

        Args:
            img_ids (list[int]): A list of image ids.
        Returns:
            torch.Tensor: The preprocessed images.
        """
        return [
            self._load_and_preprocess_image(img_data, split) for img_data in imgs_data
        ]

    def compute_ground_truths(self, image_ids: list[int], img_sizes, img_annotations) -> list[torch.Tensor]:
        """Compute the ground truths for the given image ids and category ids.

        Args:
            image_ids (list[int]): Image ids.
            img_sizes (list[tuple]): Image sizes (height, width).
            img_annotations (dict): Annotations for each image.

        Returns:
            list[torch.Tensor]: A list of tensors containing the ground truths (per image).
        """

        ground_truths = [np.zeros(img_sizes[i], dtype=np.int64) for i in range(len(image_ids))]

        for i, image_id in enumerate(image_ids):
            gt_mask = ground_truths[i]  # Direct reference to reduce indexing overhead

            for ann in img_annotations[image_id]:
                ann_cat = ann[AnnFileKeys.CATEGORY_ID]
                ann_mask = self.prompts_processor.convert_mask(ann[AnnFileKeys.SEGMENTATION], *img_sizes[i])
                # More efficient mask update
                np.copyto(gt_mask, ann_cat, where=ann_mask == 1)

        return list(map(torch.tensor, ground_truths))


    def get_sample(self, idx, split):
        if split == "train":
            images = self.train_images
            img_annotations = self.train_img_annotations
        elif split == "val":
            images = self.val_images
            img_annotations = self.val_img_annotations
        else:
            raise NotImplementedError
        
        img_id = list(images.keys())[idx]
        base_image_data = images[img_id]
        image = self._load_and_preprocess_image(base_image_data, split)

        
        image_id = base_image_data["id"]
        img_sizes = [
            (images[img_id]["height"], images[img_id]["width"])
            for img_id in [image_id]
        ]
        ground_truth = self.compute_ground_truths([image_id], img_sizes, img_annotations)[0]
        ground_truth = F.interpolate(
            ground_truth.unsqueeze(0).unsqueeze(0).float(),
            image.size()[-2:],
            mode="nearest",
        ).squeeze()
        
        dims = torch.tensor(image.shape[-2:])


        return {
            BatchKeys.IMAGES: image.unsqueeze(0),
            BatchKeys.DIMS: dims,
            BatchKeys.IMAGE_IDS: [image_id],
        }, ground_truth
        
    def __getitem__(self, index):
        return self.get_sample(index, "val")

    def extract_prompts(self, prompt_images=None) -> dict:
        """Get an item from the dataset.

        Args:
            idx_metadata (tuple[int, dict]): A tuple containing the index of the image and the batch level metadata e.g. number of examples to be chosen and type of prompts.

        Returns:
            dict: A dictionary containing the data.
        """
        prompt_images = prompt_images or self.prompt_images
        if isinstance(prompt_images, int):
            prompt_images = random.sample(list(self.train_images), prompt_images)
        
        # load, stack and preprocess the images
        images = self.load_and_preprocess_images([self.train_images[img_id] for img_id in prompt_images], "train")
        images = torch.stack(images)
        
        img_sizes = [
            (self.train_images[img_id]["height"], self.train_images[img_id]["width"])
            for img_id in prompt_images
        ]
        ground_truths = self.compute_ground_truths(prompt_images, img_sizes, self.train_img_annotations)
        ground_truths = [
            F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                images[0].size()[-2:],
                mode="nearest",
            ).squeeze()
            for mask in ground_truths
        ]
        # stack ground truths
        dims = torch.stack([torch.tensor(image.shape[-2:]) for image in images])
        ground_truths = torch.stack(ground_truths)
        flag_examples = torch.stack(
            [(ground_truths == c).sum(dim=(1, 2)) > 0 for c in self.id2class.keys()]
        ).T.bool()

        masks = F.one_hot(ground_truths.long(), self.num_classes).permute(0, 3, 1, 2).float()
        # Set background to 0
        masks[:, 0] = 0

        data_dict = {
            BatchKeys.IMAGES: images,
            BatchKeys.PROMPT_MASKS: masks,
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            BatchKeys.DIMS: dims,
            BatchKeys.IMAGE_IDS: prompt_images,
            BatchKeys.GROUND_TRUTHS: ground_truths,
        }
        return data_dict

    def __len__(self):
        return len(self.val_images)
    
    def train_len(self):
        return len(self.train_images)
