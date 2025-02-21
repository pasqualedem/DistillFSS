import pandas as pd

from sklearn.model_selection import train_test_split
from fssweed.data.utils import BatchKeys
import os
from PIL import Image
import torchvision
import numpy as np
import json
import torch
from torchvision import transforms
from pycocotools import mask as mask_utils
from torch.nn.functional import one_hot

import torch.nn.functional as F

from fssweed.utils.utils import hierarchical_uniform_sampling


def build_dataframe(gt_folder):
    subfolders = os.listdir(gt_folder)
    # each folder has several masks
    masks = [
        os.path.join(gt_folder, subfolder, mask)
        for subfolder in subfolders
        for mask in os.listdir(os.path.join(gt_folder, subfolder))
    ]
    df = pd.DataFrame(
        {
            "mask_path": masks,
        }
    )
    df['img_path'] = df['mask_path'].apply(lambda x: x.replace("-features", "").replace(".features", ".jpg"))
    df['id'] = df['img_path'].apply(lambda x: x.split("/")[-1].split(".")[0])
    df['class'] = df['mask_path'].apply(lambda x: x.split("/")[-2])
    df.set_index("id", inplace=True)
    df["id"] = df.index
    
    return df


class KvarisTestDataset:
    id2class = {0: "background", 1: "polyp"}
    num_classes = 2
    class_ids = range(0, 2)

    def __init__(
        self,
        datapath: str,
        preprocess=None,
        prompt_images=None,
    ):
        super().__init__()
        self.root = os.path.join(datapath, "kvasir")
        gt_folder = os.path.join(self.root, "kvasir-dataset-v2-features")
        
        metadata = build_dataframe(gt_folder)
        
        if "test.csv" in os.listdir(self.root):
            test_metadata_ids = pd.read_csv(os.path.join(self.root, "test.csv"))
            self.test_metadata = metadata.loc[test_metadata_ids["id"]]
            self.train_metadata = metadata.drop(test_metadata_ids["id"])
        else:
            self.train_metadata, self.test_metadata = train_test_split(metadata, test_size=0.2, random_state=42)
            self.test_metadata["id"].to_csv(os.path.join(self.root, "test.csv"), index=False)

        self.preprocess = preprocess

        self.prompt_images = prompt_images

    def __len__(self):
        return len(self.test_metadata)
    
    def __getitem__(self, index):
        return self.get_sample(index, "test")

    def get_sample(self, index, split):
        if split == "train":
            metadata = self.train_metadata
        elif split == "test":
            metadata = self.test_metadata
        else:
            raise NotImplementedError
        
        label1, mask, _, img = metadata.iloc[index]
        mask = torch.tensor(mask)

        img = self.transform(img)
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            img.size()[-2:],
            mode="nearest",
        ).squeeze()
        size = torch.tensor(img.shape[-2:])

        return {
            BatchKeys.IMAGES: img.unsqueeze(0),
            BatchKeys.DIMS: size,
            BatchKeys.IMAGE_IDS: [index],
        }, mask
        
    def train_len(self):
        return len(self.train_metadata)
    
    def read_img(self, img_path):
        return Image.open(img_path).convert("RGB")
    
    def read_mask(self, mask_path):
        with open(mask_path, "r") as f:
            mask = json.load(f)
        return mask
    
    def extract_prompts(self, prompt_images=None):
        prompt_images = prompt_images or self.prompt_images
        if isinstance(prompt_images, int):
            # linspace over the train_len
            prompt_images = hierarchical_uniform_sampling(self.train_len()-1, prompt_images)
        
        prompt_df = self.train_metadata.loc[prompt_images]
        
        images = [
            self.read_img(x.img_path)
            for x in prompt_df.itertuples()
        ]
        masks = [
                self.read_mask(x.mask_path)
            for x in prompt_df.itertuples()
        ]
        images = [self.transform(image) for image in images]
        masks = [
            F.interpolate(
                torch.tensor(mask).unsqueeze(0).unsqueeze(0).float(),
                images[0].size()[-2:],
                mode="nearest",
            ).squeeze()
            for mask in masks
        ]
        image_ids = list(prompt_images)

        sizes = torch.stack([torch.tensor(x.shape[1:]) for x in images])
        images = torch.stack(images)
        masks = torch.stack(masks)
        # Background flags are always 0
        flag_masks = torch.stack(
            [(masks == c).sum(dim=(1, 2)) > 0 for c in self.class_ids]
        ).T

        masks = one_hot(masks.long(), self.num_classes).permute(0, 3, 1, 2).float()
        # Set background to 0
        masks[:, 0] = 0

        flag_examples = flag_masks.clone().bool()
        # Set background to True
        flag_examples[:, 0] = True
        prompt_dict = {
            BatchKeys.IMAGES: images,
            BatchKeys.PROMPT_MASKS: masks,
            BatchKeys.FLAG_MASKS: flag_masks,
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            BatchKeys.IMAGE_IDS: image_ids,
            BatchKeys.DIMS: sizes,
        }
        return prompt_dict