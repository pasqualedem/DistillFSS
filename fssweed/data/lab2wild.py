import torch.nn.functional as F
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision

from PIL import Image
from torch.nn.functional import one_hot
from os.path import join, isfile
from os import listdir

from fssweed.data.utils import BatchKeys


def load_lab_mask(mask_path):
    """
    Loads lab masks.
    These masks originally had more types of apple rotting, so we are simplifying them.
    """
    mask = plt.imread(mask_path)
    mask = (mask * 255).astype("uint8")

    mask_1 = (
        np.where(mask == 2, 2, 0)
        + np.where(mask == 3, 2, 0)
        + np.where(mask == 1, 2, 0)
    )
    mask_2 = np.where(mask == 4, 1, 0)
    mask = (mask_1 + mask_2).astype("uint8")

    return mask


def load_wild_mask(mask_path):
    mask = plt.imread(mask_path)
    mask = (mask * 255).astype("uint8")

    return mask[:, :, 0]


def get_dataframe(files):

    df = pd.DataFrame(files, columns=["img_path"])
    df['id'] = df["img_path"].apply(
            lambda x: x.split("/")[-1].split(".")[0]
        )
    df["mask_path"] = df["img_path"].apply(
            lambda x: x.replace(".jpg", ".png").replace(".JPG", ".png").replace("obj", "msk")
        )
    df.set_index("id", inplace=True)
    df['id'] = df.index
    df = df[["id", "img_path", "mask_path"]]
    
    return df


class Lab2Wild:
    id2class = {0: "background", 1: "apple", 2: "rotten"}
    num_classes = len(id2class)
    class_ids = range(0, num_classes)   

    def __init__(
        self,
        datapath,
        preprocess=None,
        prompt_images=None,
    ):
        super().__init__()
        self.datapath = os.path.join(datapath, "Lab2Wild")
        self.transform = preprocess
        self.prompt_images = prompt_images
        self.train_path = os.path.join(
            self.datapath, "apples_segmentation/apples_segmentation/max_data/train/"
        )
        backgrounds_path = os.path.join(
            self.datapath, "apples_segmentation/apples_segmentation/Sk_background/"
        )
        self.test_path = os.path.join(self.datapath, "test_wild/test_wild/")

        train_images = [
            join(self.train_path, f)
            for f in listdir(self.train_path)
            if isfile(join(self.train_path, f)) and f.split(".")[-1].lower() == "jpg"
        ]
        self.train_metadata = get_dataframe(train_images)
        
        wild_test_images = [
            join(self.test_path, f)
            for f in listdir(self.test_path)
            if isfile(join(self.test_path, f)) and f.split(".")[-1].lower() == "jpg"
        ]
        self.test_metadata = get_dataframe(wild_test_images)

    def __len__(self):
        return len(self.test_metadata)

    def extract_prompts(self, prompt_images=None):
        prompt_images = prompt_images or self.prompt_images
        if isinstance(prompt_images, int):
            prompt_images = self.train_metadata.sample(prompt_images, random_state=42).index
        
        prompt_df = self.train_metadata.loc[prompt_images]
        
        images = [
            self.read_image(x.img_path)
            for x in prompt_df.itertuples()
        ]
        masks = [
            self.read_mask(
                x.mask_path,
                "train",
            )
            for x in prompt_df.itertuples()
        ]
        images = [self.transform(image) for image in images]
        masks = [
            F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
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
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            BatchKeys.IMAGE_IDS: image_ids,
            BatchKeys.DIMS: sizes,
        }
        return prompt_dict

    def read_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def read_mask(self, path, split):
        if split == "train":
            return torch.tensor(load_lab_mask(path))
        else:
            return torch.tensor(load_wild_mask(path))

    def get_sample(self, index, split):
        if split == "train":
            metadata = self.train_metadata
        elif split == "test":
            metadata = self.test_metadata
        else:
            raise NotImplementedError
        
        img_id, img_path, mask_path = metadata.iloc[index]
        img = self.read_image(img_path)
        mask = self.read_mask(mask_path, split)

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
            BatchKeys.IMAGE_IDS: [img_id],
        }, mask

    def __getitem__(self, i):
        return self.get_sample(i, "test")

    def train_len(self):
        return len(self.train_metadata)
