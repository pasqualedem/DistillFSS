r""" ISIC few-shot semantic segmentation dataset """
import os
import glob
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

from fssweed.data.utils import BatchKeys
from torch.nn.functional import one_hot

from fssweed.utils.utils import hierarchical_uniform_sampling


def get_dataframe(path):
    
    def get_id(path):
        path, name = os.path.split(path)
        subfolder = os.path.split(os.path.split(path)[0])[-1]
        name = name.split(".")[0]
        return subfolder + "." + name
        
    subfolders = [
        os.path.join(path, f)
        for f in os.listdir(path)
    ]
    files = [
        os.path.join(subfolder, "images", f)
        for subfolder in subfolders
        for f in os.listdir(os.path.join(subfolder, "images"))
    ]
    df = pd.DataFrame(files, columns=["img_path"])
    df['id'] = df["img_path"].apply(get_id)
    df['mask_path'] = df["img_path"].apply(lambda x: x.replace("images", "masks").replace(".jpg", ".png").replace(".JPG", ".png"))
    df.set_index("id", inplace=True)
    df['id'] = df.index
    df = df[["id", "img_path", "mask_path"]]
    
    return df


class Pothole(Dataset):
    id2class = {0: "background", 1:'pothole'}
    num_classes = len(id2class)
    class_ids = range(0, 2)   
    
    def __init__(self, datapath, preprocess, prompt_images=None,**kwargs):
        self.benchmark = 'isic'

        self.base_path = os.path.join(datapath, 'pothole-mix')
        self.train_folder = os.path.join(self.base_path, "training")
        self.test_folder = os.path.join(self.base_path, "testing")
        
        self.transform = preprocess
        self.prompt_images = prompt_images    
        
        self.train_img_metadata = get_dataframe(self.train_folder)
        self.test_img_metadata = get_dataframe(self.test_folder)

    def __len__(self):
        return len(self.test_img_metadata)
    
    def __getitem__(self, index):
        return self.get_sample(index, "test")

    def get_sample(self, index, split):
        if split == "train":
            metadata = self.train_img_metadata
        elif split == "test":
            metadata = self.test_img_metadata
        else:
            raise NotImplementedError
        
        img_id, img_path, mask_path = metadata.iloc[index]
        img = self.read_image(img_path)
        mask = self.read_mask(mask_path)

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
        
    def train_len(self):
        return len(self.train_img_metadata)
        
    def read_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image
        
    def extract_prompts(self, prompt_images=None):
        prompt_images = prompt_images or self.prompt_images
        if isinstance(prompt_images, int):
            prompt_images = hierarchical_uniform_sampling(self.train_len()-1, prompt_images)
            prompt_df = self.train_img_metadata.iloc[prompt_images]
        else:
            prompt_df = self.train_img_metadata.loc[prompt_images]
        
        images = [
            self.read_image(x.img_path)
            for x in prompt_df.itertuples()
        ]
        masks = [
            self.read_mask(
                x.mask_path
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
            BatchKeys.FLAG_MASKS: flag_masks,
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            BatchKeys.IMAGE_IDS: image_ids,
            BatchKeys.DIMS: sizes,
        }
        return prompt_dict
    
    def read_mask(self, mask_path):
        mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
        mask[mask > 0] = 1
        return mask

