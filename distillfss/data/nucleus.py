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

from distillfss.data.utils import BatchKeys
from torch.nn.functional import one_hot

from distillfss.utils.utils import hierarchical_uniform_sampling


class Nucleus(Dataset):
    id2class = {0: "background", 1:'nucleus'}
    num_classes = len(id2class)
    
    def __init__(self, datapath, preprocess, prompt_images=None,**kwargs):
        self.benchmark = 'isic'

        self.base_path = os.path.join(datapath, 'Nucleus')
        
        self.transform = preprocess
        self.prompt_images = prompt_images

        self.class_ids = range(0, 2)           
        
        self.train_img_metadata = pd.DataFrame(filter(lambda x: not x.endswith("csv"), os.listdir(self.base_path)), columns=["id"])
        self.train_img_metadata.set_index("id", inplace=True)
        self.train_img_metadata['id'] = self.train_img_metadata.index
        
        test_csv_path = os.path.join(self.base_path, "test.csv")
        
        if os.path.exists(test_csv_path):
            test_csv = pd.read_csv(test_csv_path)
        else:
            _, test_csv = train_test_split(self.train_img_metadata, test_size=0.2, random_state=42)
            test_csv.to_csv(test_csv_path, index=False)
            
        self.test_img_metadata = self.train_img_metadata.loc[test_csv['id']]
        self.train_img_metadata = self.train_img_metadata.drop(test_csv["id"])

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
        
        img_id = metadata.iloc[index].id
        img = self.read_image(img_id)
        mask = self.read_mask(img_id)

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
        
    def read_image(self, img_id):
        images_folder = os.path.join(self.base_path, img_id, "images")
        images = os.listdir(images_folder)
        assert len(images) == 1
        image = Image.open(os.path.join(images_folder, images[0])).convert("RGB")
        return image
        
    def extract_prompts(self, prompt_images=None):
        prompt_images = prompt_images or self.prompt_images
        if isinstance(prompt_images, int):
            prompt_images = hierarchical_uniform_sampling(self.train_len()-1, prompt_images)
            prompt_df = self.train_img_metadata.iloc[prompt_images]
        else:
            prompt_df = self.train_img_metadata.loc[prompt_images]
        
        images = [
            self.read_image(x.id)
            for x in prompt_df.itertuples()
        ]
        masks = [
            self.read_mask(
                x.id
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

    def read_binary_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask
    
    def read_mask(self, img_name):
        mask_folder = os.path.join(self.base_path, img_name, "masks")
        masks = sum(
         self.read_binary_mask(os.path.join(mask_folder, f)) for f in os.listdir(mask_folder)   
        )
        return masks

