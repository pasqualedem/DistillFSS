r""" PASCAL-5i few-shot semantic segmentation dataset """

import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

from fssweed.data.utils import BatchKeys


class DatasetPASCAL(Dataset):
    def __init__(self, datapath, preprocess, prompt_images=None):
        self.nfolds = 4
        self.num_classes = 20
        self.benchmark = "pascal"
        self.prompt_images = prompt_images

        self.img_path = os.path.join(datapath, "pascal", "JPEGImages/")
        self.ann_path = os.path.join(datapath, "pascal", "SegmentationClass/")
        self.ann_path_aug = os.path.join(datapath, "pascal", "SegmentationClassAug/")
        
        self.images_name = [
            os.path.splitext(f)[0]
            for f in set(os.listdir(self.ann_path) + os.listdir(self.ann_path_aug))
            if f.endswith(".png")
        ]

        self.transform = preprocess

    def __len__(self):
        return len(self.images_name)
        
    def __getitem__(self, idx):
        return self.get_sample(idx, "test")
    
    def train_len(self):
        return len(self.images_name)

    def get_sample(self, idx, split="train"):
        img_id = self.images_name[idx]

        img = self.read_img(img_id)
        mask = self.read_mask(img_id)

        img = self.transform(img)
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            img.size()[-2:],
            mode="nearest",
        ).squeeze()
        mask, query_ignore_idx = self.extract_ignore_idx(
            mask.float()
        )
        size = torch.tensor(img.shape[-2:])

        return {
            BatchKeys.IMAGES: img.unsqueeze(0),
            BatchKeys.DIMS: size,
            BatchKeys.IMAGE_IDS: [img_id],
        }, mask

    def extract_ignore_idx(self, mask):
        boundary = (mask / 255).floor()
        mask[boundary == 1] = 0

        return mask, boundary

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        if os.path.exists(os.path.join(self.ann_path_aug, img_name) + ".png"):
            path = self.ann_path_aug
        else:
            path = self.ann_path
        
        mask = torch.tensor(
            np.array(Image.open(os.path.join(path, img_name) + ".png"))
        )
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + ".jpg")
