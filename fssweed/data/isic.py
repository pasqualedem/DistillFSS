r""" ISIC few-shot semantic segmentation dataset """
import os
import glob

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

from fssweed.data.utils import BatchKeys
from torch.nn.functional import one_hot


def build_index(img_path, ann_path, gt_class_df):
    gt_class_df["id"] = gt_class_df["image"].apply(lambda x: x.replace("_downsampled", "")).astype(str)
    img_metadata = [
            (
                os.path.join(
                    img_path, f.replace("_segmentation.png", ".jpg")
                ),
                os.path.join(ann_path, f),
            )
            for f in filter(lambda x: x.endswith(".png"), os.listdir(ann_path))
        ]
    img_metadata_df = pd.DataFrame(img_metadata, columns=["img_path", "seg_path"])
    img_metadata_df["id"] = img_metadata_df["img_path"].apply(lambda x: os.path.split(x)[-1].split(".")[0]).astype(str)
    merged_df = gt_class_df.join(img_metadata_df.set_index("id"), on="id", how="inner")
    merged_df["label"] = merged_df[["NV","MEL","BKL","BCC","AK","DF","VASC","SCC","UNK"]].to_numpy().argmax(axis=1) + 1
    merged_df = merged_df.drop(["NV","MEL","BKL","BCC","AK","DF","VASC","SCC","UNK"], axis=1)
    merged_df = merged_df.set_index("id")
    return merged_df

# 1:1867 2:519 3:208
class DatasetISIC(Dataset):
    id2class = {0: "background", 1:'nevus', 2:'melanoma', 3:'seborrheic_keratosis'}
    num_classes = len(id2class)
    
    def __init__(self, datapath, preprocess, num=600, prompt_images=None,**kwargs):
        self.benchmark = 'isic'
        self.num = num

        self.base_path = os.path.join(datapath, 'ISIC')
        gt_class_df_path = os.path.join(self.base_path, "ISIC_2019_Training_GroundTruth.csv")
        gt_class_df = pd.read_csv(gt_class_df_path)
        
        train_img_path = os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input')
        train_ann_path = os.path.join(self.base_path, 'ISIC2018_Task1_Training_GroundTruth')
        
        test_csv = pd.read_csv(os.path.join(self.base_path, "test.csv"))
        
        self.transform = preprocess

        self.class_ids = range(0, 4)           
        
        self.train_img_metadata = build_index(train_img_path, train_ann_path, gt_class_df)
        self.test_img_metadata = self.train_img_metadata.loc[test_csv['id']]
        self.train_img_metadata = self.train_img_metadata.drop(test_csv["id"])
        

        if isinstance(prompt_images, list):
            self.prompt_images = prompt_images
        else:
            num_samples_per_class = prompt_images // self.num_classes
            selected_samples = self.train_img_metadata.groupby("label", group_keys=False).apply(
                lambda x: x.sample(n=num_samples_per_class, random_state=42)
            )
            self.prompt_images = selected_samples.index

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
        
        img_id, img, mask, label = metadata.iloc[index]
        img = Image.open(img).convert("RGB")
        mask = self.read_mask(mask, label)

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
        
    def extract_prompts(self, prompt_images=None):
        prompt_images = prompt_images or self.prompt_images
        prompt_df = self.train_img_metadata.loc[prompt_images]
        
        images = [
            Image.open(x.img_path).convert("RGB")
            for x in prompt_df.itertuples()
        ]
        masks = [
            self.read_mask(
                x.seg_path,
                x.label
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
    
    def read_mask(self, img_name, label):
        mask = self.read_binary_mask(img_name)
        mask[mask == 1] = label
        return mask

