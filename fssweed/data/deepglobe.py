r""" FSS-1000 few-shot semantic segmentation dataset """

import os
import glob

import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

from torch.nn.functional import one_hot

from fssweed.data.utils import BatchKeys


def coloredgt_to_label(coloredgt, labelset):
    """Convert an RGB ground truth image to a label map.

    Args:
        coloredgt (np.array): An (H, W, 3) RGB ground truth image.
        labelset (list): A list of RGB values, where each index represents a class.

    Returns:
        np.array: A (H, W) array where each pixel has a class index.
    """
    # Convert labelset to a NumPy array for efficient comparisons
    labelset = np.array(labelset)

    # Reshape coloredgt for broadcasting
    coloredgt_flat = coloredgt.reshape(-1, 3)  # (H*W, 3)

    # Find matching indices in labelset
    matches = (coloredgt_flat[:, None, :] == labelset[None, :, :]).all(axis=2)

    # Get class indices
    label_map = matches.argmax(axis=1).reshape(coloredgt.shape[:2])

    return label_map


class DatasetDeepglobe(Dataset):
    labelset = [
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 255],
    ]
    categories = {
        0: "background",
        1: "urban_land",
        2: "agriculture_land",
        3: "rangeland",
        4: "forest_land",
        5: "water",
        6: "barren_land",
    }
    class_ids = range(0, 7)
    id2class = categories  # Alias
    num_classes = len(class_ids)

    def __init__(
        self,
        datapath,
        preprocess,
        n_shots=1,
        val_num_samples=600,
        prompt_images=None,
        **kwargs
    ):
        self.benchmark = "deepglobe"
        self.shot = n_shots
        self.num = val_num_samples
        self.transform = preprocess

        self.base_path = os.path.join(datapath, "Deepglobe")
        self.img_path = os.path.join(self.base_path, "image")
        self.ann_path = os.path.join(self.base_path, "filter_mask")
        
        test_csv = pd.read_csv(os.path.join(self.base_path, "test.csv"))
        
        img_metadata = [
            (
                f.replace("_mask", "").replace(".png",""),
                os.path.join(
                    self.img_path, f.replace(".png", ".jpg").replace("mask", "sat")
                ),
                os.path.join(self.ann_path, f),
            )
            for f in filter(lambda x: x.endswith(".png"), os.listdir(self.ann_path))
        ]
        self.train_img_metadata = pd.DataFrame(img_metadata, columns=["id", "img_path", "seg_path"])
        self.train_img_metadata.set_index("id", inplace=True)
        
        img_ids = list(map(lambda x: x.replace("_mask", "").replace(".png",""), test_csv['id']))
        self.test_img_metadata = self.train_img_metadata.loc[img_ids]
        self.train_img_metadata = self.train_img_metadata.drop(img_ids)
        self.prompt_images = prompt_images

    def __len__(self):
        return self.num

    def extract_episode(self, idx_batchmetadata: tuple[int, int]):
        idx, _ = idx_batchmetadata
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(
            query_name, support_names
        )

        query_img = self.transform(query_img)
        query_mask = F.interpolate(
            query_mask.unsqueeze(0).unsqueeze(0).float(),
            query_img.size()[-2:],
            mode="nearest",
        ).squeeze()

        support_imgs = torch.stack(
            [self.transform(support_img) for support_img in support_imgs]
        )

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(
                smask.unsqueeze(0).unsqueeze(0).float(),
                support_imgs.size()[-2:],
                mode="nearest",
            ).squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        # batch = {'query_img': query_img,
        #          'query_mask': query_mask,
        #          'query_name': query_name,

        #          'support_imgs': support_imgs,
        #          'support_masks': support_masks,
        #          'support_names': support_names,

        #          'class_id': torch.tensor(class_sample)}
        images = torch.cat([query_img.unsqueeze(0), support_imgs], dim=0)
        support_masks = torch.cat(
            [query_mask.unsqueeze(0).unsqueeze(0), support_masks.unsqueeze(1)], dim=0
        )
        support_masks = torch.concat(
            [torch.zeros_like(support_masks), support_masks], dim=1
        )
        flags_masks = torch.stack(
            [
                torch.zeros(len(images), dtype=torch.uint8),
                torch.ones(len(images), dtype=torch.uint8),
            ],
            dim=1,
        )
        flag_examples = torch.ones(len(images), 2, dtype=torch.uint8)
        data_dict = {
            BatchKeys.IMAGES: images,
            BatchKeys.PROMPT_MASKS: support_masks,
            BatchKeys.FLAG_MASKS: flags_masks,
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            BatchKeys.DIMS: torch.tensor([img.shape[-2:] for img in images]),
            BatchKeys.CLASSES: [[0, class_sample] for _ in range(len(images))],
            BatchKeys.IMAGE_IDS: [*[query_name], *[support_names]],
            BatchKeys.GROUND_TRUTHS: support_masks[:, 1],
        }

        return data_dict

    def get_sample(self, index, split):        
        if split == "train":
            metadata = self.train_img_metadata
        elif split == "test":
            metadata = self.test_img_metadata
        else:
            raise NotImplementedError
        
        img_id = metadata.index[index]
        img, mask = metadata.iloc[index]
        img = Image.open(img).convert("RGB")
        mask = self.read_mask(mask)

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
        
    def __getitem__(self, index):
        return self.get_sample(index, "test")

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
            BatchKeys.DIMS: sizes,
            BatchKeys.IMAGE_IDS: prompt_images,
        }
        return prompt_dict
    
    def train_len(self):
        return len(self.train_img_metadata)

    def load_frame(self, query_name, support_names):
        query_id, _, query_idd = query_name.split("/")[-1].split("_")
        query_idd = query_idd.split(".")[0]
        query_img_name = (
            os.path.join(self.img_path, query_id) + "_sat_" + query_idd + ".jpg"
        )
        query_img = Image.open(query_img_name).convert("RGB")
        support_ids = [name.split("/")[-1].split("_")[0] for name in support_names]
        support_idds = [name.split("/")[-1].split("_")[2] for name in support_names]
        support_idds = [name.split(".")[0] for name in support_idds]
        support_img_names = [
            os.path.join(self.img_path, sid) + "_sat_" + sidd + ".jpg"
            for sid, sidd in zip(support_ids, support_idds)
        ]
        support_imgs = [Image.open(name).convert("RGB") for name in support_img_names]
        query_mask = self.read_binary_mask(query_name)
        support_masks = [self.read_binary_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_binary_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert("L")))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def read_mask(self, img_name):
        mask = Image.open(img_name).convert("RGB")
        return torch.tensor(coloredgt_to_label(np.array(mask), self.labelset)) + 1 # Background is 0

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]

        query_name = np.random.choice(
            self.img_metadata[class_sample], 1, replace=False
        )[0]
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(
                self.img_metadata[class_sample], 1, replace=False
            )[0]
            if query_name != support_name:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break

        return query_name, support_names, class_id

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories.values():
            os.path.join(self.base_path, cat)
            img_paths = sorted(
                [
                    path
                    for path in glob.glob(
                        "%s/*" % os.path.join(self.base_path, cat, "test", "origin")
                    )
                ]
            )
            for img_path in img_paths:
                if os.path.basename(img_path).split(".")[1] == "jpg":
                    img_metadata.append(img_path)

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for cat in self.categories.values():
            img_metadata_classwise[cat] = []

        for cat in self.categories.values():
            img_paths = sorted(
                [
                    path
                    for path in glob.glob(
                        "%s/*" % os.path.join(self.base_path, "filter_mask", cat)
                    )
                ]
            )
            for img_path in img_paths:
                if os.path.basename(img_path).split(".")[1] == "png":
                    img_metadata_classwise[cat] += [img_path]
        print(
            "Total (%s) %s images are : %d"
            % (self.split, self.benchmark, self.__len__())
        )
        return img_metadata_classwise
