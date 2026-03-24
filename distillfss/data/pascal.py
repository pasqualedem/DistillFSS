r""" PASCAL-5i few-shot semantic segmentation dataset """

import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot

from distillfss.data.utils import BatchKeys
from distillfss.utils.utils import hierarchical_uniform_sampling


class DatasetPASCAL(Dataset):
    FULL_ID2CLASS = {
        0: "background",
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "pottedplant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tvmonitor",
    }

    def __init__(
        self,
        datapath,
        preprocess,
        prompt_images=None,
        selected_classes=None,
        **kwargs,
    ):
        self.nfolds = 4
        self.benchmark = "pascal"
        self.prompt_images = prompt_images
        self.selected_classes = self._parse_selected_classes(selected_classes)
        self.id2class = {0: "background"}
        self.id2class.update(
            {
                i + 1: self.FULL_ID2CLASS[class_id]
                for i, class_id in enumerate(self.selected_classes)
            }
        )
        self.num_classes = len(self.id2class)
        self.class_ids = range(self.num_classes)

        self.base_path = os.path.join(datapath, "pascal")
        self.img_path = os.path.join(self.base_path, "JPEGImages/")
        self.ann_path = os.path.join(self.base_path, "SegmentationClass/")
        self.ann_path_aug = os.path.join(self.base_path, "SegmentationClassAug/")
        
        self.images_name = [
            os.path.splitext(f)[0]
            for f in set(os.listdir(self.ann_path) + os.listdir(self.ann_path_aug))
            if f.endswith(".png")
        ]
        self.images_name = sorted(self.images_name)

        self.img_metadata = pd.DataFrame(self.images_name, columns=["id"])
        self.img_metadata = self.img_metadata.set_index("id")
        self.img_metadata["id"] = self.img_metadata.index
        self._mask_classes_cache = {}

        test_csv_path = os.path.join(self.base_path, "test.csv")
        if os.path.exists(test_csv_path):
            test_csv = pd.read_csv(test_csv_path)
            test_ids = [x for x in test_csv["id"].tolist() if x in self.img_metadata.index]
            self.test_img_metadata = self.img_metadata.loc[test_ids]
            self.train_img_metadata = self.img_metadata.drop(test_ids)
        else:
            self.train_img_metadata, self.test_img_metadata = train_test_split(
                self.img_metadata, test_size=0.2, random_state=42
            )
            self.test_img_metadata["id"].to_csv(test_csv_path, index=False)

        self.train_img_metadata = self._filter_metadata_by_selected_classes(
            self.train_img_metadata
        )
        self.test_img_metadata = self._filter_metadata_by_selected_classes(
            self.test_img_metadata
        )
        self.train_images_by_class = self._build_train_images_by_class()

        self.transform = preprocess

    def __len__(self):
        return len(self.test_img_metadata)
        
    def __getitem__(self, idx):
        return self.get_sample(idx, "test")
    
    def train_len(self):
        return len(self.train_img_metadata)

    def get_sample(self, idx, split="train"):
        if split == "train":
            metadata = self.train_img_metadata
        elif split == "test":
            metadata = self.test_img_metadata
        else:
            raise NotImplementedError

        img_id = metadata.iloc[idx].id

        img = self.read_img(img_id)
        mask = self.prepare_mask(self.read_mask(img_id), img.size[::-1])

        img = self.transform(img)
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            img.size()[-2:],
            mode="nearest",
        ).squeeze().long()
        size = torch.tensor(img.shape[-2:])

        return {
            BatchKeys.IMAGES: img.unsqueeze(0),
            BatchKeys.DIMS: size,
            BatchKeys.IMAGE_IDS: [img_id],
        }, mask

    def extract_prompts(self, prompt_images=None):
        prompt_images = prompt_images or self.prompt_images
        if prompt_images is None:
            raise ValueError(
                "prompt_images must be provided as an integer or a list of image ids"
            )

        if isinstance(prompt_images, int):
            samples_per_class = self._compute_samples_per_class(prompt_images)
            if samples_per_class == 0:
                raise ValueError(
                    f"prompt_images={prompt_images} is too small for "
                    f"{len(self.selected_classes)} classes. "
                    f"Use at least {len(self.selected_classes)}."
                )

            selected_prompt_ids = []
            used_ids = set()
            for class_id in self.selected_classes:
                class_image_ids = [
                    x for x in self.train_images_by_class[class_id] if x not in used_ids
                ]
                n_for_class = min(samples_per_class, len(class_image_ids))
                if n_for_class == 0:
                    continue

                selected_indices = hierarchical_uniform_sampling(
                    len(class_image_ids) - 1, n_for_class
                )
                class_selected_ids = [class_image_ids[i] for i in selected_indices]
                selected_prompt_ids.extend(class_selected_ids)
                used_ids.update(class_selected_ids)

            prompt_df = self.train_img_metadata.loc[selected_prompt_ids]
        else:
            prompt_df = self.train_img_metadata.loc[prompt_images]

        images = [self.read_img(x.id) for x in prompt_df.itertuples()]
        masks = [
            self.prepare_mask(self.read_mask(x.id), images[0].size[::-1])
            for x in prompt_df.itertuples()
        ]
        images = [self.transform(image) for image in images]
        masks = [
            F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                images[0].size()[-2:],
                mode="nearest",
            ).squeeze().long()
            for mask in masks
        ]

        image_ids = list(prompt_df["id"])
        sizes = torch.stack([torch.tensor(x.shape[1:]) for x in images])
        images = torch.stack(images)
        masks = torch.stack(masks)

        flag_masks = torch.stack(
            [(masks == c).sum(dim=(1, 2)) > 0 for c in self.class_ids]
        ).T

        masks = one_hot(masks.long(), self.num_classes).permute(0, 3, 1, 2).float()
        masks[:, 0] = 0

        flag_examples = flag_masks.clone().bool()
        flag_examples[:, 0] = True

        return {
            BatchKeys.IMAGES: images,
            BatchKeys.PROMPT_MASKS: masks,
            BatchKeys.FLAG_MASKS: flag_masks,
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            BatchKeys.IMAGE_IDS: image_ids,
            BatchKeys.DIMS: sizes,
        }

    def extract_ignore_idx(self, mask):
        boundary = (mask / 255).floor()
        mask[boundary == 1] = 0

        return mask, boundary

    def _parse_selected_classes(self, selected_classes):
        if selected_classes is None:
            return list(range(1, 21))

        if isinstance(selected_classes, int):
            selected_classes = [selected_classes]

        selected_classes = sorted({int(c) for c in selected_classes if int(c) != 0})
        if invalid_classes := [c for c in selected_classes if c < 1 or c > 20]:
            raise ValueError(
                f"selected_classes contains invalid values: {invalid_classes}. "
                "Expected class ids in [1, 20]."
            )
        if not selected_classes:
            raise ValueError("selected_classes must contain at least one class id in [1, 20]")
        return selected_classes

    def _compute_samples_per_class(self, total_prompt_images):
        return total_prompt_images // len(self.selected_classes)

    def prepare_mask(self, mask, target_size):
        mask, _ = self.extract_ignore_idx(mask.float())
        return self.remap_mask(mask.long())

    def remap_mask(self, mask):
        remapped_mask = torch.zeros_like(mask)
        for new_class_id, old_class_id in enumerate(self.selected_classes, start=1):
            remapped_mask[mask == old_class_id] = new_class_id
        return remapped_mask

    def _get_image_classes(self, img_id):
        if img_id not in self._mask_classes_cache:
            mask = self.read_mask(img_id)
            classes = {
                int(c)
                for c in torch.unique(mask).tolist()
                if int(c) in self.selected_classes
            }
            self._mask_classes_cache[img_id] = classes
        return self._mask_classes_cache[img_id]

    def _filter_metadata_by_selected_classes(self, metadata):
        selected_ids = [
            row.id for row in metadata.itertuples() if self._get_image_classes(row.id)
        ]
        return metadata.loc[selected_ids]

    def _build_train_images_by_class(self):
        images_by_class = {class_id: [] for class_id in self.selected_classes}
        for row in self.train_img_metadata.itertuples():
            present_classes = self._get_image_classes(row.id)
            for class_id in present_classes:
                images_by_class[class_id].append(row.id)
        return images_by_class

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        if os.path.exists(os.path.join(self.ann_path_aug, img_name) + ".png"):
            path = self.ann_path_aug
        else:
            path = self.ann_path
        
        return torch.tensor(np.array(Image.open(os.path.join(path, img_name) + ".png")))

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + ".jpg")
