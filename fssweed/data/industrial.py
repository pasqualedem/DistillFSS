import os

import pandas as pd
import torch.nn.functional as F
import torch
import PIL.Image as Image

# from pycocotools.coco import orgCOCO
import json
import cv2

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.nn.functional import one_hot

from fssweed.data.utils import BatchKeys
from fssweed.utils.utils import hierarchical_uniform_sampling


def build_index(base_path, annotations):
    annotations = {k: v["bad"] for k, v in annotations.items()}
    classes = sum([[c] * len(annotations[c]) for c in annotations], [])
    img_paths = sum(annotations.values(), [])
    df = pd.DataFrame({"img_path": img_paths, "label": classes})
    df["img_path"] = df["img_path"].apply(lambda x: x.replace("data/", base_path + "/"))
    df["seg_path"] = df["img_path"].apply(lambda x: x.replace(".png", ".bmp"))
    df["id"] = (
        df["img_path"]
        .apply(lambda x: x.split("/")[1] + "_" + os.path.split(x)[-1].split(".")[0])
        .astype(str)
    )
    df = df.set_index("id")
    df["id"] = df.index
    df = df[["id", "img_path", "seg_path", "label"]]
    return df


class DatasetIndustrial(Dataset):
    nfolds = 4
    def __init__(
        self,
        datapath,
        preprocess,
        use_original_imgsize=False,
        prompt_images=None,
        fold=None,
        **kwargs
    ):
        self.base_path = datapath + "/Industrial"
        self.annotion_path = self.base_path + "/data.json"

        self.transform = preprocess
        self.use_original_imgsize = use_original_imgsize
        self.prompt_images = prompt_images

        annotations = self.load_json(self.base_path)
        self.id2class = {i+1: c for i, c in enumerate(annotations["clsName"])}
        self.id2class.update({0: "background"})
        self.class2id = {c: i for i, c in self.id2class.items()}
        self.num_classes = len(self.id2class)
        
        self.annotations = annotations["clsDic"]
        metadata = build_index(self.base_path, self.annotations)

        if "test.csv" in os.listdir(self.base_path):
            test_metadata_ids = pd.read_csv(os.path.join(self.base_path, "test.csv"))
            self.test_metadata = metadata.loc[test_metadata_ids["id"]]
            self.train_metadata = metadata.drop(test_metadata_ids["id"])
        else:
            self.train_metadata, self.test_metadata = train_test_split(
                metadata, test_size=0.2, random_state=42
            )
            self.test_metadata["id"].to_csv(
                os.path.join(self.base_path, "test.csv"), index=False
            )
            
        self.apply_fold(fold)

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

        img_id, img, mask, label = metadata.iloc[index]

        img = Image.open(img).convert("RGB")
        img = self.transform(img)
        mask = self.read_mask(mask, label)
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

    def extract_prompts(self, prompt_images=None):
        prompt_images = prompt_images or self.prompt_images

        if isinstance(prompt_images, int):
            num_samples_per_class = prompt_images // self.num_classes
            selected_samples = self.train_metadata.groupby(
                "label", group_keys=False
            ).apply(
                lambda x: x.iloc[
                    hierarchical_uniform_sampling(len(x) - 1, num_samples_per_class)
                ]
            )
            prompt_images = selected_samples.index

        prompt_df = self.train_metadata.loc[prompt_images]

        images = [Image.open(x.img_path).convert("RGB") for x in prompt_df.itertuples()]
        masks = [self.read_mask(x.seg_path, x.label) for x in prompt_df.itertuples()]
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

    def load_json(self, data_path):
        with open(data_path + "/data.json", "r") as f:
            img_metadata_classwise = json.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.annotations.keys():
            img_metadata += self.annotations[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, mask_path, label):
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt[gt > 0] = int(self.class2id[label])
        return torch.tensor(gt)

    def apply_fold(self, fold):
        if fold is None:
            return
        nclass_val = self.num_classes // self.nfolds
        class_ids_val = [(fold + self.nfolds * v)+1 for v in range(nclass_val)]
        self.num_classes = len(class_ids_val) + 1
        self.id2class = {i: c for i, c in self.id2class.items() if i in class_ids_val}
        self.id2class = {k+1: c for k, (i, c) in enumerate(self.id2class.items())}
        self.id2class.update({0: "background"})
        
        self.class2id = {c: i for i, c in self.id2class.items()}
        self.class_ids = range(self.num_classes)
        
        self.train_metadata = self.train_metadata[self.train_metadata["label"].isin(self.id2class.values())]
        self.test_metadata = self.test_metadata[self.test_metadata["label"].isin(self.id2class.values())]

