import torch
import numpy as np

from phenobench import PhenoBench
from torch.nn.functional import one_hot

from fssweed.data.utils import BatchKeys


class PhenoBenchTestDataset:
    id2class = {0: "background", 1: "crop", 2: "weed"}
    num_classes = len(id2class)
    def __init__(self, root, prompt_images, preprocess=None):
        self.transform = preprocess
        self.prompt_images = prompt_images
        
        self.train_data = PhenoBench(root=root, target_types=["semantics"])
        self.test_data = PhenoBench(root=root, target_types=["semantics"], split="test")
        
    def extract_prompts(self, prompt_images=None):
        prompt_images = prompt_images or self.prompt_images
        items = [self.train_data.__getitem__(i) for i in prompt_images]
        images = [item["image"] for item in items]
        sizes = torch.stack([torch.tensor(x.size) for x in images])
        images = [
            self.transform(image)
            for image in images
        ]
        images = torch.stack(images)
        masks = [torch.tensor(item["semantics"]) for item in items]
        masks = torch.stack(masks)
        # Background flags are always 0
        backflag = torch.zeros(masks.shape[0])
        contains_crop = (masks == 1).sum(dim=(1, 2)) > 0
        contains_weed = (masks == 2).sum(dim=(1, 2)) > 0
        flag_masks = torch.stack([backflag, contains_crop, contains_weed]).T
        
        masks = one_hot(masks.long(), 3).permute(0, 3, 1, 2).float()
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
        }
        return prompt_dict
        
    def __getitem__(self, i):
        item_dict = self.train_data.__getitem__(i)
        gt = torch.tensor(item_dict["semantics"].astype(np.uint8))
        image = item_dict["image"]
        size = torch.tensor(image.size) # Example dimension
        image = self.transform(image)
        return {
            BatchKeys.IMAGES: image.unsqueeze(0),
            BatchKeys.DIMS: size,
        }, gt
        
    def __len__(self):
        return len(self.test_data)