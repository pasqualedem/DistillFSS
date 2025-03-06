import os
import torch
import torchvision

from PIL import Image
from torch.nn.functional import one_hot
from tqdm import tqdm

from fssweed.data.utils import BatchKeys
from fssweed.utils.utils import hierarchical_uniform_sampling


class WeedMapTestDataset:
    id2class = {0: "background", 1: "crop", 2: "weed"}
    num_classes = 3
    def __init__(
        self,
        train_root,
        test_root,
        preprocess=None,
        prompt_images=None,
        remove_black_images=False,
    ):
        super().__init__()
        self.train_root = train_root
        self.test_root = test_root
        self.transform = preprocess
        if prompt_images is None:
            prompt_images = [
                # List of selected images from the training set
                "frame0009_2.png",
                "frame0021_2.png",
                "frame0033_3.png",
                "frame0034_1.png",
                "frame0048_0.png",
            ]
        self.prompt_images = prompt_images
        self.channels = ["R", "G", "B"]

        self.train_gt_folder = os.path.join(self.train_root, "groundtruth")
        self.test_gt_folder = os.path.join(self.test_root, "groundtruth")
        self.test_channels_folder = [
            os.path.join(self.test_root, "tile", channel) for channel in self.channels
        ]
        self.train_channels_folder = [
            os.path.join(self.train_root, "tile", channel) for channel in self.channels
        ]
        self.train_images = os.listdir(self.train_channels_folder[0])
        self.test_images = os.listdir(self.test_channels_folder[0])
        
        if remove_black_images:
            self.remove_black_images()
    
    def remove_black_images(self):
        print("Removing black images")
        black_images = []
        for filename in tqdm(self.train_images):
            image = self._get_image(self.train_channels_folder, filename)
            # If there are at least 10 black pixels
            summed = image.sum(dim=0)
            values, counts = summed.unique(return_counts=True)
            if 0 in values and counts[values == 0] > 10:
                black_images.append(filename)
        self.train_images = [x for x in self.train_images if x not in black_images]
        print(f"Removed {len(black_images)} black images")
            

    def __len__(self):
        return len(self.test_images)
    
    def _transform(self, image):
        image = Image.fromarray(image.permute(1, 2, 0).numpy().astype("uint8"))
        image = self.transform(image)
        return image
        
    def extract_prompts(self, prompt_images=None):
        prompt_images = prompt_images or self.prompt_images
        if isinstance(prompt_images, int):
            prompt_images = hierarchical_uniform_sampling(self.train_len()-1, prompt_images)
            prompt_images = [self.train_images[i] for i in prompt_images]
        images = [
            self._get_image(self.train_channels_folder, filename)
            for filename in prompt_images
        ]
        sizes = torch.stack([torch.tensor(x.shape[1:]) for x in images])
        images = [
            self._transform(image)
            for image in images
        ]
        images = torch.stack(images)
        masks = [
            self._get_gt(self.train_gt_folder, filename)
            for filename in prompt_images
        ]
        image_ids = [prompt_images]
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
            BatchKeys.IMAGE_IDS: image_ids
        }
        return prompt_dict
    
    def _get_image(self, channels_folder, filename):
        channels = []
        for channel_folder in channels_folder:
            channel_path = os.path.join(channel_folder, filename)
            channel = torchvision.io.read_image(channel_path)
            channels.append(channel)
        channels = torch.cat(channels).float()
        return channels
    
    def _get_gt(self, gt_folder, img_filename):
        field_id = gt_folder.split("/")[-2]
        gt_filename = f"{field_id}_{img_filename.split('.')[0]}_GroundTruth_iMap.png"
        path = os.path.join(gt_folder, gt_filename)
        gt = Image.open(path)
        gt = torchvision.transforms.PILToTensor()(gt)[0].long()
        # Convert crop value 10000 to 1
        gt[gt == 10000] = 1
        return gt.long()

    def get_sample(self, i, split):
        if split == "train":
            metadata = self.train_images
        elif split == "test":
            metadata = self.test_images
        else:
            raise NotImplementedError
        
        filename = metadata[i]
        gt = self._get_gt(self.test_gt_folder, filename)
        image = self._get_image(self.test_channels_folder, filename)
        size = torch.tensor(image.shape[1:]) # Example dimension
        image = self._transform(image)
        return {
            BatchKeys.IMAGES: image.unsqueeze(0),
            BatchKeys.DIMS: size,
            BatchKeys.IMAGE_IDS: [filename],
        }, gt

    def __getitem__(self, i):
        return self.get_sample(i, "test")
    
    def train_len(self):
        return len(self.train_images)