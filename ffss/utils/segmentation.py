import numpy as np
import plotly.express as px
import torch

from torchvision.transforms.functional import resize
from einops import rearrange

from ffss.data.utils import BatchKeys


class ColorMap:
    def __init__(self):
        self.cmap = [
            "#000000",
            "#00ff00",
            "#ff0000",
            "#0000ff",
        ] + px.colors.qualitative.Alphabet
        self.cmap = [
            tuple(int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
            for h in self.cmap
        ]

    def __getitem__(self, item):
        return self.cmap[item]


def tensor_to_segmentation_image(
    prediction, cmap: list = None, labels=None, return_clmap=False
) -> np.array:
    if cmap is None:
        cmap = ColorMap()
    if labels is None:
        labels = np.unique(prediction)
    segmented_image = np.ones((*prediction.shape, 3), dtype="uint8")
    for i in range(len(labels)):
        segmented_image[prediction == i] = cmap[i]
    if return_clmap:
        cmap = {labels[i]: cmap[i] for i in range(len(labels))}
        return segmented_image, cmap
    return segmented_image


def create_rgb_segmentation(segmentation, num_classes=None):
    """
    Convert a segmentation map to an RGB visualization using a precise colormap.

    Args:
        segmentation (torch.Tensor): Segmentation map of shape [B, H, W] where
                                      each pixel contains class labels (natural numbers).
        num_classes (int): The number of unique classes in the segmentation.

    Returns:
        torch.Tensor: RGB visualization of shape [B, 3, H, W].
    """
    if len(segmentation.shape) == 4:
        segmentation = segmentation.argmax(dim=1)
    if num_classes is None:
        num_classes = segmentation.max().item() + 1
    
    # Define a precise colormap for specific classes
    colormap = torch.tensor([
        [0, 0, 0],       # Class 0: Black (Background)
        [0, 128, 0],     # Class 2: Green
        [128, 0, 0],     # Class 1: Red
        [128, 128, 0],   # Class 3: Yellow
        [0, 0, 128],     # Class 4: Blue
        [128, 0, 128],   # Class 5: Magenta
        [0, 128, 128],   # Class 6: Cyan
        [192, 192, 192], # Class 7: Light Gray
    ], dtype=torch.uint8)  # Ensure dtype is uint8

    # Initialize an empty tensor for RGB output
    B, H, W = segmentation.shape
    rgb_segmentation = torch.zeros((B, 3, H, W), dtype=torch.uint8)

    # Loop through each class and assign the corresponding RGB color
    for class_id in range(num_classes):
        # Create a mask for the current class
        class_mask = (segmentation == class_id).unsqueeze(1)  # Shape: [B, 1, H, W]
        # Assign the corresponding color to the rgb_segmentation
        rgb_segmentation += class_mask * colormap[class_id].view(1, 3, 1, 1)  # Broadcasting

    return rgb_segmentation


def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Unnormalize a tensor image with mean and standard deviation.
    
    Args:
        tensor (torch.Tensor): Tensor image of size [B, 3, H, W] to be unnormalized.
        mean (list or tuple): Mean for each channel.
        std (list or tuple): Standard deviation for each channel.
    
    Returns:
        torch.Tensor: Unnormalized tensor image.
    """
    # Ensure the mean and std are tensors and have the correct shape
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    
    # Unnormalize the tensor
    tensor = tensor * std + mean
    
    return tensor

def batch_visualizer(batch, gt, pred=None):
    assert batch[BatchKeys.PROMPT_MASKS].shape[0] == 1
    masks = batch[BatchKeys.PROMPT_MASKS][0]
    images = batch[BatchKeys.IMAGES][0]
    num_classes = masks.shape[1]
    masks = masks.argmax(dim=1)
    masks = create_rgb_segmentation(masks.cpu(), num_classes=num_classes)
    drawn_gt = create_rgb_segmentation(gt.cpu(), num_classes=num_classes)
    images = torch.cat([unnormalize(image) for image in images]).cpu()
    images = resize(images, gt.shape[-2:])
    masks = torch.cat([drawn_gt, masks], dim=0)
    drawn_batch = torch.stack([images, masks], dim=1)
    if pred is not None:
        drawn_pred = create_rgb_segmentation(pred.cpu(), num_classes=num_classes)
        padding = torch.ones_like(images.unsqueeze(1))
        padding[0] = drawn_pred
        drawn_batch = torch.cat([drawn_batch, padding], dim=1)
    drawn_batch = rearrange(drawn_batch, 'b t c h w -> b c h (t w)')
    return drawn_batch
    