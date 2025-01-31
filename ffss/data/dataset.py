import random
import torch
import itertools

from typing import Any, Dict, List, Tuple
from torch.utils.data import Dataset, BatchSampler

import ffss.data.utils as utils
from ffss.data.coco import CocoLVISDataset
from ffss.data.coco20i import Coco20iDataset
from ffss.data.pascal import PascalDataset
from ffss.data.pascal5i import Pascal5iDataset
from ffss.utils.logger import get_logger

logger = get_logger(__name__)


datasets = {
    "coco": CocoLVISDataset,
    "val_coco": CocoLVISDataset,
    "coco20i": Coco20iDataset,
    "val_coco20i": Coco20iDataset,
    "lvis": CocoLVISDataset,
    "pascal": PascalDataset,
    "pascal5i": Pascal5iDataset,
    "val_pascal5i": Pascal5iDataset,
    "val_lvis": CocoLVISDataset,
}

def collate_fn(
    batched_input: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """
    Performs the collate_fn, which is useful for batching data points in a dataloader.

    Args:
        batched_input (List[Dict[str, Any]]): A list of batch_size elements, where each element is a dictionary
            containing the following entries:
            - 'target': query image as a torch tensor of shape 3 x H x W.
            - 'examples': example image as a torch tensor of shape M x 3 x H x W, where M is the number of examples
                extracted for the given query image.
            - 'prompt_mask': example image masks as a torch tensor of shape M x C x H x W, where M is the number of
                examples extracted for the given query image and C is the number of classes associated with it.
            - 'prompt_coords': example image coordinates as a torch tensor of shape M x C x N x K x 2, where M is the
                number of examples extracted for the given query image, C is the number of classes associated with the
                given image, N is the maximum number of annotations associated with a pair (image, class), and K is
                the number of points extracted.
            - 'flag_coords': example image coordinate flags as a torch tensor of shape M x C x N x K, where M is the
                number of examples extracted for the given query image, C is the number of classes associated with the
                given image, N is the maximum number of annotations associated with a pair (image, class), and K is
                the number of points extracted.
            - 'prompt_bbox': example image bounding boxes as a torch tensor of shape M x C x N x 4, where M is the
                number of examples extracted for the given query image, C is the number of classes associated with the
                given image, and N is the maximum number of annotations associated with a pair (image, class). The
                last dimension is 4 because a single bounding box is represented by the top-left and bottom-right
                coordinates.
            - 'flag_bbox': example image bounding box flags as a torch tensor of shape M x C x N x 4, where M is the
                number of examples extracted for the given query image, C is the number of classes associated with the
                given image, and N is the maximum number of annotations associated with a pair (image, class).
            - 'gt': query image classes mask as a tensor of shape H x W, in which each pixel has a certain value k if
                that pixel is in the mask of the k-th class associated with the query image.
            - 'classes': dictionary in which each pair k: v represents the ith class corresponding to class id.

    Returns:
        Tuple[Dict[str, Any], torch.Tensor]: A tuple containing the batched dictionary and the batched output masks.
            The batched dictionary has the following entries:
            - 'query_image': query image as a torch tensor of shape B x 3 x H x W.
            - 'example_images': example images as a torch tensor of shape B x M x 3 x H x W.
            - 'point_coords': example image coordinates as a torch tensor of shape B x M x C x N x K x 2, where M is
                the number of examples extracted for the given query image, C is the number of classes associated with
                the given image, N is the maximum number of annotations associated with a pair (image, class), and K
                is the number of points extracted.
            - 'point_flags': example image coordinate flags as a torch tensor of shape B x M x C x N x K, where M is
                the number of examples extracted for the given query image, C is the number of classes associated with
                the given image, N is the maximum number of annotations associated with a pair (image, class), and K
                is the number of points extracted.
            - 'boxes': example image bounding boxes as a torch tensor of shape B x M x C x N x 4, where M is the
                number of examples extracted for the given query image, C is the number of classes associated with the
                given image, and N is the maximum number of annotations associated with a pair (image, class). The
                last dimension is 4 because a single bounding box is represented by the top-left and bottom-right
                coordinates.
            - 'box_flags': example image bounding box flags as a torch tensor of shape B x M x C x N x 4, where M is
                the number of examples extracted for the given query image, C is the number of classes associated with
                the given image, and N is the maximum number of annotations associated with a pair (image, class).
            - 'mask_inputs': example image masks as a torch tensor of shape B x M x C x H x W, where M is the number
                of examples extracted for the given query image and C is the number of classes associated with it.
        The batched output masks is a torch tensor of shape B x H x W.
    """
    batched_input, dataset_names = zip(*batched_input)
    # classes
    max_classes = max([x["prompt_masks"].size(1) for x in batched_input])

    # gt
    dims = torch.stack([x["dims"] for x in batched_input])
    max_dims = torch.max(dims.view(-1, 2), 0).values.tolist()
    ground_truths = [x["ground_truths"] for x in batched_input]
    ground_truths = torch.stack(
        [utils.collate_batch_gts(x, max_dims) for x in ground_truths]
    )

    # prompt mask
    masks = [x["prompt_masks"] for x in batched_input]
    flags = [x["flag_masks"] for x in batched_input]
    masks_flags = [
        utils.collate_mask(m, f, max_classes) for (m, f) in zip(masks, flags)
    ]
    masks = torch.stack([x[0] for x in masks_flags])
    flag_masks = torch.stack([x[1] for x in masks_flags])

    # flag examples
    flag_examples = torch.stack(
        [
            utils.collate_example_flags(x["flag_examples"], max_classes)
            for x in batched_input
        ]
    )

    # aux gts
    classes = [x["classes"] for x in batched_input]

    # image ids
    image_ids = [x["image_ids"] for x in batched_input]

    # flag_gts
    flag_gts = torch.zeros((len(batched_input), max_classes), dtype=torch.bool)
    for i, x in enumerate(classes):
        flag_gts[i, : len(list(set(itertools.chain(*x)))) + 1] = 1

    # images
    images = torch.stack([x["images"] for x in batched_input])

    data_dict = {
        "images": images,
        "prompt_masks": masks,
        "flag_masks": flag_masks,
        "flag_examples": flag_examples,
        "dims": dims,
        "classes": classes,
        "image_ids": image_ids,
        "flag_gts": flag_gts,
    }

    return (data_dict, ground_truths), dataset_names
