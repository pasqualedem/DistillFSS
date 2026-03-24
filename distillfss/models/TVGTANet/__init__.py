# Code revised from https://github.com/ljm198134/TVGTANet

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from ...data.utils import BatchKeys
from .runner import TVGTANet
from ...utils.utils import ResultDict


def build_tvgtanet(num_classes=5):
     return TVGTANetMultiClass(num_classes=num_classes)


class TVGTANetMultiClass(TVGTANet):
    def __init__(self, *args, **kwargs):
        self.predict = None
        self.generate_class_embeddings = None
        self.tta_model = True
        super().__init__(*args, **kwargs)

    def _preprocess_masks(self, masks, dims):
        B, N, C, H, W = masks.size()
        # remove bg from masks
        masks = masks[:, :, 1:, ::]
        mask_size = 256

        # Repeat dims along class dimension
        support_dims = dims[:, 1:]
        repeated_dims = repeat(support_dims, "b n d -> (b n c) d", c=C)
        masks = rearrange(masks, "b n c h w -> (b n c) h w")

        # Remove padding from masks
        # pad_dims = [get_preprocess_shape(h, w, mask_size) for h, w in repeated_dims]
        # masks = [mask[:h, :w] for mask, (h, w) in zip(masks, pad_dims)]
        # masks = torch.cat(
        #     [
        #         F.interpolate(
        #             torch.unsqueeze(mask, 0).unsqueeze(0),
        #             size=(self.image_size, self.image_size),
        #             mode="nearest",
        #         )[0]
        #         for mask in masks
        #     ]
        # )
        return rearrange(masks, "(b n c) h w -> b n c h w", b=B, n=N)

    def forward(self, x):

        masks = self._preprocess_masks(x[BatchKeys.PROMPT_MASKS], x[BatchKeys.DIMS])
        assert masks.shape[0] == 1, "Only tested with batch size = 1"
        voting_masks = []
        fg_logits_masks = []
        # get logits for each class
        for c in range(masks.size(2)):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            n_shots = class_examples.sum().item()
            class_input_dict = {
                "query_img": x[BatchKeys.IMAGES][:, 0],
                "support_imgs": x[BatchKeys.IMAGES][:, 1:][class_examples].unsqueeze(0),
                "support_masks": masks[:, :, c, ::][class_examples].unsqueeze(0),
                "query_pascal_label": torch.zeros(1, dtype=torch.long, device=masks.device),  # dummy label for query
                "support_pascal_labels": torch.ones((1, n_shots), dtype=torch.long, device=masks.device),  # dummy label for support
                "class_id": c,
            }

            logit_mask, voting_mask = (
                self.predict_mask_nshot(class_input_dict)
            )
            fg_logits_masks.append(logit_mask)
            voting_masks.append(voting_mask)
                
        if masks.size(2) == 1:
            logits = torch.stack(voting_masks, dim=1)
            bg_logits = 1 - logits
            logits = torch.cat([bg_logits, logits], dim=1)
        else:
            logits = torch.stack(fg_logits_masks, dim=1)
            logits = logits.softmax(dim=1)
            # Get background logits where all voting masks are 0
            bg_logits = (1 - torch.stack(voting_masks, dim=1)).prod(dim=1, keepdim=True)
            logits = torch.cat([bg_logits, logits], dim=1)
            
        logits = self.postprocess_masks(logits, x["dims"])

        return {
            ResultDict.LOGITS: logits,
        }

    def postprocess_masks(self, logits, dims):
        max_dims = torch.max(dims.view(-1, 2), 0).values.tolist()
        dims = dims[:, 0, :]  # get real sizes of the query images
        logits = [
            F.interpolate(
                torch.unsqueeze(logit, 0),
                size=dim.tolist(),
                mode="bilinear",
                align_corners=False,
            )
            for logit, dim in zip(logits, dims)
        ]

        logits = torch.cat(
            [
                F.pad(
                    mask,
                    (
                        0,
                        max_dims[1] - dims[i, 1],
                        0,
                        max_dims[0] - dims[i, 0],
                    ),
                    mode="constant",
                    value=float("-inf"),
                )
                for i, mask in enumerate(logits)
            ]
        )
        return logits
