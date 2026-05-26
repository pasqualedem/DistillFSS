import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from distillfss.data.utils import BatchKeys
from distillfss.models.restnet.restnet import RestNetwork
from distillfss.utils.utils import ResultDict


def build_restnet(backbone="resnet50", model_checkpoint="checkpoints/restnet_fss1000-fold0-1shot.pt"):
    model = RestNetworkMultiClass(backbone)
    src_dict = torch.load(model_checkpoint, map_location="cpu")
    src_dict = {k[len("module."):]: v for k, v in src_dict.items()}
    model.load_state_dict(src_dict)
    return model


class RestNetworkMultiClass(RestNetwork):
    def __init__(self, *args, **kwargs):
        self.predict = None
        self.generate_class_embeddings = None
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
        fg_probs = []
        coarse_masks = []
        # get fg probability (B, H, W) for each class
        for c in range(masks.size(2)):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            n_shots = class_examples.sum().item()
            class_input_dict = {
                "query_img": x[BatchKeys.IMAGES][:, 0],
                "support_imgs": x[BatchKeys.IMAGES][:, 1:][class_examples].unsqueeze(0),
                "support_masks": masks[:, :, c, ::][class_examples].unsqueeze(0),
            }
            if n_shots == 1:
                logit_mask, coarse_mask = self.predict_mask_1shot(
                    class_input_dict["query_img"],
                    class_input_dict["support_imgs"][:, 0],
                    class_input_dict["support_masks"][:, 0],
                    None,
                )
                fg_probs.append(F.softmax(logit_mask, dim=1)[:, 1])
                coarse_masks.append([coarse_mask])
            elif n_shots == 0:
                fg_probs.append(torch.zeros_like(x[BatchKeys.IMAGES][:, 0, 0]))
                coarse_masks.append([None])
            else:
                voting_mask, _ = self.predict_mask_nshot(class_input_dict, n_shots)
                fg_probs.append(voting_mask)
                coarse_masks.append([None])

        votes = torch.stack(fg_probs, dim=1)  # (B, C, H, W)
        bg_logit = 1 - votes.max(dim=1, keepdim=True).values  # (B, 1, H, W)
        logits = torch.cat([bg_logit, votes], dim=1)  # (B, C+1, H, W)
            
        logits = self.postprocess_masks(logits, x["dims"])

        return {
            ResultDict.LOGITS: logits,
            ResultDict.COARSE_MASKS: coarse_masks,
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

from .distillator import DistilledRestNet

def build_restnet_distiller(teacher, num_classes, num_conv_layers=1):
    return DistilledRestNet(num_classes=num_classes, restnet=teacher, num_conv_layers=num_conv_layers)
