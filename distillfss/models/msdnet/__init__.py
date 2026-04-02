import torch
import torch.nn.functional as F

from distillfss.data.utils import BatchKeys
from distillfss.utils.utils import ResultDict

from .msd import MSDNet
from .distillator import build_msdnet_distiller
from .distillator import build_msdnet_distiller


def build_msdnet(
    layers: int = 50,
    shot: int = 1,
    reduce_dim: int = 64,
    model_checkpoint: str = "checkpoints/msdnet_pascal0_1s.pt",
    strict: bool = True,
):
    model = MSDNetMultiClass(layers=layers, shot=shot, reduce_dim=reduce_dim)

    if model_checkpoint is not None:
        src_dict = torch.load(model_checkpoint, map_location="cpu")
        if isinstance(src_dict, dict) and "state_dict" in src_dict:
            src_dict = src_dict["state_dict"]

        try:
            model.load_state_dict(src_dict, strict=strict)
        except RuntimeError:
            # Some checkpoints are saved from DDP and include a "module." prefix.
            src_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in src_dict.items()}
            model.load_state_dict(src_dict, strict=strict)

    return model


class MSDNetMultiClass(MSDNet):
    def __init__(self, *args, **kwargs):
        self.predict = None
        self.generate_class_embeddings = None
        super().__init__(*args, **kwargs)

    def _preprocess_masks(self, masks, dims):
        # Remove background channel, keep only foreground class masks.
        return masks[:, :, 1:, ...]

    def forward(self, x):
        masks = self._preprocess_masks(x[BatchKeys.PROMPT_MASKS], x[BatchKeys.DIMS])
        assert masks.shape[0] == 1, "Only tested with batch size = 1"

        query_img = x[BatchKeys.IMAGES][:, 0]
        bsz, _, qry_h, qry_w = query_img.shape
        class_probs = []
        coarse_masks = []

        original_shot = self.shot
        try:
            for c in range(masks.size(2)):
                class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
                n_shots = int(class_examples.sum().item())

                if n_shots == 0:
                    class_probs.append(torch.zeros((bsz, 1, qry_h, qry_w), device=query_img.device, dtype=query_img.dtype))
                    continue

                class_support_imgs = x[BatchKeys.IMAGES][:, 1:][class_examples].unsqueeze(0)
                class_support_masks = masks[:, :, c, ...][class_examples].unsqueeze(0)

                self.shot = n_shots
                out = super().forward(query_img, class_support_imgs, class_support_masks)
                class_probs.append(out[ResultDict.LOGITS])
                coarse_masks.append([out[ResultDict.COARSE_MASKS][0], out[ResultDict.COARSE_MASKS][1], out[ResultDict.COARSE_MASKS][2]])
        finally:
            self.shot = original_shot

        fg_probs = torch.cat(class_probs, dim=1)
        if fg_probs.size(1) == 1:
            logits = torch.cat([1 - fg_probs, fg_probs], dim=1)
        else:
            fg_logits = F.softmax(fg_probs, dim=1)
            bg_logits = (1 - fg_probs.clamp(0, 1)).prod(dim=1, keepdim=True)
            logits = torch.cat([bg_logits, fg_logits], dim=1)

        logits = self.postprocess_masks(logits, x[BatchKeys.DIMS])

        return {
            ResultDict.LOGITS: logits,
            ResultDict.COARSE_MASKS: coarse_masks,
        }

    def postprocess_masks(self, logits, dims):
        max_dims = torch.max(dims.view(-1, 2), 0).values.tolist()
        dims = dims[:, 0, :]
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