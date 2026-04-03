import torch
import torch.nn.functional as F
from einops import rearrange

from distillfss.data.utils import BatchKeys
from distillfss.utils.utils import ResultDict
from .insid3 import INSID3
from .distillator import build_insid3_distiller

_HUB_NAMES = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
}

_WEIGHTS = {
    "small": "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "base": "checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


class ConvAdapter(torch.nn.Module):
    def __init__(self, channels, depth=2):
        super().__init__()
        conv_modules = []
        for _ in range(depth - 1):
            conv_modules.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                )
            )
        conv_modules.append(torch.nn.Conv2d(channels, channels, kernel_size=1))
        self.conv = torch.nn.Sequential(*conv_modules)

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = rearrange(x, "b n c h w -> (b n) c h w")
        x = self.conv(x)
        x = rearrange(x, "(b n) c h w -> b n c h w", b=b)
        return x


def _build_encoder(model_size: str = "large"):
    return torch.hub.load(
        "facebookresearch/dinov3",
        _HUB_NAMES[model_size],
        weights=_WEIGHTS[model_size],
    )


def build_insid3(
    *,
    model_size: str = "large",
    image_size: int = 1024,
    svd_components: int = 500,
    tau: float = 0.6,
    merge_threshold: float = 0.2,
    mask_refiner: str = "bilinear",
    device: str = "cuda",
    adapter: str = None,
    differentiable: bool = False,
):
    encoder = _build_encoder(model_size)
    
    if adapter == "conv":
        adapter_module = ConvAdapter(channels=encoder.embed_dim, depth=2)
    else:        
        adapter_module = None
    
    model = INSID3MultiClass(
        encoder=encoder,
        image_size=image_size,
        svd_components=svd_components,
        tau=tau,
        merge_threshold=merge_threshold,
        mask_refiner=mask_refiner,
        device=device,
        adapter=adapter_module,
        differentiable=differentiable,
    )
    for name, param in model.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
    return model


class INSID3MultiClass(INSID3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _preprocess_masks(self, masks, dims):
        return masks[:, :, 1:, ...]

    def forward(self, x):
        masks = self._preprocess_masks(x[BatchKeys.PROMPT_MASKS], x[BatchKeys.DIMS])
        assert masks.shape[0] == 1, "Only tested with batch size = 1"

        query_img = x[BatchKeys.IMAGES][:, 0]
        bsz, _, qry_h, qry_w = query_img.shape
        decisions = []
        coarse_masks = []
        prob_masks = []

        for c in range(masks.size(2)):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            n_shots = int(class_examples.sum().item())

            if n_shots == 0:
                decisions.append(torch.zeros((bsz, qry_h, qry_w), device=query_img.device, dtype=query_img.dtype))
                prob_masks.append(torch.zeros((bsz, 1, qry_h, qry_w), device=query_img.device, dtype=query_img.dtype))
                coarse_masks.append(None)
                continue

            class_support_imgs = x[BatchKeys.IMAGES][:, 1:][class_examples]
            class_support_masks = masks[:, :, c, ...][class_examples]
            if BatchKeys.EMBEDDINGS in x:
                class_support_embs = x[BatchKeys.EMBEDDINGS][:, 1:][class_examples]
                embeddings = torch.cat([class_support_embs, x[BatchKeys.EMBEDDINGS][:, 0]], dim=0).unsqueeze(0)
            else:
                embeddings = None
            
            # predict
            predict_res = super().predict(class_support_imgs, class_support_masks, query_img[0], return_intermediates=True, embeddings=embeddings)
            
            decisions.append(predict_res["pred_mask"].unsqueeze(0))
            
            # To be compatible with distillation
            coarse_masks.append(None)
            prob_masks.append(predict_res["prob_mask"])

        # Check if decision have the same spatial size, if not interpolate to the largest one
        max_h = max(decision.shape[1] for decision in decisions)
        max_w = max(decision.shape[2] for decision in decisions)
        if any(decision.shape[1] != max_h or decision.shape[2] != max_w for decision in decisions):
            decisions = [
                F.interpolate(decision.unsqueeze(0), size=(max_h, max_w), mode="bilinear", align_corners=False).squeeze(0)
                for decision in decisions
            ]
            prob_masks = [
                F.interpolate(prob_mask, size=(max_h, max_w), mode="bilinear", align_corners=False)
                for prob_mask in prob_masks
            ]

        decisions = torch.stack(decisions, dim=1)
        probs = torch.cat(prob_masks, dim=1)

        if self.differentiable:
            # log-sum-exp trick for stability
            logsumexp = torch.logsumexp(probs, dim=1, keepdim=True)  # [B, 1, H, W]

            # denominator = 1 + sum(exp(logits))
            # work in log-space:
            # log(den) = log(1 + sum(exp(logits))) = logaddexp(0, logsumexp)
            log_den = torch.logaddexp(torch.zeros_like(logsumexp), logsumexp)

            # foreground probabilities
            p_fg = torch.exp(probs - log_den)

            # background probability
            p_bg = torch.exp(-log_den)

            # concatenate
            logits = torch.cat([p_bg, p_fg], dim=1)  # [B, C+1, H, W]
        else:
            if decisions.size(1) == 1:
                logits = torch.cat([1 - decisions.float(), decisions.float()], dim=1)
            else:
                # Take probs to decisions size
                probs = F.interpolate(probs, size=decisions.shape[2:], mode="bilinear", align_corners=False)
                # # BG probability is 1 - max_fg_prob masked by the union of all fg decisions
                max_prob = probs.max(dim=1, keepdim=True)[0]
                bg_prob = (1 - max_prob) * (1 - decisions.any(dim=1, keepdim=True).float())
                logits = torch.cat([bg_prob, probs], dim=1)
                
                # fg_logits = F.softmax(decisions.float(), dim=1) # Note: INSID3 handles multi-class prediction by predicting per class
                # bg_logits = (1 - decisions.clamp(0, 1)).prod(dim=1, keepdim=True)
                # logits = torch.cat([bg_logits, fg_logits], dim=1)
            

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
