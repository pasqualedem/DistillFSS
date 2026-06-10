import torch
import torch.nn.functional as F
from easydict import EasyDict

from distillfss.data.utils import BatchKeys
from distillfss.utils.utils import ResultDict
from .segic import SEGIC, build_model as _build_segic_model
from .distillator import DistilledSegIC, build_segic_distiller


class SegICMultiClass(SEGIC):
    """SEGIC adapted to the project's multi-class FSS batch format.

    Accepts a standard project batch (BatchKeys) and returns {ResultDict.LOGITS: logits}
    with shape (B, C+1, H, W) where C is the number of foreground classes.
    Assumes batch size B=1 (consistent with other models in this codebase).

    Optionally supports per-class projectors for specializing shared correspondences.
    """

    def postprocess_masks(self, logits, dims):
        max_dims = torch.max(dims.view(-1, 2), 0).values.tolist()
        dims = dims[:, 0, :]  # query image original sizes (B, 2)
        logits = [
            F.interpolate(
                logit.unsqueeze(0),
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
                    (0, max_dims[1] - dims[i, 1], 0, max_dims[0] - dims[i, 0]),
                    mode="constant",
                    value=float("-inf"),
                )
                for i, mask in enumerate(logits)
            ]
        )
        return logits

    def forward(self, x):
        # Remove background channel: (B, N, C, H, W)
        prompt_masks = x[BatchKeys.PROMPT_MASKS][:, :, 1:, ...]
        query_img = x[BatchKeys.IMAGES][:, 0]        # (B, 3, H, W)
        support_imgs = x[BatchKeys.IMAGES][:, 1:]    # (B, N, 3, H, W)
        dims = x[BatchKeys.DIMS]                     # (B, N+1, 2)
        bsz = query_img.shape[0]

        query_ori_sizes = dims[:, 0, :]      # (B, 2)
        support_ori_sizes = dims[:, 1:, :]   # (B, N, 2)

        # Extract query image features
        image_feat, _ = self.extract_dift_feature(
            query_img, query_ori_sizes, None, inference=True,
            is_inst_list=[False] * bsz,
        )
        vision_dim = image_feat.shape[1]

        n_classes = prompt_masks.size(2)
        all_input_prompts = []
        all_inst_feats = []

        # Optionally apply per-class projectors to query features
        query_feat = image_feat
        if hasattr(self, 'class_projectors'):
            query_feat = torch.stack([
                self.class_projectors[c](image_feat) for c in range(n_classes)
            ]).mean(0)

        for c in range(n_classes):
            # (B, N) boolean flags for which support images contain class c
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            c_examples = class_examples[0]  # (N,) — B=1
            n_shots = int(c_examples.sum().item())

            if n_shots == 0:
                all_input_prompts.append(image_feat.new_zeros(1, 1, 256))
                all_inst_feats.append(query_feat.new_zeros(1, 1, vision_dim))
                continue

            c_imgs = support_imgs[0][c_examples]           # (n_shots, 3, H, W)
            c_masks = prompt_masks[0, :, c][c_examples]   # (n_shots, H, W)
            c_ori_sizes = support_ori_sizes[0][c_examples] # (n_shots, 2)

            # Extract visual and text features for support images
            support_feat, support_prompt = self.extract_dift_feature(
                c_imgs, c_ori_sizes, None, inference=True,
                is_inst_list=[False] * n_shots,
            )
            support_prompt = self.prompt_proj(support_prompt)  # (n_shots, 256)

            # Apply class projector to support features if enabled
            if hasattr(self, 'class_projectors'):
                support_feat = self.class_projectors[c](support_feat)

            # Mask-pooled instance feature per support image
            inst_feat = self.extract_inst_feat(support_feat, c_masks[:, None])  # (n_shots, vision_dim)

            # Average prototypes over shots → (1, 1, D)
            all_input_prompts.append(support_prompt.mean(0)[None, None])
            all_inst_feats.append(inst_feat.mean(0)[None, None])

        input_prompts = torch.cat(all_input_prompts)  # (n_classes, 1, 256)
        inst_feats = torch.cat(all_inst_feats)        # (n_classes, 1, vision_dim)
        semseg_meta = (input_prompts, inst_feats)

        # is_inst_list must be a tensor so is_inst_list[i].expand(...) works
        is_inst = torch.zeros(bsz, dtype=torch.bool, device=query_img.device)

        masks_pred, _, _, _ = self.maks_decoding(
            True, query_ori_sizes, query_feat, None,
            None, None, None, [n_classes],
            semseg_meta, None, is_inst_list=is_inst,
        )
        # masks_pred: (n_classes, 1, H, W) raw logits, already cropped to query ori_size

        fg = masks_pred.sigmoid().squeeze(1).unsqueeze(0)  # (1, n_classes, H, W)
        bg = 1 - fg.max(dim=1, keepdim=True)[0]            # (1, 1, H, W)
        logits = torch.cat([bg, fg], dim=1)                 # (1, n_classes+1, H, W)

        logits = self.postprocess_masks(logits, dims)

        return {ResultDict.LOGITS: logits}


def build_segic(
    encoder_model="dinov2",
    dinov2_model="l",
    model_checkpoint=None,
    use_class_projectors=False,
    num_classes=2,
    # inference-relevant SEGIC args with sensible defaults
    use_inst_proj=True,
    use_cross_inst_prompt=False,
    no_sim_prompt=False,
    no_text=False,
    no_text_eval=True,
    diff_text_prompt_ratio=0.5,
    use_keypoint=False,
    num_keypoint=1,
    add_neg_prompt=False,
    use_aug_inst=False,
    use_neg_aug_inst=False,
    use_task_indicator=False,
    inst_for_simm=1,
    tau_simm=1.0,
    n_point=1,
    reverse_context=False,
    que_len=50,
    max_inst_used=5,
    force_input_size=None,
    checkpoint=None,  # SAM backbone checkpoint (only for encoder_model='sam')
):
    args = EasyDict(
        encoder_model=encoder_model,
        dinov2_model=dinov2_model,
        use_inst_proj=use_inst_proj,
        use_cross_inst_prompt=use_cross_inst_prompt,
        no_sim_prompt=no_sim_prompt,
        no_text=no_text,
        no_text_eval=no_text_eval,
        diff_text_prompt_ratio=diff_text_prompt_ratio,
        use_keypoint=use_keypoint,
        num_keypoint=num_keypoint,
        add_neg_prompt=add_neg_prompt,
        use_aug_inst=use_aug_inst,
        use_neg_aug_inst=use_neg_aug_inst,
        use_task_indicator=use_task_indicator,
        inst_for_simm=inst_for_simm,
        tau_simm=tau_simm,
        n_point=n_point,
        reverse_context=reverse_context,
        que_len=que_len,
        max_inst_used=max_inst_used,
        force_input_size=force_input_size,
        checkpoint=checkpoint,
    )

    model = _build_segic_model(args)
    model.__class__ = SegICMultiClass

    if use_class_projectors:
        vision_dim = model.vision_dim if hasattr(model, 'vision_dim') else None
        model.class_projectors = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(vision_dim, vision_dim, kernel_size=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(vision_dim, vision_dim, kernel_size=1),
            )
            for _ in range(num_classes)
        ])

    if model_checkpoint is not None:
        state_dict = torch.load(model_checkpoint, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        errors = model.load_state_dict(state_dict, strict=False)

    return model
