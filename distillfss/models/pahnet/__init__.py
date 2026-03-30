import math
from easydict import EasyDict
import torch
import torch.nn.functional as F

from ...data.utils import BatchKeys
from ...utils.utils import ResultDict
from .SCCANPlus import OneModel

class PAHNetModel(OneModel):
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
    
    def forward(self, batch: dict):
        # remove bg from masks
        masks = batch[BatchKeys.PROMPT_MASKS][:, :, 1:, ::]
        y_m = None
        cat_idx = None
        logits = []
        # get logits for each class
        for c in range(masks.size(2)):
            class_examples = batch[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            x = batch[BatchKeys.IMAGES][:, 0]
            s_x = batch[BatchKeys.IMAGES][:, 1:][class_examples].unsqueeze(0)
            s_y = masks[:, :, c, ::][class_examples].unsqueeze(0)
            n_shots = class_examples.sum().item()
            rounds = math.ceil(n_shots / self.shot) # Divide the episode in n rounds
            if n_shots % self.shot: # repeat the last image and mask
                rest = self.shot - (n_shots % self.shot)
                s_x = torch.cat([s_x, s_x[:, -1].unsqueeze(0).repeat(1, rest, 1, 1, 1)], dim=1)
                s_y = torch.cat([s_y, s_y[:, -1].unsqueeze(0).repeat(1, rest, 1, 1)], dim=1)
                n_shots += rest

            if n_shots > self.shot:
                round_logits = []
                for i in range(rounds): # calculate for each round
                    start = i * self.shot
                    end = min((i + 1) * self.shot, n_shots)
                    cur_s_x = s_x[:, start:end]
                    cur_s_y = s_y[:, start:end]
                    final_out, meta_out, base_out = super().forward(x, cur_s_x, cur_s_y, y_m, cat_idx)
                    round_logits.append(meta_out)
                # Take maximum over all rounds
                round_logits = torch.stack(round_logits, dim=1).max(dim=1)[0]
                logits.append(round_logits)
            else:
                final_out, meta_out, base_out = super().forward(x, s_x, s_y, y_m, cat_idx)
                logits.append(meta_out)
        logits = torch.stack(logits, dim=1)
        fg_logits = logits[:, :, 1, ::]
        bg_logits = logits[:, :, 0, ::]
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
        logits = torch.cat([bg_logits, fg_logits], dim=1)
        logits = self.postprocess_masks(logits, batch["dims"])

        return {
            ResultDict.LOGITS: logits,
        }

def build_pahnet(dataset="coco", shots=5, val_fold_idx=0):
    args = EasyDict({
        "arch": "SCCANPlus",
        "shot": shots,
        "layers": 50,
        "vgg": False,
        "split": val_fold_idx,
        "zoom_factor": 8,
        "train_h": 473,
        "train_w": 473,
        "data_set": dataset,
        "ignore_label": 255,
        "print_freq": 10,
        "model_classes": 2, # based on original PAHNet output
        "criterion": None,
        "manual_seed": 321,
        "evaluate": True,
        "snapshot_path": "checkpoints/pahnet/pascal/split0/", # placeholder configuration
        "base_weight": "checkpoints/swin/swin_fold0_pascal.pt", # placeholder configuration
        "low_fea": "layer2",
        "kshot_trans_dim": 2
    })
    
    if shots == 1:
        snapshot_path = "checkpoints/PAHNet_train_epoch_42_0.4375.pth"
    elif shots == 5:
        snapshot_path = "checkpoints/PAHNet_train5_epoch_43.5_0.5112.pth"
    else:
        raise ValueError("Invalid number of shots for PAHNet. Only 1 and 5 are supported.")
    
    # We load the appropriate arguments based on training
    model = PAHNetModel(args)
    
    weights = torch.load(snapshot_path, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(weights)

    return model
