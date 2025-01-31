import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(params):
    name = params["name"]
    kwargs = {k: v for k, v in params.items() if k != "name"}
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif name == "focal":
        return FocalLoss(**kwargs)
    else:
        raise NotImplementedError


class FocalLoss(nn.Module):
    def __init__(
        self, gamma: float = 2.0, weights=None
    ):
        super().__init__()
        self.gamma = gamma
        self.weights = weights

        self.reduction = torch.mean

    def __call__(self, x, target):
        ce_loss = F.cross_entropy(x, target, reduction="none")
        pt = torch.exp(-ce_loss)
        
        if self.weights is not None:
            weights = self.weights.to(x.device)[target]
            focal_loss = torch.pow((1 - pt), self.gamma) * weights * ce_loss
        else:
            focal_loss = torch.pow((1 - pt), self.gamma) * ce_loss

        return self.reduction(focal_loss)