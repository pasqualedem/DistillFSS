import torch
import torch.nn as nn
import torch.nn.functional as F

from fssweed.utils.utils import ResultDict


def get_loss(params):
    name = params["name"]
    kwargs = {k: v for k, v in params.items() if k != "name"}
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif name == "focal":
        return FocalLoss(**kwargs)
    elif name == "distill":
        return DistillationLoss()
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
    
    
class DistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits_loss = FocalLoss()
        self.feature_loss = nn.MSELoss()
    def forward(self, result):
        distilled_logits = result[ResultDict.DISTILLED_LOGITS]
        logits = result[ResultDict.LOGITS]
        
        logits_loss = self.logits_loss(distilled_logits, logits.argmax(dim=1))
        
        coarse_maps = result[ResultDict.COARSE_MASKS]
        distilled_coarse_maps = result[ResultDict.DISTILLED_COARSE]
        
        feature_loss = [
            (self.feature_loss(cm1, dm1) + self.feature_loss(cm2, dm2) + self.feature_loss(cm3, dm3)) / 3
            for (cm1, cm2, cm3), (dm1, dm2, dm3)
            in zip(coarse_maps, distilled_coarse_maps)
        ]
        feature_loss = torch.mean(torch.stack(feature_loss))
        
        return (logits_loss + feature_loss) / 2