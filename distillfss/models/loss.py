import torch
import torch.nn as nn
import torch.nn.functional as F

from distillfss.utils.utils import ResultDict


def get_loss(params):
    name = params["name"]
    kwargs = {k: v for k, v in params.items() if k != "name"}
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif name == "focal":
        return FocalLoss(**kwargs)
    elif name == "distill":
        return DistillationLoss()
    elif name == "refine_distill":
        return RefineDistillationLoss()
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

    def __call__(self, result, target):
        logits = result[ResultDict.LOGITS]
        ce_loss = F.cross_entropy(logits, target, reduction="none")
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
        
        logits_loss = self.logits_loss({ResultDict.LOGITS: distilled_logits}, logits.argmax(dim=1))
        
        coarse_maps = result[ResultDict.COARSE_MASKS]
        distilled_coarse_maps = result[ResultDict.DISTILLED_COARSE]
        
        feature_loss = [
            sum(self.feature_loss(c, d) for c, d in zip(cm, dm)) / len(cm)
            for cm, dm
            in zip(coarse_maps, distilled_coarse_maps)
        ]
        feature_loss = torch.mean(torch.stack(feature_loss))
        
        return (logits_loss + feature_loss) / 2
    

class RefineDistillationLoss(nn.Module):
    def __init__(self, alpha=1/3, beta=1/3, gamma=1/3):
        super().__init__()
        self.logits_loss = FocalLoss()
        self.feature_loss = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def forward(self, result, target):
        distilled_logits = result[ResultDict.DISTILLED_LOGITS]
        logits = result[ResultDict.LOGITS]

        distilled_logits_loss = self.logits_loss({ResultDict.LOGITS: distilled_logits}, target)
        logits_loss = self.logits_loss({ResultDict.LOGITS: logits}, target)

        coarse_maps = filter(lambda x: x is not None, result[ResultDict.COARSE_MASKS])
        distilled_coarse_maps = filter(lambda x: x is not None, result[ResultDict.DISTILLED_COARSE])

        feature_loss = [
            sum(self.feature_loss(c, d) for c, d in zip(cm, dm)) / len(cm)
            for cm, dm
            in zip(coarse_maps, distilled_coarse_maps)
        ]
        if feature_loss:
            feature_loss = torch.mean(torch.stack(feature_loss))
        else:
            feature_loss = torch.tensor(0.0, device=logits.device)

        return (self.alpha * logits_loss + self.beta * distilled_logits_loss + self.gamma * feature_loss) / (self.alpha + self.beta + self.gamma)