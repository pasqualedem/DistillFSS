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
        return RefineDistillationLoss(**kwargs)
    else:
        raise NotImplementedError


class FocalLoss(nn.Module):
    def __init__(
        self, gamma: float = 2.0, weights=None
    ):
        super().__init__()
        self.gamma = gamma
        if weights is not None and not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
        self.weights = weights

        self.reduction = torch.mean

    def __call__(self, result, target):
        logits = result[ResultDict.LOGITS]
        ce_loss = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce_loss)

        if self.weights is not None:
            weights = self.weights.to(logits.device)[target]
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
    def __init__(self, alpha=1/3, beta=1/3, gamma=1/3, delta=0.0, weights=None,
                 focal_gamma=2.0, kd_temp=2.0, binary_fg_weight=None):
        super().__init__()
        self.logits_loss = FocalLoss(gamma=focal_gamma, weights=weights)
        self.feature_loss = nn.MSELoss()
        # For the "independent" (decoupled multi-class) mode: each class head is
        # supervised as a BINARY [bg, fg_c] segmenter, so heads don't compete in a
        # joint softmax (which caused winner-take-all collapse of 2/3 classes).
        self.binary_loss = FocalLoss(
            gamma=focal_gamma,
            weights=None if binary_fg_weight is None else [1.0, float(binary_fg_weight)],
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # delta: classic output/logit distillation (KL of softened student logits
        # to the refined teacher's logits). The feature term (gamma) only matches
        # pre-decoder coarse maps; delta transfers the teacher's final per-class
        # DISCRIMINATION -- the piece multi-class (ISIC/Industrial) needs and that
        # GT-only supervision can't teach from ~3 examples/class.
        self.delta = delta
        self.kd_temp = kd_temp

    def forward(self, result, target):
        logits = result[ResultDict.LOGITS]  # teacher (combined) during training
        logits_loss = self.logits_loss({ResultDict.LOGITS: logits}, target)

        if ResultDict.DISTILLED_PER_CLASS in result:
            # decoupled: per-class binary CE, target_c = (gt == c+1). No cross-class
            # competition, so no collapse. distilled_logits stays None (no joint KD).
            per_class = result[ResultDict.DISTILLED_PER_CLASS]  # [B, C, 2, H, W]
            distilled_logits = None
            n_c = per_class.shape[1]
            per_losses = [
                self.binary_loss(
                    {ResultDict.LOGITS: per_class[:, c]}, (target == (c + 1)).long()
                )
                for c in range(n_c)
            ]
            distilled_logits_loss = torch.stack(per_losses).mean()
        else:
            distilled_logits = result[ResultDict.DISTILLED_LOGITS]
            distilled_logits_loss = self.logits_loss({ResultDict.LOGITS: distilled_logits}, target)

        coarse_maps = filter(lambda x: x is not None, result[ResultDict.COARSE_MASKS])
        distilled_coarse_maps = filter(lambda x: x is not None, result[ResultDict.DISTILLED_COARSE])

        feature_loss = [
            sum((self.feature_loss(c, d) if c is not None and d is not None else torch.tensor(0.0, device=logits.device)) for c, d in zip(cm, dm)) / len(cm)
            for cm, dm
            in zip(coarse_maps, distilled_coarse_maps)
        ]
        if feature_loss:
            feature_loss = torch.mean(torch.stack(feature_loss))
        else:
            feature_loss = torch.tensor(0.0, device=logits.device)

        # output/logit distillation: student (log-softmax) -> teacher (softmax, detached)
        if self.delta > 0 and distilled_logits is not None:
            T = self.kd_temp
            t = logits.detach()
            kd_loss = F.kl_div(
                F.log_softmax(distilled_logits / T, dim=1),
                F.softmax(t / T, dim=1),
                reduction="none",
            ).sum(dim=1).mean() * (T * T)
        else:
            kd_loss = torch.tensor(0.0, device=logits.device)

        denom = self.alpha + self.beta + self.gamma + self.delta
        return (self.alpha * logits_loss + self.beta * distilled_logits_loss
                + self.gamma * feature_loss + self.delta * kd_loss) / denom