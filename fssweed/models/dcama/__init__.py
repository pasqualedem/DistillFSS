import torch


from .dcama import DCAMAMultiClass
from .distillator import DistilledDCAMA


def build_dcama(
    backbone: str = "swin",
    backbone_checkpoint: str = "checkpoints/backbone.pth",
    model_checkpoint: str = "checkpoints/dcama.pth",
    image_size: int = 384,
    concat_support: bool = True,
    custom_preprocess: bool = False,
    train_backbone: bool = False,
):
    model = DCAMAMultiClass(
        backbone, backbone_checkpoint, use_original_imgsize=False, image_size=image_size, concat_support=concat_support, train_backbone=train_backbone
    )
    params = model.state_dict()
    state_dict = torch.load(model_checkpoint)

    for k1, k2 in zip(list(state_dict.keys()), params.keys()):
        state_dict[k2] = state_dict.pop(k1)

    model.load_state_dict(state_dict)
    return model


def build_dcama_distiller(
    teacher,
    num_classes
):
    return DistilledDCAMA(num_classes=num_classes, dcama=teacher)

