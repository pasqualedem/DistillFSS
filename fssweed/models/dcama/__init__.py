import torch


from .dcama import DCAMAMultiClass
from .distillator import DistilledDCAMA


def build_dcama(
    backbone: str = "swin",
    backbone_checkpoint: str = "checkpoints/backbone.pth",
    model_checkpoint: str = "checkpoints/dcama.pth",
    image_size: int = 384,
    concat_support: bool = True,
    train_backbone: bool = False,
    pe=True,
):
    model = DCAMAMultiClass(
        backbone,
        backbone_checkpoint,
        use_original_imgsize=False,
        image_size=image_size,
        concat_support=concat_support,
        train_backbone=train_backbone,
        pe=pe,
    )
    state_dict = torch.load(model_checkpoint)
    
    if pe:
        print("Using pe from state_dict")
    else:
        print("Removing pe from state_dict")
        state_dict = {k: v for k, v in state_dict.items() if ".pe." not in k}
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
    return model


def build_dcama_distiller(teacher, num_classes):
    return DistilledDCAMA(num_classes=num_classes, dcama=teacher)
