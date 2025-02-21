import torch


from .dcama import DCAMAMultiClass, WeedDCAMA
from .distillator import DistilledDCAMA, AttentionDistilledDCAMA


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

def build_weeddcama(
    backbone: str = "swin",
    backbone_checkpoint: str = "checkpoints/backbone.pth",
    model_checkpoint: str = "checkpoints/weedcama.pth",
    image_size: int = 384,
    concat_support: bool = True,
    train_backbone: bool = False,
    pe=True,
):
    model = WeedDCAMA(
        backbone,
        backbone_checkpoint,
        use_original_imgsize=False,
        image_size=image_size,
        concat_support=concat_support,
        train_backbone=train_backbone,
        pe=pe,
    )
    state_dict = torch.load(model_checkpoint)
    
    # Get .model from state_dict
    dcama_model_state_dict = {k: v for k, v in state_dict.items() if ".model." in k}
    
    remaining_model_state_dict = {k: v for k, v in state_dict.items() if ".model." not in k}
    
    # model_0 state_dict
    model_0_state_dict = {k.replace(".model.", ".model_0."): v.clone() for k, v in dcama_model_state_dict.items()}
    model_1_state_dict = {k.replace(".model.", ".model_1."): v.clone() for k, v in dcama_model_state_dict.items()}
    
    state_dict = {**remaining_model_state_dict, **model_0_state_dict, **model_1_state_dict}
    
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

def build_attn_distiller(teacher, num_classes):
    return AttentionDistilledDCAMA(num_classes=num_classes, dcama=teacher)
