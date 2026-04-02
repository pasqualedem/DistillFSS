# Code taken from https://github.com/facebookresearch/segment-anything
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch

from collections import namedtuple
from torchvision.models import resnet50
from transformers import AutoImageProcessor

from distillfss.models.bam import build_bam
from distillfss.models.la.build_lam import build_lam, build_lam_vit_mae_b

from .image_encoder import ImageEncoderViT
from .build_encoder import ENCODERS, build_vit_b, build_vit_h, build_vit_l
from .dcama import build_dcama, build_dcama_distiller, build_attn_distiller, build_weeddcama
from .dummy import build_dummy
from .dmtnet import build_dmtnet, build_dmtnet_distiller
from .msdnet import build_msdnet, build_msdnet_distiller
from .insid3 import build_insid3, build_insid3_distiller
from .patnet import build_patnet
from .TVGTANet import build_tvgtanet
from .restnet import build_restnet
from .pahnet import build_pahnet

ComposedOutput = namedtuple("ComposedOutput", ["main", "aux"])


def build_deit():
    processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model = torch.hub.load('facebookresearch/deit:main',
                           'deit_tiny_patch16_224', pretrained=True)
    return model, processor


def build_resnet50():
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = resnet50(pretrained=True)
    return model, processor


MODEL_REGISTRY = {
    "dcama": build_dcama,
    "weeddcama": build_weeddcama,
    "dummy": build_dummy,
    "deit": build_deit,
    "dmtnet": build_dmtnet,
    "msdnet": build_msdnet,
    "insid3": build_insid3,
    "patnet": build_patnet,
    "resnet50": build_resnet50,
    "restnet": build_restnet,
    "bam": build_bam,
    "la": build_lam_vit_mae_b,
    "tvgtanet": build_tvgtanet,
    "pahnet": build_pahnet,
    # Encoders only
    **ENCODERS,
}

STUDENT_REGISTRY = {
    "conv_distillator": build_dcama_distiller,
    "attn_distillator": build_attn_distiller,
    "dmtnet_distillator": build_dmtnet_distiller,
    "msdnet_distillator": build_msdnet_distiller,
    "insid3_distillator": build_insid3_distiller
}

def build_model(params):
    name = params["name"]
    
    if name in ["hdmnet"]:
        # mmcv is needed, check if it is installed
        try:
            import mmcv
        except ImportError:
            raise ImportError("mmcv is required to use hdmnet. Please install it")
        from .hdmnet import build_hdmnet
        MODEL_REGISTRY["hdmnet"] = build_hdmnet
    
    params = {k: v for k, v in params.items() if k != "name"}
    return MODEL_REGISTRY[name](**params)


def build_distillator(params):
    teacher = params["teacher"]
    teacher = build_model(teacher)
    
    student_params = params["student"]
    name = student_params["name"]
    student_params = {k: v for k, v in student_params.items() if k != "name"}
    student_params["teacher"] = teacher
    return STUDENT_REGISTRY[name](**student_params)
    

MODEL_REGISTRY["distillator"] = build_distillator
