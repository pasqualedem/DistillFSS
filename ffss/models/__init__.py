# Code taken from https://github.com/facebookresearch/segment-anything
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch

from collections import namedtuple
from torchvision.models import resnet50
from transformers import AutoImageProcessor

from .image_encoder import ImageEncoderViT
from .build_encoder import ENCODERS, build_vit_b, build_vit_h, build_vit_l
from .dcama import build_dcama
from .dummy import build_dummy
from .dmtnet import build_dmtnet

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
    "dummy": build_dummy,
    "deit": build_deit,
    "dmtnet": build_dmtnet,
    "resnet50": build_resnet50,
    # Encoders only
    **ENCODERS
}


def build_model(params):
    name = params["name"]
    params = {k: v for k, v in params.items() if k != "name"}
    return MODEL_REGISTRY[name](**params)
