# coding=utf-8  
# @Time   : 2020/12/1 14:53
# @Auto   : zzf-jeff

import torch.nn as nn
from ..utils.registry import (Registry, build_from_cfg)

LOSS = Registry('loss')
BACKBONE = Registry('backbone')
HEAD = Registry('head')
DETECTION = Registry('detection')
RECOGNITION = Registry('recognition')
NECK = Registry('neck')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_loss(cfg):
    return build(cfg, LOSS)


def build_backbone(cfg):
    return build(cfg, BACKBONE)


def build_head(cfg):
    return build(cfg, HEAD)


def build_neck(cfg):
    return build(cfg, NECK)


def build_det(cfg):
    return build(cfg, DETECTION)


def build_rec(cfg):
    return build(cfg, RECOGNITION)
