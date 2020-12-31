# coding=utf-8  
# @Time   : 2020/12/1 14:53
# @Auto   : zzf-jeff

import torch.nn as nn
from ..utils.registry import (Registry, build_from_cfg)

LOSSES = Registry('loss')
BACKBONES = Registry('backbone')
HEADS = Registry('head')
DETECTIONS = Registry('detection')
RECOGNITIONS = Registry('recognition')
NECKS = Registry('neck')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_head(cfg):
    return build(cfg, HEADS)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_det(cfg):
    return build(cfg, DETECTIONS)


def build_rec(cfg):
    return build(cfg, RECOGNITIONS)
