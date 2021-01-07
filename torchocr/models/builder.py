# coding=utf-8  
# @Time   : 2020/12/1 14:53
# @Auto   : zzf-jeff

import torch.nn as nn
from ..utils.registry import (Registry, build_from_cfg)

TRANSFORMS = Registry('transform')
BACKBONES = Registry('backbone')
HEADS = Registry('head')
NECKS = Registry('neck')
MODELS = Registry('model')


# TODO : bulid是加入support的key说明

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_head(cfg):
    return build(cfg, HEADS)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_model(cfg):
    return build(cfg, MODELS)

def build_transform(cfg):
    return build(cfg, TRANSFORMS)



