# coding=utf-8  
# @Time   : 2020/12/1 14:55
# @Auto   : zzf-jeff

from torchocr.utils.registry import (build_from_cfg, Registry)
import torch
import torch.nn as nn

OPTIMIZER = Registry('optimizer')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_optimizer(cfg, model):
    return build(cfg, OPTIMIZER)(model)
