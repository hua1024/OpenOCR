# coding=utf-8  
# @Time   : 2020/12/1 14:54
# @Auto   : zzf-jeff

from torchocr.utils.registry import (build_from_cfg, Registry)
import torch
import torch.nn as nn

LR_SCHEDULER = Registry('lr_scheduler')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_lr_scheduler(cfg):
    return build(cfg, LR_SCHEDULER)