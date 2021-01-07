# coding=utf-8  
# @Time   : 2020/12/4 14:50
# @Auto   : zzf-jeff


from torchocr.utils.registry import (build_from_cfg, Registry)
import torch
import torch.nn as nn

METRICS = Registry('metrics')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_metrics(cfg):
    return build(cfg, METRICS)