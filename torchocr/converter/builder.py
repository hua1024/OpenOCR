# coding=utf-8  
# @Time   : 2020/12/1 14:54
# @Auto   : zzf-jeff

from ..utils.registry import (Registry, build_from_cfg)

CONVERTER = Registry('converter')


def build_converter(cfg, default_args=None):
    converter = build_from_cfg(cfg, CONVERTER, default_args=default_args)
    return converter
