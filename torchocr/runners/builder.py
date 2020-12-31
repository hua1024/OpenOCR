# coding=utf-8  
# @Time   : 2020/12/28 14:33
# @Auto   : zzf-jeff

from torchocr.utils.registry import Registry, build_from_cfg

RUNNERS = Registry('runner')


def build_runner(cfg, default_args=None):
    return build_from_cfg(cfg, RUNNERS, default_args=default_args)
