# coding=utf-8  
# @Time   : 2020/12/1 14:54
# @Auto   : zzf-jeff


import torch
import torch.nn as nn
import os
from collections import defaultdict
from torch.utils.data import DataLoader
from torchocr.utils.registry import (build_from_cfg, Registry)

DATASET = Registry('dataset')
PIPELINES = Registry('pipeline')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_dataloader(dataset, data, rank=-1,shuffle=True, **kwargs):
    """Build PyTorch DataLoader. 当前不考虑dist的情况

    :param dataset:
    :param data:
    :param shuffle:
    :param kwargs:
    :return:
    """
    batch_size = data.batch_size
    num_workers = min([os.cpu_count() // data.workers_per_gpu, batch_size if batch_size > 1 else 0, 8])
    num_workers = data.num_workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        # collate_fn=collate,
        sampler=sampler,
        pin_memory=False,
        drop_last=True,
        **kwargs)
    return data_loader


## 合并同batch间不一样的shape
# todo:还是固定在不同的dataloader里面
def collate(batch):
    if len(batch) == 0:
        return None
    clt = defaultdict(list)
    for i, dic in enumerate(batch):
        clt['idx'].append(torch.tensor(i))
        for k, v in dic.items():
            clt[k].append(v)
    for k, v in clt.items():
        if isinstance(clt[k][0], (torch.Tensor)):
            clt[k] = torch.stack(v, 0)
    return clt


def build_dataset(cfg):
    return build(cfg, DATASET)


