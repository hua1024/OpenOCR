# coding=utf-8  
# @Time   : 2021/1/8 14:57
# @Auto   : zzf-jeff


import torch
import torch.nn as nn
from ..builder import HEADS
import torch.nn.functional as F


@HEADS.register_module()
class C2TDHead(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(C2TDHead, self).__init__()
        self.score_head = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        heat_map = self.score_head(x)
        if not self.training:
            pass
        return heat_map
