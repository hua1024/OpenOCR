# coding=utf-8  
# @Time   : 2020/12/25 10:30
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
from ..builder import HEADS


def conv1x1(in_channels, out_channels, stride=1, has_bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


@HEADS.register_module()
class PixelHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = conv1x1(2, 2)
        self.link = conv1x1(16, 16)

    def forward(self, x):
        (score_map, link_map) = x
        out_link = self.link(link_map)
        out_cls = self.cls(score_map)
        print(out_cls.shape)
        print(out_link.shape)
        pred = {'pred_link': out_link, 'pred_score': out_cls}
        return pred
