# coding=utf-8  
# @Time   : 2020/12/25 10:30
# @Auto   : zzf-jeff

import torch.nn as nn
import torch
from ..builder import HEADS


def conv1x1(in_channels, out_channels, stride=1, has_bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


@HEADS.register_module()
class PixelHead(nn.Module):
    def __init__(self, num_neighbours):
        super().__init__()
        self.link_out = num_neighbours * 2
        self.cls = conv1x1(2, 2)
        self.link = conv1x1(self.link_out, self.link_out)

    def forward(self, x):
        (score_map, link_map) = x
        out_link = self.link(link_map)
        out_cls = self.cls(score_map)
        if not self.training:
            outputs = torch.cat((out_cls, out_link), dim=1)
            return outputs
        pred = {'pred_link': out_link, 'pred_cls': out_cls}
        return pred
