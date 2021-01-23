# coding=utf-8  
# @Time   : 2020/12/22 18:22
# @Auto   : zzf-jeff


import torch
import torch.nn as nn
from ..builder import HEADS
import torch.nn.functional as F


@HEADS.register_module()
class PANHead(nn.Module):
    def __init__(self, in_channels, img_shape=(640, 640), scale=1, **kwargs):
        super(PANHead, self).__init__()

        self.out_conv = nn.Conv2d(in_channels=in_channels * 4, out_channels=6, kernel_size=1)

        self.img_shape = img_shape
        self.scale = scale

    def forward(self, x):
        x = self.out_conv(x)
        if not self.training:
            y = F.interpolate(x, size=(self.img_shape[0] // self.scale, self.img_shape[1] // self.scale),
                              mode='bilinear', align_corners=True)
        else:
            y = F.interpolate(x, size=self.img_shape, mode='bilinear', align_corners=True)
        return y
