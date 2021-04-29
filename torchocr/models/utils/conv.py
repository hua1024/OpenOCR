# coding=utf-8  
# @Time   : 2021/1/22 15:27
# @Auto   : zzf-jeff
from torch import nn
import torch
import torch.nn.functional as F
from .action import HSwish, HSigmoid


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', inplace=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBnLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 is_act=True,
                 act='relu',
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros',
                 inplace=True):
        super(ConvBnLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)

        self.bn = nn.BatchNorm2d(out_channels)
        self.is_act = is_act
        self.act = act
        if self.is_act:
            if self.act == "relu":
                self.action = nn.ReLU(inplace=inplace)
            elif self.act == "hardswish":
                self.action = HSwish()
            else:
                raise NotImplementedError("action : {} is not implemented!".format(self.act))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.is_act:
            x = self.action(x)
        return x
