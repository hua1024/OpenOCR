# coding=utf-8  
# @Time   : 2021/1/8 16:37
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
from ..builder import HEADS
import torch.nn.functional as F
import math


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


@HEADS.register_module()
class EASTHead(nn.Module):
    def __init__(self, model_name, img_size=512, **kwargs):
        super().__init__()
        # geometry is RBOX
        if model_name == "large":
            num_outputs = [128, 64, 1, 4, 1]
        else:
            num_outputs = [64, 32, 1, 4, 1]

        self.img_size = img_size

        self.conv1 = ConvBnRelu(
            in_channels=num_outputs[0],
            out_channels=num_outputs[0],
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = ConvBnRelu(
            in_channels=num_outputs[0],
            out_channels=num_outputs[1],
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.score_conv = ConvBnRelu(
            in_channels=num_outputs[1],
            out_channels=num_outputs[2],
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.loc_conv = ConvBnRelu(
            in_channels=num_outputs[1],
            out_channels=num_outputs[3],
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.angle_conv = ConvBnRelu(
            in_channels=num_outputs[1],
            out_channels=num_outputs[4],
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        f_score = torch.sigmoid(self.score_conv(x))
        loc = torch.sigmoid(self.loc_conv(x)) * self.img_size
        # angle对应-90°到90°
        angle = (torch.sigmoid(self.angle_conv(x)) - 0.5) * math.pi
        f_geo = torch.cat((loc, angle), 1)
        pred = {'pred_score': f_score, 'pred_geo': f_geo}
        return pred
