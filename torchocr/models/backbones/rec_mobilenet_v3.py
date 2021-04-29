# coding=utf-8  
# @Time   : 2021/3/10 18:23
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..builder import BACKBONES
from .base import BaseBackbone
import torch.utils.model_zoo as model_zoo
from torchocr.utils.checkpoints import load_checkpoint
from torchocr.models.utils.conv import ConvBnLayer
from torchocr.models.utils.action import HSigmoid

from torch.nn import init

__all__ = [
    "RecMobileNetV3"
]


# make output channels is divisor 8,
# 具体原因说的是硬件size 可以被 d = 8, 16， ...
# 整除的矩阵乘法比较块，因为这些 size 符合处理器单元的对齐位宽。
def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        inner_channels = in_channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=inner_channels, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=inner_channels, out_channels=in_channels, kernel_size=1, bias=True)
        self.relu2 = HSigmoid()

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv1(attn)
        attn = self.relu1(attn)
        attn = self.conv2(attn)
        attn = self.relu2(attn)
        return x * attn


class ResidualUnit(nn.Module):
    def __init__(self, kernel_size, in_channels, inner_channels, out_channels, act, disable_se, stride):
        super(ResidualUnit, self).__init__()
        self.conv0 = ConvBnLayer(in_channels=in_channels, out_channels=inner_channels, kernel_size=1, stride=1,
                                 padding=0, act=act)

        self.conv1 = ConvBnLayer(in_channels=inner_channels, out_channels=inner_channels, kernel_size=kernel_size,
                                 stride=stride,
                                 padding=int((kernel_size - 1) // 2), act=act, groups=inner_channels)
        if not disable_se:
            self.se = SEModule(in_channels=inner_channels, reduction=4)
        else:
            self.se = None

        self.conv2 = ConvBnLayer(in_channels=inner_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                 padding=0)

        self.not_add = in_channels != out_channels or stride != 1

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        if self.se is not None:
            y = self.se(y)
        y = self.conv2(y)
        if not self.not_add:
            y = x + y
        return y


@BACKBONES.register_module()
class RecMobileNetV3(BaseBackbone):
    def __init__(self, mode, in_channels, scale=1, disable_se=False, num_classes=1000):
        super(RecMobileNetV3, self).__init__()
        self.disable_se = disable_se
        if mode == 'large':
            cfg = [
                # k, in, inner, out se, act,stride,
                [3, 16, 16, 16, False, 'relu', 1],
                [3, 24, 64, 24, False, 'relu', (2, 1)],
                [3, 24, 72, 24, False, 'relu', 1],
                [5, 40, 72, 40, True, 'relu', (2, 1)],
                [5, 40, 120, 40, True, 'relu', 1],
                [5, 40, 120, 40, True, 'relu', 1],
                [3, 80, 240, 80, False, 'hardswish', 1],
                [3, 80, 200, 80, False, 'hardswish', 1],
                [3, 80, 184, 80, False, 'hardswish', 1],
                [3, 80, 184, 80, False, 'hardswish', 1],
                [3, 112, 480, 112, True, 'hardswish', 1],
                [3, 112, 672, 112, True, 'hardswish', 1],
                [5, 160, 672, 160, True, 'hardswish', (2, 1)],
                [5, 160, 960, 160, True, 'hardswish', 1],
                [5, 160, 960, 160, True, 'hardswish', 1],
            ]
            # size_stride = [2, 2, 2, 2]
            cls_ch_squeeze = 960
        elif mode == 'small':
            cfg = [
                # k, in, inner,out, se, act,stride,
                [3, 16, 16, 16, True, 'relu', (2, 1)],
                [3, 24, 72, 24, False, 'relu', (2, 1)],
                [3, 24, 88, 24, False, 'relu', 1],
                [5, 40, 96, 40, True, 'hardswish', (2, 1)],
                [5, 40, 240, 40, True, 'hardswish', 1],
                [5, 40, 240, 40, True, 'hardswish', 1],
                [5, 48, 120, 48, True, 'hardswish', 1],
                [5, 48, 144, 48, True, 'hardswish', 1],
                [5, 96, 288, 96, True, 'hardswish', (2, 1)],
                [5, 96, 576, 96, True, 'hardswish', 1],
                [5, 96, 576, 96, True, 'hardswish', 1],
            ]
            # size_stride = [1, 2, 2, 2]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode : {} is not implemented!".format(mode))

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]

        assert scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, scale)
        in_planes = 16
        # conv1
        self.conv1 = ConvBnLayer(
            in_channels=in_channels,
            out_channels=make_divisible(in_planes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            is_act=True,
            act='hardswish'
        )
        block_list = []
        i = 0
        in_planes = make_divisible(in_planes * scale)

        for idx, (k, input, inner, out, se, act, stride) in enumerate(cfg):
            # se layer
            se = se and not self.disable_se
            block_list.append(
                ResidualUnit(
                    kernel_size=k,
                    in_channels=in_planes,
                    inner_channels=make_divisible(scale * inner),
                    out_channels=make_divisible(scale * out),
                    stride=stride,
                    disable_se=se,
                    act=act
                )
            )

            in_planes = make_divisible(scale * out)
            i += 1
        self.blocks = nn.Sequential(*block_list)

        self.conv2 = ConvBnLayer(
            in_channels=in_planes,
            out_channels=make_divisible(cls_ch_squeeze * scale),
            kernel_size=1,
            stride=1,
            is_act=True,
            act='hardswish'
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = make_divisible(scale * cls_ch_squeeze)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x