# coding=utf-8
# @Time   : 2020/10/24 12:13
# @Auto   : zzf-jeff
'''
1.resnet 的d变形
2.新增resnet200
resnet18 : [2, 2, 2, 2]
resnet34 : [3, 4, 6, 3]
resnet50 : [3, 4, 6, 3]
resnet101 : [3, 4, 23, 3]
resnet152 : [3, 8, 36, 3]
resnet200 : [3, 12, 48, 3]

stride=2，放到第二个1x1卷积
替换第一层7*7-->3个3*3，这个同步到resnet_vd重，resnet取消使用，感觉对于28*28的输入图片，第一个卷积7*7会太大
shortcut层采用avgpool+conv1x1 代替conv1x1
'''

import torch
import torch.nn as nn
import math
from ..builder import BACKBONES
from .base import BaseBackbone

__all__ = [
    "RecResNetVd"
]


# conv+bn+relu
class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, is_relu=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if is_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# 跨层连接
# 使用avgpool(2*2,s=2)+conv(1*1)
# 减少信息丢失？
class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, is_first=False):
        super(ShortCut, self).__init__()
        if stride != 1 or in_channels != out_channels:
            if is_first:
                self.conv = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                       padding=0, stride=1, is_relu=False)
            else:

                self.conv = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
                    ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                               padding=0, is_relu=False)
                )
        elif is_first:
            self.conv = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=0, is_relu=False)
        else:
            self.conv = None

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        return x


# 用于18，34的block结果，使用2个3*3卷积
# 除了改变shortcut外,BasicBlock没有没有修改
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, is_first):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                                padding=1, is_relu=True)
        self.conv2 = ConvBnRelu(in_channels=out_channels, out_channels=out_channels * self.expansion,
                                kernel_size=3, padding=1, stride=1, is_relu=False)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels * self.expansion,
                                 stride=stride, is_first=is_first)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = y + self.shortcut(x)
        return self.relu(y)


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
# 将下采样从第一个卷积移动到第三个
# shortcut修改
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, is_first):
        super(BottleneckBlock, self).__init__()
        self.conv1 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                padding=0, is_relu=True)
        self.conv2 = ConvBnRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                                padding=1, is_relu=True)
        self.conv3 = ConvBnRelu(in_channels=out_channels, out_channels=out_channels * self.expansion,
                                kernel_size=1, stride=1, padding=0, is_relu=False)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels * self.expansion,
                                 stride=stride, is_first=is_first)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = y + self.shortcut(x)
        return self.relu(y)


@BACKBONES.register_module()
class RecResNetVd(BaseBackbone):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (BottleneckBlock, (3, 4, 6, 3)),
        101: (BottleneckBlock, (3, 4, 23, 3)),
        152: (BottleneckBlock, (3, 8, 36, 3)),
        200: (BottleneckBlock, (3, 12, 48, 3))
    }

    def __init__(self, depth, in_channels, num_classes=1000):
        super(RecResNetVd, self).__init__()
        self.in_channels = 64

        self.block = self.arch_settings[depth][0]
        self.num_block = self.arch_settings[depth][1]

        self.conv1 = nn.Sequential(
            ConvBnRelu(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1,
                       is_relu=True),
            ConvBnRelu(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
                       is_relu=True),
            ConvBnRelu(in_channels=32, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1,
                       is_relu=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(block=self.block, out_channels=64, num_blocks=self.num_block[0], stride=1)
        self.conv3_x = self._make_layer(block=self.block, out_channels=128, num_blocks=self.num_block[1], stride=(2, 1))
        self.conv4_x = self._make_layer(block=self.block, out_channels=256, num_blocks=self.num_block[2], stride=(2, 1))
        self.conv5_x = self._make_layer(block=self.block, out_channels=512, num_blocks=self.num_block[3], stride=(2, 1))
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * self.block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, is_first=False):
        '''
        堆叠block网络结构
        '''
        # 除了第一层的block块结构,stride为1,其它都为2,构建strides按照block个数搭建网络
        # [1,2]
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for idx, _stride in enumerate(strides):
            # 除了第一层外，后续shortcut都做修改
            layers.append(block(self.in_channels, out_channels, _stride, is_first=stride == 1 and idx == 0))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        return x
