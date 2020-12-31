# coding=utf-8
# @Time   : 2020/10/24 12:13
# @Auto   : zzf-jeff


'''
1.第一个卷积7x7相对h=32太大，改为多个3x3代替
2.W下采样为1/4, stride=(2, 1)

'''
import torch
import torch.nn as nn

from ..builder import BACKBONES
from .base import BaseBackbone



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
# 经过处理后的x要与x的维度相同(尺寸和深度)
# 如果不相同，需要添加卷积+BN来变换为同一维度
class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShortCut, self).__init__()
        if stride != 1 or in_channels != out_channels:
            self.conv = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=0, is_relu=False)
        else:
            self.conv = None

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        return x


# 用于18，34的block结果，使用2个3*3卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                                padding=1, is_relu=True)
        self.conv2 = ConvBnRelu(in_channels=out_channels, out_channels=out_channels * BasicBlock.expansion,
                                kernel_size=3, padding=1, stride=1, is_relu=False)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels * BasicBlock.expansion,
                                 stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = y + self.shortcut(x)
        return self.relu(y)


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckBlock, self).__init__()
        self.conv1 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                padding=0, is_relu=True)
        self.conv2 = ConvBnRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                                padding=1, is_relu=True)
        self.conv3 = ConvBnRelu(in_channels=out_channels, out_channels=out_channels * BottleneckBlock.expansion,
                                kernel_size=1, stride=1, padding=0, is_relu=False)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels * BottleneckBlock.expansion,
                                 stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = y + self.shortcut(x)
        return self.relu(y)


@BACKBONES.register_module()
class RecResNet(BaseBackbone):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (BottleneckBlock, (3, 4, 6, 3)),
        101: (BottleneckBlock, (3, 4, 23, 3)),
        152: (BottleneckBlock, (3, 8, 36, 3))
    }

    def __init__(self, depth, in_channels, num_classes=1000):
        super(RecResNet, self).__init__()
        self.in_channels = 64
        self.block = RecResNet.arch_settings[depth][0]
        self.num_block = RecResNet.arch_settings[depth][1]

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
        # self.out = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        '''
        堆叠block网络结构
        '''
        # 除了第一层的block块结构,stride为1,其它都为2,构建strides按照block个数搭建网络
        # [1,2]
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for _stride in strides:
            layers.append(block(self.in_channels, out_channels, _stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        # out = self.out(out)
        return out
