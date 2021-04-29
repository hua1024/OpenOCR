# coding=utf-8  
# @Time   : 2020/12/23 18:37
# @Auto   : zzf-jeff


import torch
import numpy as np
import torch.nn as nn
import math
from torchocr.utils.checkpoints import load_checkpoint
from ..builder import BACKBONES
from .base import BaseBackbone


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            padding_mode='zeros',
            is_deploy=False):
        super(RepVGGBlock, self).__init__()
        self.is_deploy = is_deploy
        self.groups = groups
        self.in_channels = in_channels

        # set k=3x3 and pad=1
        assert kernel_size == 3
        assert padding == 1

        self.norm = nn.ReLU()
        if is_deploy:
            # if deploy , bn and conv is merge, so need bias=True
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            # k=1x1，not pad can set shape equal
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=0, groups=groups)

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.norm(self.rbr_reparam(x))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)

        return self.norm(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

    # repvgg block transform deploy

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        # conv和bn 融合的公式
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel, bias


@BACKBONES.register_module()
class DetRepVGG(BaseBackbone):
    # paper config
    optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    g2_map = {l: 2 for l in optional_groupwise_layers}
    g4_map = {l: 4 for l in optional_groupwise_layers}
    arch_settings = {
        'A0': (RepVGGBlock, (2, 4, 14, 1), (0.75, 0.75, 0.75, 2.5), None),
        'A1': (RepVGGBlock, (2, 4, 14, 1), (1, 1, 1, 2.5), None),
        'A2': (RepVGGBlock, (2, 4, 14, 1), (1.5, 1.5, 1.5, 2.75), None),
        'B0': (RepVGGBlock, (4, 6, 16, 1), (1, 1, 1, 2.5), None),
        'B1': (RepVGGBlock, (4, 6, 16, 1), (2, 2, 2, 4), None),
        'B1g2': (RepVGGBlock, (4, 6, 16, 1), (2, 2, 2, 4), g2_map),
        'B1g4': (RepVGGBlock, (4, 6, 16, 1), (2, 2, 2, 4), g4_map),
        'B2': (RepVGGBlock, (4, 6, 16, 1), (2.5, 2.5, 2.5, 5), None),
        'B2g2': (RepVGGBlock, (4, 6, 16, 1), (2.5, 2.5, 2.5, 5), g2_map),
        'B2g4': (RepVGGBlock, (4, 6, 16, 1), (2.5, 2.5, 2.5, 5), g4_map),
        'B3': (RepVGGBlock, (4, 6, 16, 1), (3, 3, 3, 5), None),
        'B3g2': (RepVGGBlock, (4, 6, 16, 1), (3, 3, 3, 5), g2_map),
        'B3g4': (RepVGGBlock, (4, 6, 16, 1), (3, 3, 3, 5), g4_map),
    }

    def __init__(self, depth, in_channels, is_deploy, num_classes=1000):
        super(DetRepVGG, self).__init__()
        # config setting
        self.block = self.arch_settings[depth][0]
        self.num_blocks = self.arch_settings[depth][1]
        self.width_multiplier = self.arch_settings[depth][2]
        self.override_groups_map = self.arch_settings[depth][3]
        if self.override_groups_map is None:
            self.override_groups_map = dict()

        self.is_deploy = is_deploy

        # network
        self.in_planes = min(64, int(64 * self.width_multiplier[0]))
        self.stage0 = RepVGGBlock(in_channels=in_channels, out_channels=self.in_planes, kernel_size=3, stride=2,
                                  padding=1, is_deploy=self.is_deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * self.width_multiplier[0]), self.num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * self.width_multiplier[1]), self.num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * self.width_multiplier[2]), self.num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * self.width_multiplier[3]), self.num_blocks[3], stride=2)
        # self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            # get network group number
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)

            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, is_deploy=self.is_deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1

        return nn.Sequential(*blocks)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif isinstance(pretrained, str):
            load_checkpoint(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.stage0(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        return (c2, c3, c4, c5)
