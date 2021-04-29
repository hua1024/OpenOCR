# coding=utf-8  
# @Time   : 2020/11/27 10:07
# @Auto   : zzf-jeff

'''
    处理了stand和fusion-last的写法,没想到如何兼容fusion-first
         transitionBlock=True,
         transitionDense=True, 为 stand

         transitionBlock=False,
         transitionDense=True, 为 fusion-last
'''
import torch
import torch.nn as nn
from ..builder import BACKBONES
import torch.nn.functional as F

from .base import BaseBackbone
from collections import OrderedDict
import torch.utils.checkpoint as cp

__all__ = [
    "RecCSPDenseNet"
]


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _CSPTransition(torch.nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_CSPTransition, self).__init__()
        self.add_module('norm', torch.nn.BatchNorm2d(num_input_features))
        self.add_module('relu', torch.nn.ReLU(inplace=True))
        self.add_module('conv', torch.nn.Conv2d(num_input_features, num_output_features,
                                                kernel_size=1, stride=1, bias=False))


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        # # 调整h的channel为1，尽量保证t=1/4w,只对h方向进行pool
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1)))


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _CSPDenseBlock(torch.nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False,
                 transition=False):
        super(_CSPDenseBlock, self).__init__()
        # 两个part,part 比例为0.5 -->part1,part2
        self.csp_num_features1 = num_input_features // 2
        self.csp_num_features2 = num_input_features - self.csp_num_features1

        # csp dense block 的transition in channels
        trans_in_features = num_layers * growth_rate + self.csp_num_features2

        for i in range(num_layers):
            # part2 -->densenet
            layer = _DenseLayer(
                self.csp_num_features2 + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

        # 这里默认reduction=0.5
        self.transition = _CSPTransition(trans_in_features, trans_in_features) if transition else None

    def forward(self, x):

        features = [x[:, self.csp_num_features1:, ...]]
        for name, layer in self.named_children():
            if 'denselayer' in name:
                new_feature = layer(*features)
                features.append(new_feature)
        dense = torch.cat(features, 1)

        if self.transition is not None:
            dense = self.transition(dense)

        # part1 is shortcut
        # part2 = dense net + csp transition(without maxpool)
        return torch.cat([x[:, :self.csp_num_features1, ...], dense], 1)


@BACKBONES.register_module()
class RecCSPDenseNet(BaseBackbone):
    arch_settings = {
        121: (32, [6, 12, 24, 16]),
        161: (48, [6, 12, 36, 24]),
        169: (32, [6, 12, 32, 32]),
        201: (32, [6, 12, 48, 32]),
        264: (32, [6, 12, 64, 48])
    }

    def __init__(self,
                 depth,
                 in_channels,
                 num_classes=1000,
                 reduction=0.5,
                 transitionBlock=True,
                 transitionDense=True,
                 bn_size=4,
                 drop_rate=0,
                 memory_efficient=False):

        super(RecCSPDenseNet, self).__init__()
        (self.growth_rate, self.num_block) = self.arch_settings[depth]

        num_init_features = 2 * self.growth_rate

        # 7*7替换成多个3*3的
        self.features = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2,
                                      padding=3, bias=False)),
            ('norm0', torch.nn.BatchNorm2d(num_init_features)),
            ('relu0', torch.nn.ReLU(inplace=True)),
            ('pool0', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features

        for i, num_layers in enumerate(self.num_block):
            block = _CSPDenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=self.growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                transition=transitionBlock
            )
            self.features.add_module('csp_denseblock%d' % (i + 1), block)

            # ** 因为denseblock 是bc结构，reduction=0.5 里面的transition做了一次//2的操作
            # 外面的也同步channels
            num_features = self.growth_rate * num_layers + num_features

            if (i != len(self.num_block) - 1) and transitionDense:
                num_output_features = int(reduction * num_features)
                trans = _Transition(num_input_features=num_features, num_output_features=num_output_features)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_output_features

        self.features.add_module('norm5', torch.nn.BatchNorm2d(num_features))

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)

        return features
