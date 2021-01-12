# coding=utf-8  
# @Time   : 2020/12/25 9:35
# @Auto   : zzf-jeff


from torch import nn
import torch
import torch.nn.functional as F
from ..builder import NECKS


def conv1x1(in_channels, out_channels, stride=1, has_bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


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


class DeConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', inplace=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@NECKS.register_module()
class PixelWithUnet(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out1_cls = conv1x1(in_channels[0], 2)
        self.out1_link = conv1x1(in_channels[0], 16)
        self.out2_cls = conv1x1(in_channels[1], 2)
        self.out2_link = conv1x1(in_channels[1], 16)
        self.out3_cls = conv1x1(in_channels[2], 2)
        self.out3_link = conv1x1(in_channels[2], 16)
        self.out4_cls = conv1x1(in_channels[3], 2)
        self.out4_link = conv1x1(in_channels[3], 16)

    def forward(self, x):
        c1, c2, c3, c4 = x
        #
        out_cls = self.out4_cls(c4)
        out_link = self.out4_link(c4)
        #
        out_cls = self._upsample(out_cls, self.out3_cls(c3))
        out_link = self._upsample(out_link, self.out3_link(c3))
        out_cls = self._upsample(out_cls, self.out2_cls(c2))
        out_link = self._upsample(out_link, self.out2_link(c2))
        out_cls = self._upsample(out_cls, self.out1_cls(c1))
        out_link = self._upsample(out_link, self.out1_link(c1))

        return (out_cls, out_link)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H // scale, W // scale), mode='nearest')
        return F.interpolate(x, size=(H // scale, W // scale), mode='nearest')





@NECKS.register_module()
class C2TDWithUnet(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        类似于pixel 和 east的unet结构

        :param in_channels:[256, 512, 512, 1024]
        :param kwargs:
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.score_map = nn.ModuleList(self._make_layer(in_channels, out_channels))

    def _make_layer(self, in_channels, out_channels):
        """
        :param in_channel:
        :param out_channel:
        :return:
        """
        head = []
        for i in in_channels:
            head += [nn.Conv2d(i, out_channels, kernel_size=1)]
        return head

    def forward(self, x):

        score_maps = []
        for _x, score_map in zip(x, self.score_map):
            score_maps.append(score_map(_x))
        # todo:有点奇怪的add？写法也待优化
        # score_maps[2].shape =  score_maps[3].shape
        feature = score_maps[2] + score_maps[3]
        for i in [1, 0]:
            feature = F.interpolate(feature, None, 2, mode='nearest') + score_maps[i]

        return feature


# todo interpolate 好还是 DeConv,同時unet中concat好像好过add
@NECKS.register_module()
class EASTWithUnet(nn.Module):
    def __init__(self, in_channels, model_name, **kwargs):
        super().__init__()
        in_channels = in_channels[::-1]
        if model_name == "large":
            out_channels = 128
        else:
            out_channels = 64
        self.h1_conv = ConvBnRelu(
            in_channels=out_channels + in_channels[1],
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.h2_conv = ConvBnRelu(
            in_channels=out_channels + in_channels[2],
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.h3_conv = ConvBnRelu(
            in_channels=out_channels + in_channels[3],
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.g0_deconv = DeConvBnRelu(
            in_channels=in_channels[0],
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.g1_deconv = DeConvBnRelu(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.g2_deconv = DeConvBnRelu(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.g3_deconv = DeConvBnRelu(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        f = x[::-1]
        h = f[0]
        g = self.g0_deconv(h)
        h = torch.cat([g, f[1]], dim=1)
        h = self.h1_conv(h)
        g = self.g1_deconv(h)
        h = torch.cat([g, f[2]], dim=1)
        h = self.h2_conv(h)
        g = self.g2_deconv(h)
        h = torch.cat([g, f[3]], dim=1)
        h = self.h3_conv(h)
        g = self.g3_deconv(h)

        return g
