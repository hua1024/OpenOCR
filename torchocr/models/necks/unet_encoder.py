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
        self.out5_cls = conv1x1(in_channels[4], 2)
        self.out5_link = conv1x1(in_channels[4], 16)

    def forward(self, x):
        out_1, out_2, out_3, out_4, out_5 = x

        score_5 = self.out5_cls(out_5) + self.out4_cls(out_4)
        score_4 = self._upsample(score_5, self.out3_cls(out_3))
        score_3 = self._upsample(score_4, self.out2_cls(out_2))
        score_2 = self._upsample(score_3, self.out1_cls(out_1))

        link_5 = self.out5_link(out_5) + self.out4_link(out_4)
        link_4 = self._upsample(link_5, self.out3_link(out_3))
        link_3 = self._upsample(link_4, self.out2_link(out_2))
        link_2 = self._upsample(link_3, self.out1_link(out_1))

        link_map_list = (link_2, link_3, link_4, link_5)
        score_map_list = (score_2, score_3, score_4, score_5)
        return (link_map_list, score_map_list)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H // scale, W // scale), mode='nearest')
        return F.interpolate(x, size=(H // scale, W // scale), mode='nearest')
