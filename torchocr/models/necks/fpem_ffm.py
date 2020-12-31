# coding=utf-8  
# @Time   : 2020/12/23 11:24
# @Auto   : zzf-jeff


from torch import nn
import torch
import torch.nn.functional as F
from ..builder import NECKS


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


class SeparableConv2d(nn.Module):
    '''可分离卷积

    '''

    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FPEM(nn.Module):
    def __init__(self, in_channels):
        super(FPEM, self).__init__()
        self.up_add1 = SeparableConv2d(in_channels, in_channels, stride=1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, stride=1)
        self.up_add3 = SeparableConv2d(in_channels, in_channels, stride=1)
        self.down_add1 = SeparableConv2d(in_channels, in_channels, stride=2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, stride=2)
        self.down_add3 = SeparableConv2d(in_channels, in_channels, stride=2)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H, W), mode='nearest') + y
        return F.interpolate(x, size=(H, W), mode='nearest') + y

    def forward(self, c2, c3, c4, c5):
        # up add
        c4 = self.up_add1(self._upsample_add(c5, c4))
        c3 = self.up_add2(self._upsample_add(c4, c3))
        c2 = self.up_add3(self._upsample_add(c3, c2))
        # dowm add
        c3 = self.down_add1(self._upsample_add(c3, c2))
        c4 = self.down_add2(self._upsample_add(c4, c3))
        c5 = self.down_add3(self._upsample_add(c5, c4))

        return c2, c3, c4, c5


@NECKS.register_module()
class FPEM_FFM(nn.Module):
    def __init__(self, in_channels, out_channels=128, num_fpem=2):
        super(FPEM_FFM, self).__init__()
        self.in5 = ConvBnRelu(in_channels[-1], out_channels, kernel_size=1, stride=1, padding=0)
        self.in4 = ConvBnRelu(in_channels[-2], out_channels, kernel_size=1, stride=1, padding=0)
        self.in3 = ConvBnRelu(in_channels[-3], out_channels, kernel_size=1, stride=1, padding=0)
        self.in2 = ConvBnRelu(in_channels[-4], out_channels, kernel_size=1, stride=1, padding=0)
        self.fpems = nn.ModuleList()
        for i in range(num_fpem):
            self.fpems.append(FPEM(out_channels))

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)
        # c2_ffm, c3_ffm, c4_ffm, c5_ffm = '', '', '', ''
        # FPEM
        for idx, fpem in enumerate(self.fpems):
            c2, c3, c4, c5 = fpem(in2, in3, in4, in5)
            if idx == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm += c2
                c3_ffm += c3
                c4_ffm += c4
                c5_ffm += c5

        # FFM
        c5 = F.interpolate(c5_ffm, size=c2_ffm.size()[2:], mode='nearest')
        c4 = F.interpolate(c4_ffm, size=c2_ffm.size()[2:], mode='nearest')
        c3 = F.interpolate(c3_ffm, size=c2_ffm.size()[2:], mode='nearest')
        y = torch.cat([c2_ffm, c3, c4, c5], dim=1)
        return y
