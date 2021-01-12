# coding=utf-8  
# @Time   : 2020/12/22 14:48
# @Auto   : zzf-jeff

'''
DB_FPN 输出 channels = 256
PSE_FPN 输出 channels = 256*4
'''

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


@NECKS.register_module()
class DB_FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256, **kwargs):
        super(DB_FPN, self).__init__()
        inner_channels = out_channels // 4
        # inx 为将输入的channels 转为256
        self.in5 = ConvBnRelu(in_channels[-1], out_channels, kernel_size=1, stride=1, padding=0)
        self.in4 = ConvBnRelu(in_channels[-2], out_channels, kernel_size=1, stride=1, padding=0)
        self.in3 = ConvBnRelu(in_channels[-3], out_channels, kernel_size=1, stride=1, padding=0)
        self.in2 = ConvBnRelu(in_channels[-4], out_channels, kernel_size=1, stride=1, padding=0)

        # out 为将输入的channels 转为256//4方便后面的cat，在通用目标检测中用来做px的smooth
        self.out5 = ConvBnRelu(out_channels, inner_channels, kernel_size=3, stride=1, padding=1)
        self.out4 = ConvBnRelu(out_channels, inner_channels, kernel_size=3, stride=1, padding=1)
        self.out3 = ConvBnRelu(out_channels, inner_channels, kernel_size=3, stride=1, padding=1)
        self.out2 = ConvBnRelu(out_channels, inner_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self._upsample_add(in5, in4)  # 1/16
        out3 = self._upsample_add(out4, in3)  # 1/8
        out2 = self._upsample_add(out3, in2)  # 1/4

        p5 = self._upsample(self.out5(in5), out2)
        p4 = self._upsample(self.out4(out4), out2)
        p3 = self._upsample(self.out3(out3), out2)
        p2 = self.out2(out2)
        # 对fuse有没有必要做多一层conv，smooth
        fuse = torch.cat((p5, p4, p3, p2), 1)
        return fuse

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H // scale, W // scale), mode='nearest')
        return F.interpolate(x, size=(H // scale, W // scale), mode='nearest')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H, W), mode='nearest') + y
        return F.interpolate(x, size=(H, W), mode='nearest') + y

    # def weights_init(self, m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         nn.init.kaiming_normal_(m.weight.data)
    #     elif classname.find('BatchNorm') != -1:
    #         m.weight.data.fill_(1.)
    #         m.bias.data.fill_(1e-4)


@NECKS.register_module()
class PSE_FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256, **kwargs):
        super(PSE_FPN, self).__init__()
        # inner_channels = out_channels // 4
        # inx 为将输入的channels 转为256
        self.in5 = ConvBnRelu(in_channels[-1], out_channels, kernel_size=1, stride=1, padding=0)
        self.in4 = ConvBnRelu(in_channels[-2], out_channels, kernel_size=1, stride=1, padding=0)
        self.in3 = ConvBnRelu(in_channels[-3], out_channels, kernel_size=1, stride=1, padding=0)
        self.in2 = ConvBnRelu(in_channels[-4], out_channels, kernel_size=1, stride=1, padding=0)

        #
        self.out5 = ConvBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.out4 = ConvBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.out3 = ConvBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.out2 = ConvBnRelu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.out_conv = ConvBnRelu(out_channels*4, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        c2, c3, c4, c5 = x
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self._upsample_add(in5, in4)  # 1/16
        out3 = self._upsample_add(out4, in3)  # 1/8
        out2 = self._upsample_add(out3, in2)  # 1/4

        p5 = self._upsample(self.out5(in5), out2)
        p4 = self._upsample(self.out4(out4), out2)
        p3 = self._upsample(self.out3(out3), out2)
        p2 = self.out2(out2)
        # 对fuse有没有必要做多一层conv，smooth
        fuse = torch.cat((p5, p4, p3, p2), 1)

        fuse = self.out_conv(fuse)

        return fuse

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H // scale, W // scale), mode='nearest')
        return F.interpolate(x, size=(H // scale, W // scale), mode='nearest')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H, W), mode='nearest') + y
        return F.interpolate(x, size=(H, W), mode='nearest') + y

