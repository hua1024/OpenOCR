# coding=utf-8  
# @Time   : 2020/12/22 16:42
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
from ..builder import HEADS


@HEADS.register_module()
class DBHead(nn.Module):
    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k
        inner_channels = in_channels // 4
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels, inner_channels, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels, 1, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid()
        )
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels, inner_channels, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels, 1, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid()
        )

    # thresh map 如果用nn.Upsample感觉效果会好一点
    def _init_upsample(self, in_channels, out_channels, smooth=False):
        pass

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    # db公式 B= 1/1+e -k(P-T)
    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):

        probability_map = self.binarize(x)
        # 推理只采用probability_map
        if not self.training:
            return probability_map.detach().cpu().numpy()

        threshold_maps = self.thresh(x)
        binary_map = self.step_function(probability_map, threshold_maps)
        y = torch.cat((probability_map, threshold_maps, binary_map), dim=1)
        return y



