# coding=utf-8  
# @Time   : 2020/12/2 9:40
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
from ..builder import HEADS
import torch.nn.functional as F


@HEADS.register_module()
class CTCHead(nn.Module):
    def __init__(self, in_channels, n_class, **kwargs):
        super(CTCHead, self).__init__()
        self.embedding = nn.Linear(in_channels, n_class)
        self.n_class = n_class

    def forward(self, x):
        output = self.embedding(x)
        if not self.training:
            output = F.softmax(output, dim=2)
        return output

    # def forward(self, x):
    #     T, b, c = x.size()
    #     t_rec = x.view(T * b, c)
    #     output = self.embedding(t_rec)
    #     output = output.view(T, b, -1) # [T,B,C]
    #     output = output.permute(1, 0, 2)
    #     if not self.training:
    #         output = F.softmax(output, dim=2)
    #     return output