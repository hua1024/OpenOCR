# coding=utf-8  
# @Time   : 2020/12/29 11:57
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
from ..builder import HEADS


class Reshape(nn.Module):
    def __init__(self, in_channels, n_class, **kwargs):
        super(Reshape, self).__init__()
        self.embedding = nn.Linear(in_channels, n_class)
        self.n_class = n_class

    def forward(self, x):
        T, b, h = x.size()
        t_rec = x.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class DecodeWithAttn(nn.Module):
    def __init__(self,hidden_channel,n_class,dropout_p=0.1, max_length=71):
        super().__init__()
        pass


@HEADS.register_module()
class AttnHead(nn.Module):
    def __init__(self, in_channels, n_class, hidden_channel=256, dropout_p=0.1, max_length=71, **kwargs):
        super(AttnHead, self).__init__()
        self.reshape = Reshape(in_channels,n_class)
        self.hidden_channel = hidden_channel


    def forward(self, x):
        pass
