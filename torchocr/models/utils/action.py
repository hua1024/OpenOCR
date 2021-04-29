# coding=utf-8  
# @Time   : 2021/3/12 14:37
# @Auto   : zzf-jeff

from torch import nn
import torch
import torch.nn.functional as F



class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class HSigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out
