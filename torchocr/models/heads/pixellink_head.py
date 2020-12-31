# coding=utf-8  
# @Time   : 2020/12/25 10:30
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
from ..builder import HEADS


@HEADS.register_module()
class PixelHead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        link_map_list, score_map_list = x
        if self.mode == '2s':
            link_map = link_map_list[0]
            score_map = score_map_list[0]
        elif self.mode == '4s':
            link_map = link_map_list[1]
            score_map = score_map_list[1]
        else:
            print('Only support 2s or 4s')
            raise

        return (link_map, score_map)
