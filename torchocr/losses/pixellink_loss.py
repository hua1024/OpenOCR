# coding=utf-8  
# @Time   : 2021/1/11 17:53
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
import torch.nn.functional as F
from .det_basic_loss import DiceLoss
from .builder import LOSSES


@LOSSES.register_module()
class PixelLinkLoss(nn.Module):
    pass