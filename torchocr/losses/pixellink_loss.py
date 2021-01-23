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
    def __init__(self):
        super().__init__()
        pass

    def forward(self, pred, batch):
        pred_link = pred['pred_link']
        pred_cls = pred['pred_cls']

        pixel_cls_label = batch['pixel_cls_label']
        pixel_cls_weight = batch['pixel_cls_weight']
        pixel_link_label = batch['pixel_link_label']
        pixel_link_weight = batch['pixel_link_weight']

        pos_mask = (pixel_cls_label > 0)
        neg_mask = (pixel_cls_label == 0)
