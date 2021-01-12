# coding=utf-8  
# @Time   : 2020/12/29 22:31
# @Auto   : zzf-jeff


import torch
import torch.nn as nn
from .builder import LOSSES
from .det_basic_loss import (BalanceCrossEntropyLoss, DiceLoss, MaskL1Loss)


@LOSSES.register_module()
class DBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, reduction='mean', eps=1e-6):
        """DB loss

        :param alpha:
        :param beta:
        :param ohem_ratio:
        :param reduction:
        :param eps:
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio, eps=eps)
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)

    def forward(self, pred, batch):
        """

        :param pred:
        :param batch:
            bach为一个dict{
                'shrink_map': 收缩图,b*c*h,w
                'shrink_mask: 收缩图mask,b*c*h,w
                'threshold_map: 二值化边界gt,b*c*h,w
                'threshold_mask: 二值化边界gtmask,b*c*h,w
                }
        :return:
        """
        # 对应head中的结构
        # shrink_map 就是 probability map
        probability_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]
        binary_maps = pred[:, 2, :, :]
        loss_probability_maps = self.bce_loss(probability_maps, batch['shrink_map'], batch['shrink_mask'])[0]
        loss_threshold_maps = self.l1_loss(threshold_maps, batch['threshold_map'], batch['threshold_mask'])
        loss_dict = dict(loss_shrink_maps=loss_probability_maps, loss_threshold_maps=loss_threshold_maps)
        # 测试只输出probability_maps
        if pred.size()[1] > 2:
            loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'], batch['shrink_mask'])
            loss_dict['loss_binary_maps'] = loss_binary_maps
            total_loss = self.alpha * loss_probability_maps + self.beta * loss_threshold_maps + loss_binary_maps
            loss_dict['loss'] = total_loss
        else:
            loss_dict['loss'] = loss_probability_maps

        return loss_dict

