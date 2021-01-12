# coding=utf-8  
# @Time   : 2021/1/12 10:54
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from .builder import LOSSES
from .det_basic_loss import DiceLoss


@LOSSES.register_module()
class PSELoss(nn.Module):
    def __init__(self, text_ratio=0.7, eps=1e-6):
        super().__init__()
        self.dice_loss = DiceLoss(eps)
        self.text_ratio = text_ratio

    def ohem_single(self, pred_text, gt_text, training_mask):
        pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

        if pos_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_num = (int)(np.sum(gt_text <= 0.5))
        neg_num = (int)(min(pos_num * 3, neg_num))

        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_score = pred_text[gt_text <= 0.5]
        # 将负样本得分从高到低排序
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        selected_mask = ((pred_text >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    def ohem_batch(self, pred_texts, gt_texts, training_masks):
        pred_texts = pred_texts.data.cpu().numpy()
        gt_texts = gt_texts.data.cpu().numpy()
        training_masks = training_masks.data.cpu().numpy()
        selected_masks = []
        # batch
        for i in range(pred_texts.shape[0]):
            selected_masks.append(self.ohem_single(pred_texts[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

        selected_masks = np.concatenate(selected_masks, 0)
        selected_masks = torch.from_numpy(selected_masks).float()
        return selected_masks

    def forward(self, pred, batch):
        gt_texts = batch['gt_texts']
        gt_kernels = batch['gt_kernels']
        training_masks = batch['training_masks']
        pred_texts = pred[:, -1, :, :]
        pred_kernels = pred[:, :-1, :, :]
        # training mask ohem
        selected_masks = self.ohem_batch(pred_texts, gt_texts, training_masks)
        selected_masks = selected_masks.to(pred.device)
        # lc loss
        loss_text = self.dice_loss(pred_texts, gt_texts, selected_masks)
        # ls loss
        loss_kernels = []
        mask0 = torch.sigmoid(pred_texts).data.cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = selected_masks.to(pred.device)
        kernels_num = gt_kernels.size()[1]
        for i in range(kernels_num):
            pred_kernel_i = pred_kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.dice_loss(pred_kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)

        loss_kernels = torch.stack(loss_kernels).mean(0)
        loss_text = loss_text.mean()
        loss_kernels = loss_kernels.mean()

        loss = self.text_ratio * loss_text + (1 - self.text_ratio) * loss_kernels

        loss_dict = {'loss': loss, 'loss_text': loss_text, 'loss_kernels': loss_kernels}

        return loss_dict
