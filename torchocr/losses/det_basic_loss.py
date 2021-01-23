# coding=utf-8  
# @Time   : 2020/12/29 22:32
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ohem_single(pred_text, gt_text, training_mask):
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


def ohem_batch(pred_texts, gt_texts, training_masks):
    pred_texts = pred_texts.data.cpu().numpy()
    gt_texts = gt_texts.data.cpu().numpy()
    training_masks = training_masks.data.cpu().numpy()
    selected_masks = []
    # batch
    for i in range(pred_texts.shape[0]):
        selected_masks.append(ohem_single(pred_texts[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = np.concatenate(selected_masks, 0)
    selected_masks = torch.from_numpy(selected_masks).float()
    return selected_masks


class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt,
                mask,
                return_origin=False):
        """bce loss forward

        :param pred: shape :math:`(N, 1, H, W)`, the prediction of network
        :param gt: gt: shape :math:`(N, 1, H, W)`, the target
        :param mask: mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        :param return_origin:
        :return:
        """
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = F.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)
        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        if return_origin:
            return balance_loss, loss
        return balance_loss, None

class DiceLoss(nn.Module):
    def __init__(self,eps=1e-6):
        super(DiceLoss,self).__init__()
        self.eps = eps

    def forward(self,pre_score,gt_score,train_mask):
        pre_score = pre_score.contiguous().view(pre_score.size()[0], -1)
        gt_score = gt_score.contiguous().view(gt_score.size()[0], -1)
        train_mask = train_mask.contiguous().view(train_mask.size()[0], -1)

        pre_score = pre_score * train_mask
        gt_score = gt_score * train_mask

        a = torch.sum(pre_score * gt_score, 1)
        b = torch.sum(pre_score * pre_score, 1) + self.eps
        c = torch.sum(gt_score * gt_score, 1) + self.eps
        d = (2 * a) / (b + c)
        dice_loss = torch.mean(d)
        assert dice_loss <= 1
        return 1 - dice_loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss
