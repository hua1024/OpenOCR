# coding=utf-8  
# @Time   : 2020/12/29 22:32
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# class DiceLoss(nn.Module):
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#
#     def forward(self,
#                 pred: torch.Tensor,
#                 gt,
#                 mask,
#                 weights=None):
#         return self._compute(pred, gt, mask, weights)
#
#     def _compute(self, pred, gt, mask, weights):
#         # if pred.dim() == 4:
#         #     pred = pred[:, 0, :, :]
#         #     gt = gt[:, 0, :, :]
#         assert pred.shape == gt.shape
#         assert pred.shape == mask.shape
#         if weights is not None:
#             assert weights.shape == mask.shape
#             mask = weights * mask
#         intersection = (pred * gt * mask).sum()
#         union = (pred * mask).sum() + (gt * mask).sum() + self.eps
#         loss = 1 - 2.0 * intersection / union
#         assert loss <= 1
#         return loss

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
