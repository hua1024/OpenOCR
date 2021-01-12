# coding=utf-8  
# @Time   : 2021/1/9 12:06
# @Auto   : zzf-jeff


import torch
import torch.nn as nn
import torch.nn.functional as F
from .det_basic_loss import DiceLoss
from .builder import LOSSES


@LOSSES.register_module()
class EASTLoss(nn.Module):
    def __init__(self, weight_angle=10, eps=1e-6, **kwargs):
        super(EASTLoss, self).__init__()
        # self.dice_loss = DiceLoss(eps=eps)
        self.eps = eps
        self.weight_angle = weight_angle

    def get_dice_loss(self, gt_score, pred_score):
        inter = torch.sum(gt_score * pred_score)
        union = torch.sum(gt_score) + torch.sum(pred_score) + self.eps
        return 1. - (2 * inter / union)

    def get_geo_loss(self, gt_geo, pred_geo):
        # scale-normalized平滑L1
        d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, split_size_or_sections=1, dim=1)
        d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
        # AABB
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        iou_loss_map = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
        # angel
        angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
        return iou_loss_map, angle_loss_map

    def forward(self, pred, batch):
        pred_score = pred['pred_score']
        pred_geo = pred['pred_geo']
        gt_score_map, gt_geo_map, gt_train_mask = batch['score_map'], batch['geo_map'], batch['training_mask']

        if torch.sum(gt_score_map) < 1:
            return torch.sum(pred_score + pred_geo) * 0

        # dice loss
        dice_loss = self.get_dice_loss(pred_score, gt_score_map * (1 - gt_train_mask))
        # dice_loss = dice_loss * 0.01
        # iou loss
        iou_loss_map, angle_loss_map = self.get_geo_loss(pred_geo, gt_geo_map)

        # geo loss = Laabb+10*Langle
        angle_loss = torch.sum(angle_loss_map * gt_score_map) / torch.sum(gt_score_map)
        iou_loss = torch.sum(iou_loss_map * gt_score_map) / torch.sum(gt_score_map)

        geo_loss = self.weight_angle * angle_loss + iou_loss
        total_loss = dice_loss + geo_loss

        losses = {"loss": total_loss, "dice_loss": dice_loss, "geo_loss": geo_loss}

        return losses


