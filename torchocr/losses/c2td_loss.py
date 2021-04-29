# coding=utf-8  
# @Time   : 2021/1/8 14:17
# @Auto   : zzf-jeff


import torch
import torch.nn as nn
import torch.nn.functional as F
from .det_basic_loss import ohem_batch

from .builder import LOSSES


@LOSSES.register_module()
class C2TDLoss(nn.Module):

    def __init__(self, thresh=0.5, neg_pos=3):
        super().__init__()
        self.thresh = thresh
        self.negpos_ratio = neg_pos

    def forward(self, pred, batch):
        """

        :param pred: [batch_size, h, w, 3]
        :param batch: [batch_size, h, w, (score, top, bottom, weight)]
                score: heat_map 概率
                top: 上边界回归距离
                bottom: 下边界回归距离
                weight: pixel_cls_weight
        :return:
        """


        heat_map = batch['heat_map']
        print('heat_map',heat_map.shape)
        print('pred', pred.shape)
        batch_size, h, w, _ = heat_map.shape

        labels = heat_map[:, 0, :, :]
        print('labels', labels.shape)
        logits = pred[:, 0:1, :, :]
        print('logits', logits.shape)

        pixel_cls_weight = heat_map[:, 3, :, :]

        pos_mask = (labels >= self.thresh)
        neg_mask = (labels == 0)

        # pixel cls loss
        pixel_cls_loss = F.cross_entropy(logits, pos_mask.to(torch.long), reduction='none')
        pixel_cls_scores = F.softmax(logits, dim=1)
        pixel_neg_scores = pixel_cls_scores[:, 0, :, :]
        # ohem
        selected_neg_pixel_mask = ohem_batch(pixel_neg_scores, pos_mask, neg_mask)
        selected_neg_pixel_mask = selected_neg_pixel_mask.to(logits.device)
        n_pos = pos_mask.view(-1).sum()
        n_neg = selected_neg_pixel_mask.view(-1).sum()

        pixel_cls_weights = (pixel_cls_weight + selected_neg_pixel_mask).to(torch.float)

        heatmap_loss = (pixel_cls_loss * pixel_cls_weights).view(-1).sum() / (n_pos + n_neg)
        print('heatmap_loss', heatmap_loss)
        # pixel link loss



        # 对于其中的正样本要预测上下的坐标
        cord_true = heat_map[:, 1:3, :, :]
        cord_pred = pred[:, 1:3, :, :]

        print()

        # pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(cord_true)  # 将最后一维扩充， 再进行
        # location_loss = F.smooth_l1_loss(cord_true[pos_idx], cord_pred[pos_idx])
        # total_loss = heatmap_loss + location_loss
        total_loss = heatmap_loss
        losses = {"loss": total_loss, "location_loss": '', "heatmap_loss": heatmap_loss}

        return losses


if __name__ == '__main__':
    centre_line_loss = C2TDLoss(0.5)
    y_true = torch.rand(1, 100, 100, 3)
    y_pred = torch.rand(1, 100, 100, 3)
    centre_line_loss(y_true, y_pred)
