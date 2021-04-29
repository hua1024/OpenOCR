# coding=utf-8  
# @Time   : 2021/1/11 17:53
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES
from .det_basic_loss import ohem_batch


@LOSSES.register_module()
class PixelLinkLoss(nn.Module):
    def __init__(self, num_neighbours):
        super().__init__()
        self.num_neighbours = num_neighbours

    def forward(self, pred, batch):
        """

        :param pred:
        :param batch:
        :return:
        """
        pred_link = pred['pred_link']
        pred_cls = pred['pred_cls']

        # print('pred_link', pred_link.shape)
        # print('pred_cls', pred_cls.shape)

        pixel_cls_label = batch['cls_label']
        pixel_cls_weight = batch['cls_weight']
        pixel_link_label = batch['link_label']
        pixel_link_weight = batch['link_weight']

        # print('pixel_cls_label', pixel_cls_label.shape)
        # print('pixel_cls_weight', pixel_cls_weight.shape)
        # print('pixel_link_label', pixel_link_label.shape)
        # print('pixel_link_weight', pixel_link_weight.shape)

        pos_mask = (pixel_cls_label > 0)
        neg_mask = (pixel_cls_label == 0)

        # print('pos_mask', pos_mask.shape)
        # print('neg_mask', neg_mask.shape)

        # pixel cls loss
        pixel_cls_loss = F.cross_entropy(pred_cls, pos_mask.to(torch.long), reduction='none')
        pixel_cls_scores = F.softmax(pred_cls, dim=1)
        pixel_neg_scores = pixel_cls_scores[:, 0, :, :]
        # ohem
        selected_neg_pixel_mask = ohem_batch(pixel_neg_scores, pos_mask, neg_mask)
        selected_neg_pixel_mask = selected_neg_pixel_mask.to(pred_cls.device)
        n_pos = pos_mask.view(-1).sum()
        n_neg = selected_neg_pixel_mask.view(-1).sum()

        pixel_cls_weights = (pixel_cls_weight + selected_neg_pixel_mask).to(torch.float)

        cls_loss = (pixel_cls_loss * pixel_cls_weights).view(-1).sum() / (n_pos + n_neg)
        # print('cls_loss', cls_loss)
        # pixel link loss
        shape = pred_link.shape

        if n_pos == 0:
            link_loss = (pred_link * 0).view(-1).sum()
        else:
            pixel_link_logits_flat = pred_link.contiguous().view(shape[0], 2, self.num_neighbours, shape[2], shape[3])
            # link_label_flat = pixel_link_label.permute(0, 3, 1, 2)
            link_label_flat = pixel_link_label
            pixel_link_loss = F.cross_entropy(pixel_link_logits_flat, link_label_flat.to(torch.long), reduction='none')

            def get_loss(label):
                link_mask = (link_label_flat == label)
                # link_weight_mask = pixel_link_weight.permute(0, 3, 1, 2) * link_mask.to(torch.float)
                link_weight_mask = pixel_link_weight * link_mask.to(torch.float)
                n_links = link_weight_mask.view(-1).sum()
                loss = (pixel_link_loss * link_weight_mask).view(-1).sum() / n_links
                return loss

            neg_loss = get_loss(0)
            pos_loss = get_loss(1)
            neg_lambda = 1.0
            link_loss = pos_loss + neg_loss * neg_lambda

        total_loss = cls_loss * 2 + link_loss
        losses = {"loss": total_loss, "link_loss": link_loss, "cls_loss": cls_loss}

        return losses

