# coding=utf-8  
# @Time   : 2021/1/8 14:17
# @Auto   : zzf-jeff


import torch
import torch.nn as nn
import torch.nn.functional as F
# from .builder import LOSSES


# @LOSSES.register_module()
class C2TDLoss(nn.Module):

    def __init__(self, thresh=0.5, neg_pos=3):
        super().__init__()
        self.thresh = thresh
        self.negpos_ratio = neg_pos

    def forward(self,y_pred,y_true):
        """

        :param pred:
        :param batch: [batch_size, h, w, (score, top, bottom)]
                score: heat_map 概率
                top: 上边界回归距离
                bottom: 下边界回归距离
        :return:
        """
        y_pred = y_pred.permute(0, 2, 3, 1)
        batch_size, h, w, _ = y_true.shape
        labels = y_true[:, :, :, 0]
        logits = y_pred[:, :, :, 0]
        # print('y_pred', y_pred.shape)
        # print('y_true', y_true.shape)
        pixel_cls_weight = y_true[:, :, :, 3]

        pos = labels >= self.thresh  # positive 是跟labels 同样尺寸大小的 0 - 1 tensor
        # negative = labels < self.thresh

        num_pos = torch.sum(pos.long(), dim=[1, 2]).unsqueeze(1)  # batch_size * 1

        # 分类这里要进行OHEM， 参照SSD pytorch的写法
        loss_c = torch.abs(labels - logits)
        loss_c[pos] = 0.
        loss_c = loss_c.view(batch_size, -1)
        _, loss_idx = loss_c.sort(1, descending=True)  # 我也不知道写的啥，但我照着写了
        _, idx_rank = loss_idx.sort(1)  # ???, 怎么还排序了???  得到对应像素排序的序号
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=loss_c.size(
            1) - 1)  # Clamp all elements in input into the range [ min, max ] and return a resulting tensor:
        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg = neg.view(batch_size, h, w)

        # 正负样本均衡 1:3
        pos_loss = torch.abs(torch.add(labels[pos], logits[pos] * (-1)))
        pos_loss = torch.mul(pos_loss, pixel_cls_weight[pos])
        all_num_pos = torch.sum(num_pos)

        # heatmap_loss = \
        #     self.negpos_ratio*F.smooth_l1_loss(labels[pos], logits[pos]) +\
        #     F.smooth_l1_loss(labels[neg], logits[neg])
        # change by hcn
        heatmap_loss = \
            self.negpos_ratio * torch.sum(pos_loss) / all_num_pos + \
            F.smooth_l1_loss(labels[neg], logits[neg])

        # 对于其中的正样本要预测上下的坐标
        cord_true = y_true[:, :, :, 1:3]
        cord_pred = y_pred[:, :, :, 1:3]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(cord_true)  # 将最后一维扩充， 再进行
        location_loss = F.smooth_l1_loss(cord_true[pos_idx], cord_pred[pos_idx])
        return heatmap_loss, location_loss

if __name__ == '__main__':
    centre_line_loss = C2TDLoss(0.5)
    y_true = torch.rand(1, 100, 100, 3)
    y_pred = torch.rand(1, 100, 100, 3)
    centre_line_loss(y_true, y_pred)