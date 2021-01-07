# coding=utf-8  
# @Time   : 2020/12/1 14:53
# @Auto   : zzf-jeff

import torch
import torch.nn as nn
from torch.autograd import Variable

from .builder import LOSSES


@LOSSES.register_module()
class CTCLoss(nn.Module):
    def __init__(self, zero_infinity=True, blank=0, reduction='mean'):
        """torch 内置ctc loss的API使用

        :param zero_infinity: Bool for set inf loss to zero
        :param blank: The blank char, default is zero
        :param reduction: Process output loss ,default is 'mean'('none','sum','mean')
        """
        super(CTCLoss, self).__init__()
        self.criterion = nn.CTCLoss(zero_infinity=zero_infinity, blank=blank, reduction=reduction)

    def forward(self, pred, batch):
        """Run ctc loss

        :param pred: Input pred,shape
        :param target: Label shape: (N,S) or sum(target_lengths)
        :param target_length: Label length shape : (N)
        :return:
        """
        batch_size = pred.size(0)
        pred = pred.log_softmax(2)
        pred = pred.permute(1, 0, 2)
        # timestep * batchsize
        pred_length = torch.tensor([pred.size(0)] * batch_size, dtype=torch.long)
        target = batch['label']
        target_length = batch['length']
        loss = self.criterion(pred, target, pred_length, target_length)

        return {'loss': loss}
