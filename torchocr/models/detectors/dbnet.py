# coding=utf-8  
# @Time   : 2020/12/2 23:56
# @Auto   : zzf-jeff

import torch.nn as nn
from ..builder import DETECTIONS
from .segment import SegmentModel


@DETECTIONS.register_module()
class DBNet(SegmentModel):
    def __init__(self, backbone, loss, postprocess=None, neck=None, head=None, pretrained=None,device='cpu'):
        super(DBNet, self).__init__(
            backbone=backbone,
            loss=loss,
            postprocess=postprocess,
            neck=neck,
            head=head,
            pretrained=pretrained,
            device=device
        )


    def post_process(self, data_batch, **kwargs):
        pred = self.simple_test(data_batch)
        img_h_w_list = data_batch['img_h_w_list']
        out_list = self.postprocess(pred, img_h_w_list)
        return out_list
