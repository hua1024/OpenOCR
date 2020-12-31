# coding=utf-8  
# @Time   : 2020/12/23 14:17
# @Auto   : zzf-jeff


import torch.nn as nn
from ..builder import DETECTIONS
from .segment import SegmentModel


@DETECTIONS.register_module()
class PANNet(SegmentModel):
    def __init__(self, backbone, loss, postprocess=None, neck=None, head=None, pretrained=None):
        super(PANNet, self).__init__(
            backbone=backbone,
            loss=loss,
            postprocess=postprocess,
            neck=neck,
            head=head,
            pretrained=pretrained
        )

    def forward_test(self, img, **kwargs):
        pass
