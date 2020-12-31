# coding=utf-8  
# @Time   : 2020/12/23 18:37
# @Auto   : zzf-jeff


import torch.nn as nn
from ..builder import (build_backbone, build_head, build_neck, DETECTIONS)
from .base import BaseDetection

@DETECTIONS.register_module()
class PixelLink(BaseDetection):
    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(PixelLink, self).__init__()
        self.backbone = build_backbone(cfg=backbone)
        if neck is not None:
            self.neck = build_neck(cfg=neck)
        if head is not None:
            self.head = build_head(cfg=head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, **kwargs):
        x = self.extract_feat(img)
        if self.with_head:
            x = self.head(x)
        return x