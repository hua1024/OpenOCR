# coding=utf-8  
# @Time   : 2020/12/31 10:39
# @Auto   : zzf-jeff


import torch.nn as nn
import torch
from ..builder import (build_backbone, build_head, build_neck, DETECTIONS, build_loss)
from .base import BaseDetection
from torchocr.postprocess.builder import build_postprocess


@DETECTIONS.register_module()
class SegmentModel(BaseDetection):
    def __init__(self, backbone, loss, postprocess=None, neck=None, head=None, pretrained=None, device='cpu'):
        super(SegmentModel, self).__init__()
        self.device = device
        self.backbone = build_backbone(cfg=backbone)
        if neck is not None:
            self.neck = build_neck(cfg=neck)
        if head is not None:
            self.head = build_head(cfg=head)
        self.init_weights(pretrained=pretrained)
        self.criterion = build_loss(loss)

        self.criterion = self.criterion.to(self.device)
        if postprocess is not None:
            self.postprocess = build_postprocess(cfg=postprocess)

    def init_weights(self, pretrained=None):
        super(SegmentModel, self).init_weights(pretrained)
        # 这里关于neck，head的weights还需要考虑
        self.backbone.init_weights(pretrained=pretrained)

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, data_batch, **kwargs):
        img = self.set_input(data_batch)
        pred = self.extract_feat(img)
        if self.with_head:
            pred = self.head(pred)
        dict_loss = self.criterion(pred, data_batch)

        return dict_loss

    def simple_test(self, data_batch, **kwargs):
        img = self.set_input(data_batch)
        pred = self.extract_feat(img)
        if self.with_head:
            pred = self.head(pred)
        return pred

    def set_input(self, data_batch):
        img = data_batch['img']
        return img

    def post_process(self, data_batch, **kwargs):
        pass

    def forward_dummy(self, img):
        pass
