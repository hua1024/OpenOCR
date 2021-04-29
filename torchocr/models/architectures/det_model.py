# coding=utf-8  
# @Time   : 2020/12/31 10:39
# @Auto   : zzf-jeff


from torchocr.models.builder import (build_backbone, build_head, build_neck, build_transform, MODELS)
from torchocr.models.architectures.base import BaseModel


@MODELS.register_module()
class DetectionModel(BaseModel):
    def __init__(self, backbone, neck=None, head=None, transform=None, pretrained=None):
        super(DetectionModel, self).__init__()
        if transform is not None:
            self.transform = build_transform(cfg=transform)
        self.backbone = build_backbone(cfg=backbone)
        if neck is not None:
            self.neck = build_neck(cfg=neck)
        if head is not None:
            self.head = build_head(cfg=head)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(DetectionModel, self).init_weights(pretrained)
        if self.training:
            self.backbone.init_weights(pretrained=pretrained)
            if self.with_neck:
                self.neck.init_weights(pretrained=pretrained)
            if self.with_head:
                self.head.init_weights(pretrained=pretrained)

    def extract_feat(self, img):
        if self.with_transform:
            img = self.transform(img)
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, img, **kwargs):
        pred = self.extract_feat(img)
        if self.with_head:
            pred = self.head(pred)
        return pred
