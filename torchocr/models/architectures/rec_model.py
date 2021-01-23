# coding=utf-8  
# @Time   : 2020/12/31 10:39
# @Auto   : zzf-jeff


from ..builder import (build_backbone, build_head, build_neck, build_transform, MODELS)
from ..architectures.base import BaseModel


@MODELS.register_module()
class RecognitionModel(BaseModel):
    def __init__(self, backbone, neck=None, head=None, transform=None, pretrained=None):
        super(RecognitionModel, self).__init__()
        if transform is not None:
            self.transform = build_transform(cfg=transform)
        self.backbone = build_backbone(cfg=backbone)
        if neck is not None:
            self.neck = build_neck(cfg=neck)
        if head is not None:
            self.head = build_head(cfg=head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(RecognitionModel, self).init_weights(pretrained)
        # 这里关于neck，head的weights还需要考虑
        self.backbone.init_weights(pretrained=pretrained)

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

    def forward_dummy(self, img):
        pass
