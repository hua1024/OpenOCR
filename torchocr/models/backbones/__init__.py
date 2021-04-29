# coding=utf-8  
# @Time   : 2020/12/1 11:10
# @Auto   : zzf-jeff


from .det_resnet import DetResNet
from .det_repvgg import DetRepVGG
from .det_vgg import VGGPixelWithDilation, VGGPixel
from .det_mobilenet_v3 import DetMobileNetV3

from .rec_resnet import RecResNet
from .rec_densenet import RecDenseNet
from .rec_cspdensenet import RecCSPDenseNet
from .rec_mobilenet_v3 import RecMobileNetV3

__all__ = [
    'DetMobileNetV3',
    'DetResNet',
    'VGGPixelWithDilation',
    'VGGPixel',
    'DetRepVGG',
    'RecResNet',
    'RecDenseNet',
    'RecCSPDenseNet',
    'RecMobileNetV3'
]
