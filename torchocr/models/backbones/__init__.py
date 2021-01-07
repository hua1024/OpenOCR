# coding=utf-8  
# @Time   : 2020/12/1 11:10
# @Auto   : zzf-jeff

from .rec_densenet import RecDenseNet
from .rec_resnet import RecResNet
from .det_resnet import DetResNet
from .det_vgg import PixelVGG

__all__ = [
    'RecDenseNet',
    'DetResNet',
    'PixelVGG',
    'RecResNet',
]
