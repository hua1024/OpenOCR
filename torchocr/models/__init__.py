# coding=utf-8  
# @Time   : 2020/12/1 11:05
# @Auto   : zzf-jeff

from .losses import *
from .backbones import *
from .heads import *
from .architectures import *
from .necks import *


from .builder import (BACKBONE, build_backbone)
from .builder import (LOSS, build_loss)
from .builder import (HEAD, build_head)
from .builder import (DETECTION, build_det)
from .builder import (RECOGNITION, build_rec)
from .builder import (NECK, build_neck)


__all__ = [
    'BACKBONE', 'build_backbone',
    'LOSS', 'build_loss',
    'HEAD', 'build_head',
    'RECOGNITION', 'build_rec',
    'NECK', 'build_neck',
]
