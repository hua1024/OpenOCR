# coding=utf-8  
# @Time   : 2020/12/1 11:05
# @Auto   : zzf-jeff

from .losses import *
from .backbones import *
from .heads import *
from .recognizer import *
from .necks import *
from .detectors import *


from .builder import (BACKBONES, build_backbone)
from .builder import (LOSSES, build_loss)
from .builder import (HEADS, build_head)
from .builder import (DETECTIONS, build_det)
from .builder import (RECOGNITIONS, build_rec)
from .builder import (NECKS, build_neck)


__all__ = [
    'BACKBONES', 'build_backbone',
    'LOSSES', 'build_loss',
    'HEADS', 'build_head',
    'RECOGNITIONS', 'build_rec',
    'NECKS', 'build_neck',
]
