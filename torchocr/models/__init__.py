# coding=utf-8  
# @Time   : 2020/12/1 11:05
# @Auto   : zzf-jeff

from .backbones import *
from .heads import *
from .necks import *
from .transforms import *
from .architectures import *

from .builder import (BACKBONES, build_backbone)
from .builder import (HEADS, build_head)
from .builder import (NECKS, build_neck)
from .builder import (TRANSFORMS,build_transform)
from .builder import (MODELS,build_model)


__all__ = [
    'BACKBONES', 'build_backbone',
    'HEADS', 'build_head',
    'NECKS', 'build_neck',
]
