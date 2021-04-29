# coding=utf-8  
# @Time   : 2021/1/7 9:55
# @Auto   : zzf-jeff

from .make_shrink_map import MakeShrinkMap
from .make_border_map import MakeBorderMap
from .east_process import EASTProcessTrain
from .pse_process import PSEProcessTrain
from .pixellink_process import PixelLinkProcessTrain
from .c2td_process import C2TDProcessTrain

__all__ = [
    'MakeShrinkMap',
    'MakeBorderMap',
    'EASTProcessTrain',
    'PSEProcessTrain',
    'PixelLinkProcessTrain',
    'C2TDProcessTrain'
]
