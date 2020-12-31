# coding=utf-8  
# @Time   : 2020/12/1 11:10
# @Auto   : zzf-jeff

from .ctc_head import CTCHead
from .db_head import DBHead
from .pse_head import PSEHead
from .pan_head import PANHead
from .pixellink_head import PixelHead

__all__ = [
    'CTCHead',
    'DBHead',
    'PSEHead',
    'PANHead',
    'PixelHead'
]
