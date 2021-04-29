# coding=utf-8  
# @Time   : 2020/12/1 11:13
# @Auto   : zzf-jeff

from .db_postprocess import DBPostProcess
from .rec_postprocess import CTCLabelDecode
from .pixellink_postprocess import PixelLinkPostProcess
from .builder import (POSTPROCESS, build_postprocess)

__all__ = [
    'POSTPROCESS', 'build_postprocess',
    'PixelLinkPostProcess',
    'DBPostProcess',
    'CTCLabelDecode'
]
