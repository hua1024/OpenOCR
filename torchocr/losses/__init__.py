# coding=utf-8  
# @Time   : 2020/12/1 11:11
# @Auto   : zzf-jeff

from .ctc_loss import *
from .db_loss import *
from .east_loss import EASTLoss
from .builder import (LOSSES, build_loss)
from .pse_loss import PSELoss
from .pixellink_loss import PixelLinkLoss
from .c2td_loss import C2TDLoss

__all__ = [
    'LOSSES', 'build_loss',
    'PSELoss',
    'PixelLinkLoss',
    'C2TDLoss'
]
