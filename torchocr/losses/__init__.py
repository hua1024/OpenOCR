# coding=utf-8  
# @Time   : 2020/12/1 11:11
# @Auto   : zzf-jeff

from .ctc_loss import *
from .db_loss import *
from .builder import (LOSSES, build_loss)

__all__ = [
    'LOSSES', 'build_loss'
]
