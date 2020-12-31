# coding=utf-8  
# @Time   : 2020/12/29 12:06
# @Auto   : zzf-jeff


import torch.nn as nn
from ..builder import (build_backbone, build_head, build_neck, RECOGNITIONS)
from .base import BaseRecognition


@RECOGNITIONS.register_module()
class AttenctionOCR(BaseRecognition):
    pass
