# coding=utf-8  
# @Time   : 2020/12/29 11:55
# @Auto   : zzf-jeff


from .base import BaseEncodeConverter
from ..compose import PIPELINES
import torch
import numpy as np


@PIPELINES.register_module()
class AttnLabelEncode(BaseEncodeConverter):
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(AttnLabelEncode,self).__init__(
            max_text_length,
            character_dict_path,
            character_type,
            use_space_char)

        self.beg_str = "sos"
        self.end_str = "eos"

    def add_special_char(self, dict_character):
        dict_character = [self.beg_str, self.end_str] + dict_character
        return dict_character

