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
        super(AttnLabelEncode, self).__init__(
            max_text_length,
            character_dict_path,
            character_type,
            use_space_char)

        self.beg_str = "sos"
        self.end_str = "eos"

    def encode(self, text):
        """Support batch or single str.

        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index .
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list

    def add_special_char(self, dict_character):
        dict_character = [self.beg_str, self.end_str] + dict_character
        return dict_character

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        return data
