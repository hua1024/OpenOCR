# coding=utf-8  
# @Time   : 2020/12/29 11:55
# @Auto   : zzf-jeff


from .base import BaseEncodeConverter
from ..compose import PIPELINES
import torch
import numpy as np
import os


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

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "Unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx

    def __call__(self, data):
        try:
            if data is None:
                return None
            text = data['label']
            text = self.encode(text)
            if text is None:
                return None

            if len(text) > self.max_text_len:
                return None

            data['length'] = np.array(len(text))
            # ?
            text = [0] + text + [len(self.character) - 1] + [0] * (self.max_text_len - len(text) - 2)
            data['label'] = np.array(text)
            return data
        except Exception as e:
            file_name = os.path.basename(__file__).split(".")[0]
            print('{} --> '.format(file_name), e)
            return None
