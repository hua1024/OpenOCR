# coding=utf-8  
# @Time   : 2020/12/2 18:55
# @Auto   : zzf-jeff
from .base import BaseEncodeConverter
from ..compose import PIPELINES
import torch
import numpy as np
import os


## todo:decode是否放在后处理部分会好一点
@PIPELINES.register_module()
class CTCLabelEncode(BaseEncodeConverter):
    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        """encode label

        :param max_text_length:
        :param character_dict_path:
        :param character_type:
        :param use_space_char:
        :param kwargs:
        """
        super(CTCLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             character_type, use_space_char)

    def encode(self, text):
        """Support batch or single str.

        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
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
        # 默认ctc将-(blank)放在开头
        dict_character = ['blank'] + dict_character
        return dict_character

    def __call__(self, data):
        try:
            if data is None:
                return None
            text = data['label']
            text = self.encode(text)
            if text is None:
                print('Label is not in rec-key')
                return None

            data['length'] = np.array(len(text))
            text = text + [0] * (self.max_text_len - len(text))
            data['label'] = np.array(text)
            return data
        except Exception as e:
            file_name = os.path.basename(__file__).split(".")[0]
            print('{} --> '.format(file_name), e)
            return None
