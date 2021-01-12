# coding=utf-8  
# @Time   : 2021/1/7 12:11
# @Auto   : zzf-jeff

from .builder import POSTPROCESS

from abc import ABCMeta, abstractmethod
import numpy as np


class BaseDncodeConverter(metaclass=ABCMeta):
    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False):
        """rec label converter

        :param max_text_length:
        :param character_dict_path:
        :param character_type:
        :param use_space_char:
        """
        support_character_type = [
            'ch', 'en'
        ]
        assert character_type in support_character_type, "Only {} are supported now but get {}".format(
            support_character_type, character_type)

        if character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type in ["ch"]:
            self.character_str = ""
            assert character_dict_path is not None, "character_dict_path should not be None when character_type is ch"
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)
        else:
            raise Exception('dict_character is empty')
        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


@POSTPROCESS.register_module()
class CTCLabelDecode(BaseDncodeConverter):
    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             character_type, use_space_char)

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                        batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        """

        :param preds:
        :param label:
        :param args:
        :param kwargs:
        :return:
        """
        # todo: 蹩脚的方式
        if label is not None:
            if isinstance(label,dict):
                label = label['label']
        preds = preds.cpu().detach().numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label
