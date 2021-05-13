# coding=utf-8  
# @Time   : 2021/1/7 12:11
# @Auto   : zzf-jeff

from .builder import POSTPROCESS

from abc import ABCMeta, abstractmethod
import numpy as np


class BaseDecodeConverter(metaclass=ABCMeta):
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
class CTCLabelDecode(BaseDecodeConverter):
    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             character_type, use_space_char)

    def decode(self, text_index, text_prob=None, is_remove_duplicate=True):
        """CTC decode

        :param text_index:
        :param text_prob:
        :param is_remove_duplicate:
        :return:
        """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                # 先去blank
                if int(text_index[batch_idx][idx]) in ignored_tokens:
                    continue
                # 推理代码在去重复
                if is_remove_duplicate:
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                        batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            # if result is '',np.mean([]) will be warning set nan
            conf_mean = np.mean(conf_list) if text else 1.0
            result_list.append((text, conf_mean,conf_list))
        return result_list

    def __call__(self, preds, batch=None, *args, **kwargs):
        """Recognition postprocess
        CTC后处理
        :param preds:
            batch pred [B,T,C]
        :param batch:
            if using infer batch=None
            if using val label = batch['label']
        :param args:
        :param kwargs:
        :return:
            if using infer, return text
            if using val, return (text,label）
        """
        if batch is not None:
            if isinstance(batch, dict):
                batch = batch['label']
        preds = preds.cpu().detach().numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob)
        if batch is None:
            return text
        label = self.decode(batch, is_remove_duplicate=False)
        return (text, label)


@POSTPROCESS.register_module()
class AttnLabelDecode(BaseDecodeConverter):
    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(AttnLabelDecode, self).__init__(character_dict_path,
                                              character_type, use_space_char)

        self.beg_str = "sos"
        self.end_str = "eos"

    def add_special_char(self, dict_character):
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
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
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx

    def decode(self, text_index, text_prob=None, is_remove_duplicate=True):
        """CTC decode

        :param text_index:
        :param text_prob:
        :param is_remove_duplicate:
        :return:
        """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        [beg_idx, end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                # 先去blank
                if int(text_index[batch_idx][idx]) in ignored_tokens:
                    continue
                # 结束符
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                # 推理代码在去重复
                if is_remove_duplicate:
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                        batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            # if result is '',np.mean([]) will be warning set nan
            conf_mean = np.mean(conf_list) if text else 1.0
            result_list.append((text, conf_mean))

        return result_list

    def __call__(self, preds, batch=None, *args, **kwargs):
        """Recognition postprocess
        seq2seq后处理，Attention解码
        :param preds:
            batch pred [B,T,C]
        :param batch:
            if using infer batch=None
            if using val label = batch['label']
        :param args:
        :param kwargs:
        :return:
            if using infer, return text
            if using val, return (text,label）
        """
        if batch is not None:
            if isinstance(batch, dict):
                batch = batch['label']
        preds = preds.cpu().detach().numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob)
        if batch is None:
            return text
        label = self.decode(batch, is_remove_duplicate=False)
        return (text, label)
