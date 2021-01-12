# coding=utf-8  
# @Time   : 2020/12/2 18:49
# @Auto   : zzf-jeff

from abc import ABCMeta, abstractmethod


class BaseEncodeConverter(metaclass=ABCMeta):
    def __init__(self,
                 max_text_length,
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

        self.max_text_len = max_text_length
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
    def encode(self, *args, **kwargs):
        pass

    def get_ignored_tokens(self):
        return [0]  # for ctc blank