# coding=utf-8  
# @Time   : 2020/12/2 18:49
# @Auto   : zzf-jeff

from abc import ABCMeta, abstractmethod


class BaseConverter(metaclass=ABCMeta):
    def __init__(self, character):
        self.character = list(character)
        self.dict = {}
        for idx, char in enumerate(self.character):
            self.dict[char] = idx

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass
