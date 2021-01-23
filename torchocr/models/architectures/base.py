# coding=utf-8  
# @Time   : 2020/12/2 23:55
# @Auto   : zzf-jeff

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import torch
import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(BaseModel, self).__init__()


    @property
    def with_transform(self):
        """构建neck结构

        :return:
        """
        return hasattr(self, 'transform') and self.transform is not None

    @property
    def with_head(self):
        """构建head结构

        :return:
        """
        return hasattr(self, 'head') and self.head is not None

    @property
    def with_neck(self):
        """构建neck结构

        :return:
        """
        return hasattr(self, 'neck') and self.neck is not None


    @abstractmethod
    def extract_feat(self, img):
        """特征提取，包括backbone+neck

        :param img:
        :return:
        """
        pass

    @abstractmethod
    def forward(self, img, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print('load model from: {}'.format(pretrained))
