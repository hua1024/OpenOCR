# coding=utf-8  
# @Time   : 2020/12/2 23:55
# @Auto   : zzf-jeff

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import torch
import torch.nn as nn


class BaseDetection(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(BaseDetection, self).__init__()

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

    @property
    def with_postprocess(self):
        """构建后处理

        :return:
        """
        return hasattr(self, 'postprocess') and self.postprocess is not None

    @abstractmethod
    def extract_feat(self, img):
        """特征提取，包括backbone+neck

        :param img:
        :return:
        """
        pass

    @abstractmethod
    def forward_train(self, data_batch, **kwargs):
        """网络train function,主要输出loss

        :param data_batch:
        :param kwargs:
        :return:
        """
        pass

    def forward_test(self, data_batch, **kwargs):
        return self.simple_test(data_batch, **kwargs)

    @abstractmethod
    def simple_test(self, data_batch, **kwargs):
        pass

    def forward(self, data_batch, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(data_batch, **kwargs)
        else:
            return self.forward_test(data_batch, **kwargs)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print('load model from: {}'.format(pretrained))

    @abstractmethod
    def set_input(self, data_batch):
        pass
