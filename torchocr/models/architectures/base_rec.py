# coding=utf-8  
# @Time   : 2020/12/2 23:55
# @Auto   : zzf-jeff

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import torch
import torch.nn as nn


class BaseRecognition(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(BaseRecognition, self).__init__()

    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    @abstractmethod
    def forward_train(self, imgs, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    def forward(self, img, **kwargs):
        return self.forward_train(img, **kwargs)
