# coding=utf-8  
# @Time   : 2020/12/3 11:44
# @Auto   : zzf-jeff
from .builder import OPTIMIZER
import torch


@OPTIMIZER.register_module()
class Adam(object):
    def __init__(self, weight_decay, momentum, beta1, beta2, lr, **kwargs):
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr

    def __call__(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                     betas=(self.beta1, self.beta2),
                                     weight_decay=self.weight_decay)
        return optimizer


@OPTIMIZER.register_module()
class SGD(object):
    def __init__(self, weight_decay, momentum, lr, **kwargs):
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr = lr

    def __call__(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        return optimizer


@OPTIMIZER.register_module()
class RMSP(object):
    def __init__(self, alpha, weight_decay, momentum, lr, **kwargs):
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr = lr

    def __call__(self, model):
        optimizer = torch.optim.RMSprop(model.parameters(), lr=self.lr,
                                        alpha=self.alpha,
                                        weight_decay=self.weight_decay,
                                        momentum=self.momentum)
        return optimizer
