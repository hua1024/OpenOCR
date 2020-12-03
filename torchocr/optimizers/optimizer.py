# coding=utf-8  
# @Time   : 2020/12/3 11:44
# @Auto   : zzf-jeff
from .builder import OPTIMIZER
import torch


@OPTIMIZER.register_module()
class AdamDecay(object):
    def __init__(self, params):
        self.weight_decay = params['weight_decay']
        self.momentum = params['momentum']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']

    def __call__(self, model, lr):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     betas=(self.beta1, self.beta2),
                                     weight_decay=self.weight_decay)
        return optimizer


@OPTIMIZER.register_module()
class SGDDecay(object):
    def __init__(self, params):
        self.weight_decay = params['weight_decay']
        self.momentum = params['momentum']

    def __call__(self, model, lr):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        return optimizer


@OPTIMIZER.register_module()
class RMSPropDecay(object):
    def __init__(self, params):
        self.alpha = params['alpha']
        self.weight_decay = params['weight_decay']
        self.momentum = params['momentum']

    def __call__(self, model, lr):
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr,
                                        alpha=self.alpha,
                                        weight_decay=self.weight_decay,
                                        momentum=self.momentum)
        return optimizer
