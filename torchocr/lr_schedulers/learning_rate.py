# coding=utf-8  
# @Time   : 2020/12/3 11:53
# @Auto   : zzf-jeff

import torch
import torch.optim as optim

import math
import numpy as np
from .builder import LR_SCHEDULER

from torch.optim.lr_scheduler import _LRScheduler


# class WarmUpLR(_LRScheduler):
#     """warmup_training learning rate scheduler
#     Args:
#         optimizer: optimzier(e.g. SGD)
#         total_iters: totoal_iters of warmup phase
#     """
#
#     def __init__(self, optimizer, total_iters, last_epoch=-1):
#         self.total_iters = total_iters
#         super().__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         """we will use the first m batches, and set the learning
#         rate to base_lr * m / total_iters
#         """
#         return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

@LR_SCHEDULER.register_module()
class DecayLearningRate(object):

    def __init__(self, epochs, lr, **kwargs):
        self.epochs = epochs
        self.lr = lr
        self.factor = 0.9

    def __call__(self, optimizer):

        lr_lambda = lambda epoch: float(self.lr) * np.power(1.0 - epoch / float(self.epochs + 1), self.factor).item()

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler


@LR_SCHEDULER.register_module()
class StepLR(object):
    # 等间隔调整学习率 StepLR
    def __init__(self, step_size, gamma, **kwargs):
        self.step_size = step_size
        self.gamma = gamma

    def __call__(self, optimizer):
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return scheduler


@LR_SCHEDULER.register_module()
class MultiStepLR(object):
    # 按需调整学习率 MultiStepLR
    def __init__(self, gamma, milestones, **kwargs):
        self.gamma = gamma
        self.milestones = milestones

    def __call__(self, optimizer):
        # 基于LambdaLR
        # multistep_lr = lambda epoch: gamma**len([m for m in milestones if m <= epoch])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones,
                                                   gamma=self.gamma)  # learning rate decay
        return scheduler


@LR_SCHEDULER.register_module()
class MultiStepWarmup(object):
    # 按需调整学习率 MultiStepLR + WarmUp
    def __init__(self, gamma, milestones, warm_up_epochs, **kwargs):
        self.gamma = gamma
        self.milestones = milestones
        self.warm_up_epochs = warm_up_epochs

    def __call__(self, optimizer):
        lr_lambda = lambda epoch: (epoch + 1) / self.warm_up_epochs if epoch <= self.warm_up_epochs \
            else self.gamma ** len([m for m in self.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler


@LR_SCHEDULER.register_module()
class CosineAnnealingLR(object):
    # 以余弦函数为周期，并在每个周期最大值时重新设置学习率。
    # 以初始学习率为最大学习率，以 2∗Tmax 为周期，在一个周期内先下降，后上升。
    def __init__(self, T_max, eta_min=0, **kwargs):
        self.T_max = T_max
        self.eta_min = eta_min

    def __call__(self, optimizer):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.eta_min)
        return scheduler


@LR_SCHEDULER.register_module()
class CosineWarmup(object):
    # CosineAnnealingLR+Warmup
    def __init__(self, warm_up_epochs, epochs, **kwargs):
        self.warm_up_epochs = warm_up_epochs
        self.epochs = epochs

    def __call__(self, optimizer):
        lr_lambda = lambda epoch: (epoch + 1) / self.warm_up_epochs if epoch <= self.warm_up_epochs \
            else 0.5 * (math.cos(
            (epoch - self.warm_up_epochs) / (self.epochs - self.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler
