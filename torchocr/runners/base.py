# coding=utf-8  
# @Time   : 2020/12/28 10:55
# @Auto   : zzf-jeff


'''
learn for :https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/base_runner.py
'''

import torch
from torch.optim import Optimizer
import logging
import os.path as osp
from abc import ABCMeta, abstractmethod

from .priority import get_priority
from .checkpoints import load_checkpoint
from .iter_timer import IterTimerHook
from .log_buffer import LogBuffer

from ..hooks import HOOKS, Hook
from ..utils import checks
from ..utils import files
from ..utils.registry import build_from_cfg


class BaseRunner(metaclass=ABCMeta):
    def __init(self,
               model,
               batch_processor=None,
               optimizer=None,
               work_dir=None,
               logger=None,
               meta=None,
               max_iters=None,
               max_epochs=None):
        super().__init__()

        # 上面应该对传进来的参数进行类型验证
        self.model = model
        self.batch_processor = batch_processor
        self.optimizer = optimizer
        self.logger = logger
        self.meta = meta

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        if checks.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            files.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')

        self._max_epochs = max_epochs
        self._max_iters = max_iters
        self.log_buffer = LogBuffer()

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    @abstractmethod
    def train(self, data_loaders, **kwargs):
        pass

    @abstractmethod
    def val(self, data_loader, **kwargs):
        pass

    @abstractmethod
    def run(self, data_loaders, workflow, **kwargs):
        pass

    @abstractmethod
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl,
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        pass

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif checks.is_dict(self.optimizer):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def register_hook_from_cfg(self, hook_cfg):
        """Register a hook from its cfg.

        Args:
            hook_cfg (dict): Hook config. It should have at least keys 'type'
              and 'priority' indicating its type and priority.

        Notes:
            The specific hook class to register should not use 'type' and
            'priority' arguments during initialization.
        """
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop('priority', 'NORMAL')
        hook = build_from_cfg(hook_cfg, HOOKS)
        self.register_hook(hook, priority=priority)

    def call_hook(self, fn_name):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def resume(self, checkpoint, resume_optimizer=True, map_location='default'):
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = load_checkpoint(self.model, checkpoint,
                                             map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            # 考虑了多个不同的优化器
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer but got {}'.format(type(self.optimizer)))
        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def register_checkpoint_hook(self, checkpoint_config):
        if checkpoint_config is None:
            return
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = build_from_cfg(checkpoint_config, HOOKS)
        else:
            hook = checkpoint_config
        self.register_hook(hook)

    def register_logger_hooks(self, log_config):
        if log_config is None:
            return
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = build_from_cfg(info, HOOKS, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook)

    # def register_training_hooks(self,
    #                             lr_config,
    #                             optimizer_config=None,
    #                             checkpoint_config=None,
    #                             log_config=None,
    #                             momentum_config=None):
    #     """Register default hooks for training.
    #
    #     Default hooks include:
    #
    #     - LrUpdaterHook
    #     - MomentumUpdaterHook
    #     - OptimizerStepperHook
    #     - CheckpointSaverHook
    #     - IterTimerHook
    #     - LoggerHook(s)
    #     """
    #     self.register_lr_hook(lr_config)
    #     self.register_momentum_hook(momentum_config)
    #     self.register_optimizer_hook(optimizer_config)
    #     self.register_checkpoint_hook(checkpoint_config)
    #     self.register_hook(IterTimerHook())
    #     self.register_logger_hooks(log_config)
    #

    def register_training_hooks(self,
                                checkpoint_config=None,
                                log_config=None, ):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - MomentumUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        self.register_checkpoint_hook(checkpoint_config)
        self.register_hook(IterTimerHook())
        self.register_logger_hooks(log_config)
