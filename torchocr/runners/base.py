# coding=utf-8  
# @Time   : 2020/12/28 10:55
# @Auto   : zzf-jeff


'''
learn for :https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/base_runner.py
'''
from collections import OrderedDict
import torch
from torch.optim import Optimizer
import os.path as osp
from abc import abstractmethod
from torchocr.utils.checkpoints import load_checkpoint
from ..utils import check, file_util
from torchocr.models.utils.ema import ModelEMA


class BaseRunner(object):
    def __init__(self,
                 global_cfg,
                 model,
                 optimizer,
                 lr_scheduler,
                 postprocess,
                 criterion,
                 train_loader,
                 eval_loader,
                 metric,
                 logger,
                 max_epochs=None,
                 max_iters=None,
                 meta=None):
        super(BaseRunner, self).__init__()
        # 上面应该对传进来的参数进行类型验证
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric = metric
        self.postprocess = postprocess
        self.logger = logger
        self.global_cfg = global_cfg
        self.lr_scheduler = lr_scheduler
        self.mode = None
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.meta = meta

        if check.is_str(global_cfg.work_dir):
            self.work_dir = osp.abspath(global_cfg.work_dir)
            file_util.mkdir_or_exist(self.work_dir)
        else:
            raise TypeError('"work_dir" must be a str')

        if global_cfg.is_ema:
            self.ema = self.init_ema()
        else:
            self.ema = None

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')

        self._max_epochs = global_cfg.total_epochs
        self._max_iters = self._max_epochs * len(train_loader)

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

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
    def run(self, data_loaders, **kwargs):
        pass

    @abstractmethod
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl,
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        pass

    def init_ema(self):
        return ModelEMA(self.model, decay=0.9999, updates=0)

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif check.is_dict(self.optimizer):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value if _loss is not None)
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key and 'Y' not in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.detach().item()

        return loss, log_vars

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger_info('load checkpoint from {}'.format(filename))
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
        self.logger_info('resumed epoch {}, iter {}'.format(self.epoch, self.iter))

    def set_input(self, data_batch):
        """dict 可变参数传递

        :param data_batch:
        :return:
        """
        for key, value in data_batch.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    data_batch[key] = value.to(self.model.device)

    def logger_info(self, s):
        if self.global_cfg['local_rank'] == 0:
            self.logger.info(s)
