# coding=utf-8  
# @Time   : 2020/12/4 9:46
# @Auto   : zzf-jeff

from ..utils.torch_utils import select_device
from ..lr_schedulers import build_lr_scheduler
from ..optimizers import build_optimizer
from ..models import build_rec, build_loss
from ..converters import build_converter
from ..datasets import (build_rec_dataset, build_rec_dataloader)
from ..utils.logger import setup_logger


class RecTrainer(object):
    def __init__(self, model_cfg, train_cfg, hyp_cfg):
        super(RecTrainer, self).__init__()
        self.train_dataloader = self._build_dataloader(train_cfg['data']['train'])
        if 'val' in train_cfg['data']:
            self.val_dataloader = self._build_dataloader(
                train_cfg['data']['val'])
        else:
            self.val_dataloader = None

        self.max_iterations = hyp_cfg['max_iterations']
        self.max_epochs = hyp_cfg['max_epochs']

        self.logger = setup_logger('trainer', distributed_rank=-1)

        self.converter = self._build_converter(hyp_cfg['converters'])

        self.model = self._build_model(cfg=model_cfg)
        self.criterion = self._build_criterion(hyp_cfg['criterion'])
        self.optimizer = self._build_optimizer(hyp_cfg['optimizer'])(self.model, hyp_cfg['init_lr'])
        self.lr_scheduler = self._build_lr_scheduler(hyp_cfg['lr_scheduler'])(self.optimizer)

    def train_batch(self):
        pass

    def validate_batch(self):
        pass

    def dist_run(self):
        pass

    def non_dist_run(self):
        pass

    def _build_dataloader(self, cfg):
        dataset = build_rec_dataset(cfg=cfg['dataset'])
        dataloader = build_rec_dataloader()
        return dataloader

    def _build_optimizer(self, cfg):
        return build_optimizer(cfg)

    def _build_model(self, cfg):
        return build_rec(cfg)

    def _build_criterion(self, cfg):
        return build_loss(cfg)

    def _build_lr_scheduler(self, cfg):
        return build_lr_scheduler(cfg)

    def _build_converter(self, cfg):
        return build_converter(cfg)
