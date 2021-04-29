# coding=utf-8  
# @Time   : 2021/1/6 10:45
# @Auto   : zzf-jeff


import os.path as osp
import time
import torch
import platform
import shutil
from getpass import getuser
from socket import gethostname
from ..utils import path_util
from ..utils.stats import TrainingStats
from .base import BaseRunner
from torchocr.utils.checkpoints import save_checkpoint
from .test_runner import eval

try:
    from torch.cuda import amp
except:
    print('if use amp torch must more than 1.6.0')


class TrainRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

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
                 logger
                 ):
        super(TrainRunner, self).__init__(
            global_cfg,
            model,
            optimizer,
            lr_scheduler,
            postprocess,
            criterion,
            train_loader,
            eval_loader,
            metric,
            logger
        )

        if type(self.global_cfg.eval_batch_step) == list and len(self.global_cfg.eval_batch_step) >= 2:
            self.start_eval_step = self.global_cfg.eval_batch_step[0]
            self.eval_batch_step = self.global_cfg.eval_batch_step[1]
            self.logger_info(
                "During the training process, after the {}th iteration, an evaluation is run every {} iterations".
                    format(self.start_eval_step, self.eval_batch_step))

        log_smooth_window = global_cfg.log_smooth_window
        self.train_stats = TrainingStats(log_smooth_window, ['lr'])

        if self.global_cfg.is_amp:
            self.scaler = amp.GradScaler(enabled=self.global_cfg.is_cuda)

        if metric is not None:
            self.main_indicator = metric.main_indicator
            self.meta = {self.main_indicator: 0.0}

    def train_batch(self, data_batch, **kwargs):
        self.model.train()
        self.mode = 'train'
        batch_start = time.time()
        # 数据加载进device
        self.set_input(data_batch)

        self.optimizer.zero_grad()
        # forward
        if self.global_cfg.is_amp:
            with amp.autocast(enabled=self.global_cfg.is_cuda):
                pred = self.model(data_batch['image'], **kwargs)
                loss_dict = self.criterion(pred, data_batch)
        else:
            pred = self.model(data_batch['image'], **kwargs)
            loss_dict = self.criterion(pred, data_batch)

        # backward
        avg_loss = loss_dict['loss']
        if self.global_cfg.is_amp:
            self.scaler.scale(avg_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            avg_loss.backward()
            self.optimizer.step()

        # ema model
        if self.ema is not None:
            self.ema.update(self.model)

        train_batch_cost = time.time() - batch_start
        # info
        lr = self.current_lr()[0]
        loss, log_vars = self.parse_losses(loss_dict)
        stats = {k: v for k, v in log_vars.items()}
        stats['lr'] = lr
        self.train_stats.update(stats)

        if self._iter > 0 and self._iter % self.global_cfg.print_batch_step == 0:
            logs = self.train_stats.log()
            strs = 'epoch: [{}/{}], iter: {}, {} , batch_time: {:.5f}'.format(
                self.epoch, self._max_epochs, self._iter, logs, train_batch_cost)
            self.logger_info(strs)

        self._iter += 1

    def run(self, **kwargs):
        self.logger_info(
            'Start running, host: {}@{}, work_dir: {} '.format(self.work_dir, getuser(), gethostname()))
        self.logger_info('max: {} epochs, max {} iters'.format(self._max_epochs, self._max_iters))

        for epoch in range(self.epoch, self._max_epochs):
            for i, data_batch in enumerate(self.train_loader):
                if i >= len(self.train_loader):
                    break
                # batch train
                # if dist ,DistributedSampler need set epoch to shuffle data
                if self.global_cfg['distributed']:
                    self.train_loader.sampler.set_epoch(epoch)
                self.train_batch(data_batch)
                # eval
                if self._iter > self.start_eval_step and \
                        (self._iter - self.start_eval_step) % self.eval_batch_step == 0 \
                        and self.global_cfg.is_eval and self.global_cfg['local_rank'] == 0:

                    if self.ema is not None:
                        model = self.ema.ema.module if hasattr(self.ema, 'module') else self.ema.ema
                        # get model device to test
                        model.device = self.model.device
                        cur_metirc = eval(model, self.eval_loader, self.postprocess,
                                          self.metric)
                    else:
                        cur_metirc = eval(self.model, self.eval_loader, self.postprocess,
                                          self.metric)
                    cur_metirc_str = 'cur metirc, {}'.format(', '.join(
                        ['{}: {}'.format(k, v) for k, v in cur_metirc.items()]))
                    self.logger_info(cur_metirc_str)

                    if cur_metirc[self.main_indicator] >= self.meta[self.main_indicator]:
                        self.meta[self.main_indicator] = cur_metirc[self.main_indicator]
                        self.meta['best_epoch'] = epoch
                        self.save_checkpoint(self.work_dir, filename_tmpl='best.pth')
            # end batch
            self.lr_scheduler.step()
            self._epoch += 1
            if (self._epoch + 1) % self.global_cfg.checkpoint_interval_epoch == 0 or \
                    (self._epoch + 1) == self.global_cfg.total_epochs:
                self.save_checkpoint(self.work_dir)

    def save_checkpoint(self, out_dir, filename_tmpl='epoch_{}.pth', save_optimizer=True,
                        meta=None, create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if self.global_cfg['local_rank'] == 0:
            if meta is None:
                meta = dict(epoch=self.epoch, iter=self.iter)
            elif isinstance(meta, dict):
                meta.update(epoch=self.epoch, iter=self.iter)
            else:
                raise TypeError('meta should be a dict or None, but got {}'.format(type(meta)))
            if self.meta is not None:
                meta.update(self.meta)
            filename = filename_tmpl.format(self.epoch)
            filepath = osp.join(out_dir, filename)
            optimizer = self.optimizer if save_optimizer else None

            if self.ema is not None:
                model = self.ema.ema.module if hasattr(self.ema, 'module') else self.ema.ema
                save_checkpoint(model, filepath, optimizer=optimizer, meta=meta)
            else:
                save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)

            # in some environments, `os.symlink` is not supported, you may need to
            # set `create_symlink` to False
            try:
                if create_symlink:
                    dst_file = osp.join(out_dir, 'latest.pth')
                    if platform.system() != 'Windows':
                        path_util.symlink(filename, dst_file)
                    else:
                        shutil.copy(filepath, dst_file)
            except:
                self.logger.warning('create_symlink failed')
