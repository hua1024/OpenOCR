# coding=utf-8  
# @Time   : 2020/12/4 9:42
# @Auto   : zzf-jeff

import argparse
import os
import sys
import torch
import os.path as osp
import time
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from torchocr.utils.config_util import Config
from torchocr.utils import file_util
from torchocr.utils.logger import Logging, get_logger
from torchocr.utils.torch_util import set_random_seed, select_device
from torchocr.models import build_model
from torchocr.datasets import build_dataset, build_dataloader
from torchocr.utils.collect_env import collect_env
from torchocr.losses import build_loss
from torchocr.optimizers import build_optimizer
from torchocr.lr_schedulers import build_lr_scheduler
from torchocr.metrics import build_metrics
from torchocr.postprocess import build_postprocess
from torchocr.runners.train_runner import TrainRunner


def parse_args():
    parser = argparse.ArgumentParser(description='OCR train')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    # 通用配置
    global_config = cfg.options

    # set cudnn_benchmark,如数据size一致能加快训练
    if global_config.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if global_config.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        global_config.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # create work_dir
    file_util.mkdir_or_exist(global_config.work_dir)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(global_config.work_dir, '{}.log'.format(timestamp))
    logger = get_logger(name='ocr', log_file=log_file)

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([('{}: {}'.format(k, v))
                          for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    ## log some basic info
    logger.info('Config:\n{}'.format(cfg.text))

    # set random seeds
    logger.info('Set random seed to {}, deterministic: {}'.format(global_config.seed, args.deterministic))
    set_random_seed(global_config.seed, deterministic=args.deterministic)

    # build model
    model = build_model(cfg.model)
    device = select_device(global_config.device)
    model = model.to(device)
    # TODO : distributedDataparallel代替DataParallel
    # if device.type != 'cpu' and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.device = device

    # build train dataset
    train_dataset = build_dataset(cfg.data.train)
    train_loader = build_dataloader(train_dataset, data=cfg.data)

    # if is eval , build eval dataloader,postprocess,metric
    if global_config.is_eval:
        eval_dataset = build_dataset(cfg.data.val)
        eval_loader = build_dataloader(eval_dataset, data=cfg.data)
        # build postprocess
        postprocess = build_postprocess(cfg.postprocess)
        # build metric
        metric = build_metrics(cfg.metric)
    else:
        eval_loader = None
        postprocess = None
        metric = None

    # build optimizer
    optimizer = build_optimizer(cfg.optimizer, model)
    # build lr_scheduler
    lr_scheduler = build_lr_scheduler(cfg.lr_scheduler, optimizer)
    # build loss
    criterion = build_loss(cfg.loss).to(device)

    runner = TrainRunner(global_config, model, optimizer, lr_scheduler, postprocess, criterion, train_loader,
                         eval_loader, metric, logger)

    runner.run()


if __name__ == '__main__':
    main()
