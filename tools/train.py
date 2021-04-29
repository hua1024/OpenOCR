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
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from torchocr.utils.config_util import Config
from torchocr.utils import file_util
from torchocr.utils.logger import get_logger
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
from torchocr.utils.dist_utils import init_dist


def parse_args():
    parser = argparse.ArgumentParser(description='OCR train')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--resume', action='store_true', help='resume model')
    parser.add_argument('--distributed', action='store_true', help='using DDP')
    parser.add_argument('--amp', action='store_true', help='using apex')
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--ema', action='store_true', help='using EMA')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    global_config = cfg.options  # 通用配置
    # local_rank = 0 is logger
    global_config['local_rank'] = args.local_rank
    # amp train
    if args.amp:
        global_config['is_amp'] = True
    else:
        global_config['is_amp'] = False

    # ema train
    if args.ema:
        global_config['is_ema'] = True
    else:
        global_config['is_ema'] = False

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

    # # log env info
    if args.local_rank == 0:
        env_info_dict = collect_env()
        env_info = '\n'.join([('{}: {}'.format(k, v))
                              for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
        ## log some basic info
        logger.info('Config:\n{}'.format(cfg.text))
        # set random seeds
        logger.info('Set random seed to {}, deterministic: {}'.format(global_config.seed, args.deterministic))

    # set random seed
    set_random_seed(global_config.seed, deterministic=args.deterministic)

    # select device
    # dist init
    if torch.cuda.device_count() > 1 and args.distributed:
        device = init_dist(launcher='pytorch', backend='nccl', rank=args.local_rank)
        global_config['distributed'] = True
    else:
        device, gpu_ids = select_device(global_config.gpu_ids)
        global_config.gpu_ids = gpu_ids
        global_config['distributed'] = False

    # build train dataset
    train_dataset = build_dataset(cfg.train_data.dataset)
    train_loader = build_dataloader(train_dataset, loader_cfg=cfg.train_data.loader,
                                    distributed=global_config['distributed'])

    # if is eval , build eval dataloader,postprocess,metric
    # 移动到前面，由于rec-head的输出需要用postprocess计算
    if global_config.is_eval:
        eval_dataset = build_dataset(cfg.test_data.dataset)
        eval_loader = build_dataloader(eval_dataset, loader_cfg=cfg.test_data.loader,
                                       distributed=global_config['distributed'])
        # build postprocess
        postprocess = build_postprocess(cfg.postprocess)
        # build metric
        metric = build_metrics(cfg.metric)
    else:
        eval_loader = None
        postprocess = None
        metric = None

    # for rec cal head number
    if hasattr(postprocess, 'character'):
        char_num = len(getattr(postprocess, 'character'))
        cfg.model.head.n_class = char_num

    # build model
    model = build_model(cfg.model)
    model = model.to(device)

    # set model to device
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and global_config['distributed'] == True:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        device = torch.device('cuda', args.local_rank)
        is_cuda = True
    elif device.type != 'cpu' and global_config['distributed'] == False and len(gpu_ids) >= 1:
        model = nn.DataParallel(model, device_ids=global_config.gpu_ids)
        model.gpu_ids = gpu_ids
        is_cuda = True
    else:
        is_cuda = False

    global_config['is_cuda'] = is_cuda

    model.device = device

    # build optimizer
    optimizer = build_optimizer(cfg.optimizer, model)
    # build lr_scheduler
    lr_scheduler = build_lr_scheduler(cfg.lr_scheduler, optimizer)
    # build loss
    criterion = build_loss(cfg.loss).to(device)

    runner = TrainRunner(global_config, model, optimizer, lr_scheduler, postprocess, criterion, train_loader,
                         eval_loader, metric, logger)

    # # Resume
    if global_config.resume_from is not None and args.resume:
        runner.resume(global_config.resume_from, map_location=device)

    if global_config.load_from is not None:
        runner.load_checkpoint(global_config.load_from, map_location=device)

    runner.run()


if __name__ == '__main__':
    main()
