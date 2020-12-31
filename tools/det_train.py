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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), './'))

from torchocr.utils.config import Config
from torchocr.runners.dist_utils import get_dist_info, init_dist
from torchocr.utils import files
from torchocr.utils.logger import get_logger
from torchocr.utils.torch_utils import set_random_seed,select_device
from torchocr.models import build_det
from torchocr.datasets import build_det_dataset
from torchocr.apis.det_train import train_detector


def parse_args():
    parser = argparse.ArgumentParser(description='OCR det train')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')

    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--mode', type=str, help='detect or recognition')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    # set cudnn_benchmark,如数据size一致能加快训练
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    files.mkdir_or_exist(cfg.work_dir)
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_logger(name='ocr', log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    meta['env_info'] = 'test'
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('Config:\n{}'.format(cfg.pretty_text))

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_det(cfg.model)

    device = select_device(cfg.gpu_ids)

    model = model.cuda(device)
    # model.device = device
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.device = device
    datasets = [build_det_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_det_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(config=cfg.text,CLASSES=datasets[0].CLASSES)
    model.CLASSES = datasets[0].CLASSES
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    train_detector(model,datasets,cfg,validate=args.validate,timestamp=timestamp,meta=meta)


if __name__ == '__main__':
    main()
