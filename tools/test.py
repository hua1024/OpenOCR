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
from torchocr.runners.test_runner import eval
from torchocr.utils.torch_util import select_device
from torchocr.models import build_model
from torchocr.datasets import build_dataset, build_dataloader
from torchocr.metrics import build_metrics
from torchocr.postprocess import build_postprocess
from torchocr.utils.checkpoints import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='OCR train')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--model_path', required=True, type=str, help='rec model path')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    # 通用配置
    global_config = cfg.options

    # build model
    model = build_model(cfg.model)
    load_checkpoint(model, args.model_path)
    device = select_device(global_config.device)
    model = model.to(device)

    model.device = device

    eval_dataset = build_dataset(cfg.data.test)
    eval_loader = build_dataloader(eval_dataset, data=cfg.data)
    # build postprocess
    postprocess = build_postprocess(cfg.postprocess)
    # build metric
    metric = build_metrics(cfg.metric)

    result_metirc = eval(model, eval_loader, postprocess, metric)
    print(result_metirc)


if __name__ == '__main__':
    main()
