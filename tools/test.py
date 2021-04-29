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
from torchocr.runners.test_runner import eval, engine_eval
from torchocr.models import build_model
from torchocr.datasets import build_dataset, build_dataloader
from torchocr.metrics import build_metrics
from torchocr.postprocess import build_postprocess
from torchocr.utils.checkpoints import load_checkpoint, save_checkpoint
from tools.deploy.trt_inference import TRTModel

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='OCR train')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--model_path', required=True, type=str, help='rec model path')
    parser.add_argument('--simple', action='store_true', help='rm optimizer in model')
    parser.add_argument('--mode', type=str, default='torch', help='run mode : torch/onnx/engine')
    parser.add_argument('--engine_path', type=str, default=None, help='rec engine path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    # set pretrained model None
    cfg.model.pretrained = None

    # build postprocess
    postprocess = build_postprocess(cfg.postprocess)
    # for rec cal head number
    if hasattr(postprocess, 'character'):
        char_num = len(getattr(postprocess, 'character'))
        cfg.model.head.n_class = char_num

    eval_dataset = build_dataset(cfg.test_data.dataset)
    eval_loader = build_dataloader(eval_dataset, loader_cfg=cfg.test_data.loader)
    # build metric
    metric = build_metrics(cfg.metric)

    mode = args.mode
    if mode == 'torch':
        # build model
        model = build_model(cfg.model)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        load_checkpoint(model, args.model_path, map_location=device)
        if args.simple:
            (filepath, filename) = os.path.split(args.model_path)
            simple_model_path = os.path.join(filepath, 'sim_{}'.format(filename))
            save_checkpoint(model, simple_model_path)

        model = model.to(device)
        model.device = device
        result_metirc = eval(model, eval_loader, postprocess, metric)

    elif mode == 'engine':

        engine_path = args.engine_path
        model = TRTModel(engine_path)
        result_metirc = engine_eval(model, eval_loader, postprocess, metric)

    print(result_metirc)


if __name__ == '__main__':
    main()
