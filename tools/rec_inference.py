# coding=utf-8
# @Time   : 2020/12/28 16:10
# @Auto   : zzf-jeff


# coding=utf-8
# @Time   : 2020/12/28 16:10
# @Auto   : zzf-jeff

import os
import sys
import torch
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from torchocr.utils.config_util import Config
from torchocr.postprocess import build_postprocess
from torchocr.models import build_model
from torchocr.utils.checkpoints import load_checkpoint
from torchocr.datasets.pipelines.img_aug.rec_resize_img import RecResizeImg
from tools.deploy.onnx_inference import ONNXModel
from tools.deploy.trt_inference import TRTModel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class RecInfer(object):
    def __init__(self, cfg, args):
        self.postprocess = build_postprocess(cfg.postprocess)
        self.mode = args.mode
        if self.mode == 'torch':
            model_path = args.model_path
            # for rec cal head number
            if hasattr(self.postprocess, 'character'):
                char_num = len(getattr(self.postprocess, 'character'))
                cfg.model.head.n_class = char_num
            self.model = build_model(cfg.model)
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            load_checkpoint(self.model, model_path, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()
        elif self.mode == 'onnx':
            onnx_path = args.onnx_path
            self.model = ONNXModel(onnx_path)
        elif self.mode == 'engine':
            engine_path = args.engine_path
            self.model = TRTModel(engine_path)

        # pre-heat
        self.resize = RecResizeImg(image_shape=[3, 32, 800], infer_mode=False, character_type='ch')

    def predict(self, img):
        # 预处理根据训练来
        data = {'image': img}
        data = self.resize(data)

        if self.mode == 'torch':
            image = np.expand_dims(data['image'], axis=0)
            input_data = torch.from_numpy(image)
            input_data = input_data.to(self.device)
            out = self.model(input_data)
        elif self.mode == 'onnx':
            image = np.expand_dims(data['image'], axis=0)
            input_data = np.array(image, dtype=np.float32, order='C')
            out = self.model.run(input_data)
            out = torch.Tensor(out)
        elif self.mode == 'engine':
            image = np.expand_dims(data['image'], axis=0)
            input_data = np.array(image, dtype=np.float32, order='C')
            # input_data = np.concatenate((input_data, input_data), axis=0)
            out = self.model.run(input_data)
            out = torch.Tensor(out)

        pred_result = self.postprocess(out)
        return pred_result


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='det infer')
    parser.add_argument('--config', required=True, type=str, help='config of model')
    parser.add_argument('--model_path', required=True, type=str, help='rec model path')
    parser.add_argument('--mode', required=False, type=str, default='torch', help='run mode : torch/onnx/engine')
    parser.add_argument('--onnx_path', required=False, type=str, default=None, help='rec onnx path')
    parser.add_argument('--engine_path', required=False, type=str, default=None, help='rec engine path')
    parser.add_argument('--img_path', required=True, type=str, help='img path for predict')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import cv2

    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)
    # 通用配置
    global_config = cfg.options

    img = cv2.imread(args.img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model = RecInfer(cfg, args)
    rec_reuslt = model.predict(img)
    print(rec_reuslt)
