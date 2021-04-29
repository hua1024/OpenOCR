# coding=utf-8  
# @Time   : 2020/12/28 16:10
# @Auto   : zzf-jeff

import os
import sys
import torch
import numpy as np
import glob
from tqdm import tqdm
import shutil
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from torchocr.utils.config_util import Config
from torchocr.postprocess import build_postprocess
from torchocr.models import build_model
from torchocr.utils.checkpoints import load_checkpoint
from torchocr.datasets.pipelines.transforms import NormalizeImage, ToCHWImage
from torchocr.datasets.pipelines.img_aug.det_resize_img import DetResizeForTest
from tools.deploy.onnx_inference import ONNXModel
from tools.deploy.trt_inference import TRTModel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class DetInfer:
    def __init__(self, cfg, args):

        self.mode = args.mode
        self.postprocess = build_postprocess(cfg.postprocess)
        if self.mode == 'torch':
            model_path = args.model_path
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

        self.normalize = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_chw = ToCHWImage()
        self.resize = DetResizeForTest(short_size=736, mode='db')

    def predict(self, img):
        # 预处理根据训练来
        data = {'image': img}
        data = self.resize(data)
        data = self.normalize(data)
        data = self.to_chw(data)

        if self.mode == 'torch':
            image = np.expand_dims(data['image'], axis=0)
            input_data = torch.from_numpy(image)
            input_data = input_data.to(self.device)
            with torch.no_grad():
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


        shape_list = np.expand_dims(data['shape'], axis=0)
        post_result = self.postprocess(out, shape_list)
        if post_result:
            boxes = post_result[0]['points']
            scores = post_result[0]['scores']
            if len(boxes) > 0:
                box_list, score_list = boxes, scores
            else:
                box_list, score_list = [], []
            return box_list, score_list
        return [], []


def load_data(path):
    if os.path.isfile(path):
        img_list = [path]
    else:
        img_list = glob.glob(os.path.join(path, '*.png'))
    return img_list


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='det infer')
    parser.add_argument('--config', required=True, type=str, help='config of model')
    parser.add_argument('--model_path', required=True, type=str, help='rec model path')
    parser.add_argument('--img_path', required=True, type=str, help='img path for predict')
    parser.add_argument('--mode', required=False, type=str, default='torch', help='run mode : torch/onnx/engine')
    parser.add_argument('--onnx_path', required=False, type=str, default=None, help='rec onnx path')
    parser.add_argument('--engine_path', required=False, type=str, default=None, help='rec engine path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt
    from torchocr.utils.vis import draw_bbox

    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)
    # set pretrained model None
    cfg.model.pretrained = None
    # 通用配置
    model = DetInfer(cfg, args)

    img_list = load_data(args.img_path)

    if not os.path.exists('output'):
        os.makedirs('output')

    start_time = time.time()
    for file in tqdm(img_list):
        ori_img = cv2.imread(file)
        base_name = os.path.basename(file)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        try:

            box_list, score_list = model.predict(img)
            rec_path = os.path.join('output', base_name)
            print(score_list)
            if len(box_list) > 0:
                res_img = draw_bbox(ori_img, box_list)
                cv2.imwrite(rec_path, res_img)
            else:
                print(file)
                shutil.copy(file, rec_path)
        except Exception as e:
            print(e)
            print(file)

    end_time = time.time()
    print(end_time-start_time)
