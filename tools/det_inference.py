# coding=utf-8  
# @Time   : 2020/12/28 16:10
# @Auto   : zzf-jeff

import os
import sys
import torch
from torchvision import transforms
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from torchocr.utils.config_util import Config
from torchocr.postprocess import build_postprocess
from torchocr.models import build_model
from torchocr.utils.checkpoints import load_checkpoint
from torchocr.datasets.pipelines.transforms import NormalizeImage, ToCHWImage
from torchocr.datasets.pipelines.img_aug.resize_img import DetResizeForTest

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class DetInfer:
    def __init__(self, cfg, model_path):

        self.model = build_model(cfg.model)
        load_checkpoint(self.model, model_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.postprocess = build_postprocess(cfg.postprocess)
        self.normalize = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_chw = ToCHWImage()
        self.resize = DetResizeForTest([736, 1280])

    def predict(self, img, is_output_polygon=False):
        # 预处理根据训练来
        data = {'image': img}
        data = self.resize(data)
        data = self.normalize(data)
        data = self.to_chw(data)
        image = np.expand_dims(data['image'], axis=0)
        shape_list = np.expand_dims(data['shape'], axis=0)
        tensor = torch.from_numpy(image)
        tensor = tensor.to(self.device)
        out = self.model(tensor)
        post_result = self.postprocess(out, shape_list)
        boxes = post_result[0]['points']
        scores = post_result[0]['scores']
        print(boxes)
        if len(boxes) > 0:
            print(boxes,scores)
            box_list, score_list = boxes,scores
        else:
            box_list, score_list = [], []
        return box_list, score_list


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='det infer')
    parser.add_argument('--config', required=True, type=str, help='config of model')
    parser.add_argument('--model_path', required=True, type=str, help='rec model path')
    parser.add_argument('--img_path', required=True, type=str, help='img path for predict')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt
    from torchocr.utils.vis import draw_bbox

    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)
    # 通用配置
    global_config = cfg.options

    img = cv2.imread(args.img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model = DetInfer(cfg, args.model_path)
    box_list, score_list = model.predict(img, is_output_polygon=False)
    img = draw_bbox(img, box_list)
    plt.imshow(img)
    plt.savefig('test.png')
