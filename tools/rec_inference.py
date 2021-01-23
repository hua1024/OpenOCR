# coding=utf-8
# @Time   : 2020/12/28 16:10
# @Auto   : zzf-jeff


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
from torchocr.datasets.pipelines.img_aug.resize_img import RecResizeImg

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class RecInfer(object):
    def __init__(self, cfg, model_path):
        self.model = build_model(cfg.model)
        load_checkpoint(self.model, model_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.postprocess = build_postprocess(cfg.postprocess)

        self.resize = RecResizeImg(image_shape=[3, 32, 100], infer_mode=False, character_type='ch')


    def predict(self, img):
        # 预处理根据训练来
        data = {'image': img}
        data = self.resize(data)
        image = np.expand_dims(data['image'], axis=0)
        tensor = torch.from_numpy(image)
        tensor = tensor.to(self.device)
        out = self.model(tensor)
        pred_result = self.postprocess(out)
        return pred_result


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

    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)
    # 通用配置
    global_config = cfg.options

    img = cv2.imread(args.img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model = RecInfer(cfg, args.model_path)
    rec_reuslt = model.predict(img)

    print(rec_reuslt)
