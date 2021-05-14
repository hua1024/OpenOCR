# coding=utf-8
# @Time   : 2020/12/28 16:10
# @Auto   : zzf-jeff
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import torch
import time
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from torchocr.utils.config_util import Config
from torchocr.postprocess import build_postprocess
from torchocr.models import build_model
from torchocr.utils.checkpoints import load_checkpoint
from torchocr.datasets.pipelines.img_aug.rec_resize_img import RecResizeImg

# from tools.deploy.onnx_inference import ONNXModel
# from tools.deploy.trt_inference import TRTModel

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


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
            out = self.model.run(input_data)
            out = torch.Tensor(out)

        pred_result = self.postprocess(out)
        return pred_result


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='rec infer')
    parser.add_argument('--config', type=str, help='config of model')
    parser.add_argument('--model_path', type=str, help='rec model path')
    parser.add_argument('--mode', type=str, default='torch', help='run mode : torch/onnx/engine')
    parser.add_argument('--onnx_path', type=str, default=None, help='rec onnx path')
    parser.add_argument('--engine_path', type=str, default=None, help='rec engine path')
    parser.add_argument('--img_path', type=str, help='img path for predict')
    parser.add_argument('--output', type=str, default='rec_output', help='infer result vis')
    args = parser.parse_args()
    return args


def load_data(path):
    import glob
    if os.path.isfile(path):
        img_list = [path]
    else:
        img_list = glob.glob(os.path.join(path, '*.png'))
    return img_list


def load_data_by_txt(path):
    def read_txt(txt_path, split_type):
        '''
        读取txt文件的标注信息，格式为
        xxx/a/1.png,a
        xxx/a/2.png,a
        Args:
            txt_path: train/valid/test data txt or json
        Returns:
            imgs：list, all data info
        '''
        with open(txt_path, 'r', encoding='utf-8') as f:
            infos = list(map(lambda line: line.strip().split(split_type, 1), f))
        return infos

    def load_annotations(ann_file):
        infos = read_txt(ann_file, split_type=' ')
        data_infos = []
        for info in infos:
            if len(info) != 2:
                continue
            img_path, label = info[0], info[1]
            data_infos.append({'img_path': img_path, 'label': label})
        return data_infos

    data_infos = []
    infos = read_txt(path, split_type=' ')
    for info in infos:
        txt_file, use_flag = info
        if int(use_flag) == 1:
            data_infos += load_annotations(txt_file)
    return data_infos


def img_add_text(img, font, text, left, top, color):
    text = " " if text == "" else text
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text((left, top), text, color, font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def visualize_one_line(img, pred, label, font):
    img_dst = np.ones((img.shape[0] + 120, img.shape[1] + 100, img.shape[2]), dtype=np.uint8) * 255
    img_dst[:img.shape[0], :img.shape[1], :] = img
    img_dst = img_add_text(img_dst, font, " pred ({}) :{}".format(len(pred), pred), 0, img.shape[0],
                           color=(0, 255, 0))
    if label is not None:
        img_dst = img_add_text(img_dst, font, "label ({}) :{}".format(len(label), label), 0, img.shape[0] + 60,
                               color=(255, 0, 0))
    return img_dst


def visual_result(img, pred, label, font, *args, **kwargs):
    img_dst = visualize_one_line(img, pred, label, font)
    return img_dst


def test_vis():
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)
    model = RecInfer(cfg, args)
    output = args.output
    font_ttf = "test/STKAITI.TTF"  # 可视化字体类型
    font = ImageFont.truetype(font_ttf, 20)  # 字体与字体大小
    data_infos = load_data_by_txt(args.img_path)

    if not os.path.exists(output):
        os.makedirs(output)

    true_path = os.path.join(output, 'true_img')
    false_path = os.path.join(output, 'error_img')
    if not os.path.exists(true_path):
        os.makedirs(true_path)
    if not os.path.exists(false_path):
        os.makedirs(false_path)

    for idx, data in enumerate(tqdm(data_infos)):
        file = data['img_path']
        label = data['label']
        img = cv2.imread(file)
        base_name = os.path.basename(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rec_result = model.predict(img)
        rec_str, rec_score, rec_score_list = rec_result[0]

        img_dst = visual_result(img, rec_str, label=label, font=font)

        if label == rec_str:
            out_path = os.path.join(true_path, base_name)
        else:
            out_path = os.path.join(false_path, base_name)
            with open(os.path.join(output, 'error.txt'), 'a+', encoding='utf-8') as fw:
                fw.write(file + '\n')
        cv2.imwrite(out_path, img_dst)


def main():
    args = parse_args()
    cfg_path = args.config
    output = args.output
    cfg = Config.fromfile(cfg_path)
    cfg.model.pretrained = None
    model = RecInfer(cfg, args)
    img_list = load_data(args.img_path)
    font_ttf = "test/STKAITI.TTF"  # 可视化字体类型
    font = ImageFont.truetype(font_ttf, 20)  # 字体与字体大小

    output = args.output

    if not os.path.exists(output):
        os.makedirs(output)

    for file in img_list:
        img = cv2.imread(file)
        base_name = os.path.basename(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rec_result = model.predict(img)
        rec_str, rec_score, rec_score_list = rec_result[0]
        out_path = os.path.join(output, base_name)
        img_dst = visual_result(img, rec_str, label=None, font=font)
        cv2.imwrite(out_path, img_dst)


if __name__ == '__main__':
    # main()
    test_vis()
