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

# from tools.deploy.onnx_inference import ONNXModel
# from tools.deploy.trt_inference import TRTModel

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


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


def load_data_by_txt(path):
    import json

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
        infos = read_txt(ann_file, split_type=",")
        data_infos = []
        for (img_path, gt_path) in tqdm(infos):
            labels, texts = get_bboxs(gt_path)
            data_infos.append({'img_path': img_path, 'label': labels, 'text': texts})
        return data_infos

    def expand_points_num(boxes):
        # 对于部分多边形多点的，要expand点到shape一致
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)

        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes

    def get_bboxs(gt_path):
        labels = []
        texts = []
        with open(gt_path, 'r', encoding='utf-8') as file:
            instances = json.loads(file.read())
            for instance in instances:
                pts = instance['points']
                pts = pts.split(',')
                texts.append(['ocr'])
                labels.append(pts)

            new_labels = []
            for i, point in enumerate(labels):
                box = [[int(point[i]), int(point[i + 1])] for i in range(0, len(point), 2)]
                new_labels.append(box)

            new_labels = expand_points_num(new_labels)
            new_labels = np.array(new_labels, dtype='float32')
        return new_labels, texts

    data_infos = []
    infos = read_txt(path, split_type=',')
    for info in infos:
        txt_file, use_flag = info
        if int(use_flag) == 1:
            data_infos += load_annotations(txt_file)
    return data_infos


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='det infer')
    parser.add_argument('--config', required=True, type=str, help='config of model')
    parser.add_argument('--model_path', required=True, type=str, help='rec model path')
    parser.add_argument('--img_path', required=True, type=str, help='img path for predict')
    parser.add_argument('--mode', required=False, type=str, default='torch', help='run mode : torch/onnx/engine')
    parser.add_argument('--onnx_path', required=False, type=str, default=None, help='rec onnx path')
    parser.add_argument('--engine_path', required=False, type=str, default=None, help='rec engine path')
    parser.add_argument('--output', type=str, default='det_output', help='infer result vis')
    args = parser.parse_args()
    return args


def test_vis():
    import cv2
    from matplotlib import pyplot as plt
    from torchocr.utils.vis import draw_bbox

    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)
    cfg.model.pretrained = None
    # 通用配置
    model = DetInfer(cfg, args)

    data_infos = load_data_by_txt(args.img_path)
    output = args.output

    if not os.path.exists(output):
        os.makedirs(output)

    for idx, data in enumerate(tqdm(data_infos)):
        file = data['img_path']
        label = data['label']
        ori_img = cv2.imread(file)
        base_name = os.path.basename(file)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        # try:
        box_list, score_list = model.predict(img)
        rec_path = os.path.join(output, base_name)

        if len(box_list) > 0:
            res_img = draw_bbox(ori_img, box_list, color=(0, 0, 255), thickness=2)
            res_img = draw_bbox(res_img, label, color=(255, 0, 0), thickness=3)
        else:
            res_img = draw_bbox(ori_img, label, color=(255, 0, 0), thickness=3)

        cv2.imwrite(rec_path, res_img)
        # except Exception as e:
        #     print('{} error, is {}'.format(file, e))


def main():
    import cv2
    from matplotlib import pyplot as plt
    from torchocr.utils.vis import draw_bbox

    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)
    cfg.model.pretrained = None
    # 通用配置
    model = DetInfer(cfg, args)

    img_list = load_data(args.img_path)

    output = args.output

    if not os.path.exists(output):
        os.makedirs(output)

    for file in tqdm(img_list):
        ori_img = cv2.imread(file)
        base_name = os.path.basename(file)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        # try:
        box_list, score_list = model.predict(img)
        rec_path = os.path.join(output, base_name)
        if len(box_list) > 0:
            res_img = draw_bbox(ori_img, box_list, color=(0, 0, 255), thickness=2)
            cv2.imwrite(rec_path, res_img)
        else:
            shutil.copy(file, rec_path)
        # except Exception as e:
        #     print('{} error, is {}'.format(file, e))


if __name__ == '__main__':
    # main()
    test_vis()
