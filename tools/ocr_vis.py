# coding=utf-8  
# @Time   : 2021/5/11 11:54
# @Auto   : zzf-jeff

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
from tqdm import tqdm
import cv2
import numpy as np
from PIL import ImageFont


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from torchocr.utils.config_util import Config
from tools.det_inference import DetInfer
from tools.rec_inference import RecInfer

from torchocr.utils.vis import draw_img_boxes, vis_ocr_result

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def load_data(path):
    import glob
    if os.path.isfile(path):
        img_list = [path]
    else:
        img_list = glob.glob(os.path.join(path, '*.png'))
    return img_list


def transform_polys_to_bboxs(polygons):
    bboxs = []
    for polygon in polygons:
        poly = np.array(polygon).astype(np.int).reshape(-1, 2)
        xmin = int(np.min(poly[:, 0]))
        xmax = int(np.max(poly[:, 0]))
        ymin = int(np.min(poly[:, 1]))
        ymax = int(np.max(poly[:, 1]))
        bboxs.append([xmin, ymin, xmax, ymax])

    return bboxs


def main():
    args = parse_args()

    font_ttf = "test/STKAITI.TTF"  # 可视化字体类型
    font = ImageFont.truetype(font_ttf, 20)  # 字体与字体大小

    output = args.output
    det_cfg = Config.fromfile(args.det_config)
    rec_cfg = Config.fromfile(args.rec_config)
    det_cfg.model.pretrained = None
    det_cfg.model.pretrained = None

    rec_model = RecInfer(rec_cfg, args.rec_weights)
    det_model = DetInfer(det_cfg, args.det_weights)



    if not os.path.exists(output):
        os.makedirs(output)

    img_list = load_data(args.img_path)

    for file in tqdm(img_list):

        ori_img = cv2.imread(file)
        base_name = os.path.basename(file)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        boxes_list, scores_list = det_model.predict(img)

        boxes_list = transform_polys_to_bboxs(boxes_list)

        output_path = os.path.join(output, base_name)

        res_img = ori_img.copy()

        ocr_dst = []

        for bbox in boxes_list:
            x1, y1, x2, y2 = bbox
            crop_img = ori_img[y1:y2, x1:x2]
            # one line
            rec_dst = rec_model.predict(crop_img)[0]
            pred_str, pred_score, pred_score_char = rec_dst
            ocr_dst.append((bbox, pred_str))

        res_img = vis_ocr_result(res_img, ocr_dst, font=font)

        cv2.imwrite(output_path, res_img)


def parse_args():
    parser = argparse.ArgumentParser(description='OCR train')
    parser.add_argument('--det_config', type=str, help='ocr det config path')
    parser.add_argument('--rec_config', type=str, help='ocr rec config path')
    parser.add_argument('--det_weights', type=str, help='ocr det weights path')
    parser.add_argument('--rec_weights', type=str, help='ocr rec weights path')
    parser.add_argument('--img_path', type=str, help='img path for predict')
    parser.add_argument('--output', type=str, default='ocr_output', help='infer result vis')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
