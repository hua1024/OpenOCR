# coding=utf-8  
# @Time   : 2021/1/12 10:12
# @Auto   : zzf-jeff


import numpy as np
import cv2
import torch
import pyclipper
from torchocr.datasets.builder import PIPELINES


@PIPELINES.register_module()
class PSEProcessTrain():
    def __init__(self, img_size=640, n=6, m=0.5, **kwargs):
        self.img_size = img_size
        self.n = n
        self.m = m

    def generate_map(self, im_size, text_polys, text_tags, training_mask, i, n, m):
        """gen pse need map
        生成shrink map
        :param text_polys:
        :param text_tags:
        :param training_mask:
        :param i:
        :param n:
        :param m:
        :return:
        """
        h, w = im_size
        score_map = np.zeros((h, w), dtype=np.uint8)
        for poly, tag in zip(text_polys, text_tags):
            poly = poly.astype(np.int)
            # r
            r_i = 1 - (1 - m) * (n - i) / (n - 1)
            # d
            d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, closed=True)
            # 采用pyclipper直接求shrink
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked_poly = np.array(pco.Execute(-d_i))
            # draw score_map one
            cv2.fillPoly(score_map, shrinked_poly, 1)
            # ignore draw zero
            if tag:
                cv2.fillPoly(training_mask, shrinked_poly, 0)

        return score_map, training_mask

    def __call__(self, data):
        img = data['image']
        text_polys = data['polys']
        text_tags = data['ignore_tags']

        # resize的方式还是有点缺陷的，应该crop
        # h, w = img.shape[:2]
        # short_edge = min(h, w)
        # if short_edge < self.img_size:
        #     # 保证短边 >= inputsize
        #     scale = self.img_size / short_edge
        #     img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        #     text_polys *= scale

        h, w = img.shape[:2]
        training_mask = np.ones((h, w), dtype=np.uint8)
        score_maps = []
        for i in range(1, self.n + 1):
            # s1-->sn ,从小到大
            score_map, training_mask = self.generate_map(
                (h, w), text_polys, text_tags, training_mask, i, self.n, self.m)
            score_maps.append(score_map)

        score_maps = np.array(score_maps, dtype=np.float32)
        gt_texts = score_maps[-1, :, :]
        gt_kernels = score_maps[:-1, :, :]

        data['gt_texts'] = torch.from_numpy(gt_texts)
        data['gt_kernels'] = torch.from_numpy(gt_kernels)
        data['training_masks'] = torch.from_numpy(training_mask)

        return data
