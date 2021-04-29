# coding=utf-8  
# @Time   : 2021/1/7 10:11
# @Auto   : zzf-jeff
from torchocr.datasets.pipelines.compose import PIPELINES
from torchvision import transforms
import cv2
import numpy as np
import math
import sys
import os


@PIPELINES.register_module()
class Resize(object):
    """Image Resize --> 图片resize操作
    Args:
        size : resize后的尺寸
        image ：原始图片
    Returns:
        result :  Over Resize
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        transform = transforms.Resize(self.size)
        data['image'] = transform(data['image'])
        return data


## todo: 如何兼容多种检测不同的resize操作
@PIPELINES.register_module()
class DetResizeForTest(object):
    def __init__(self, mode, **kwargs):
        self.mode = mode
        if self.mode == 'db':
            self.short_size = kwargs['short_size']

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape
        if self.mode == 'db':
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        else:
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type1(self, img):
        resize_h, resize_w = (736, 1280)
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]

    def resize_image_db1(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.short_size
            new_width = new_height / height * width
        else:
            new_width = self.short_size
            new_height = new_width / width * height

        new_height = int(round(new_height / 32) * 32)
        new_width = int(round(new_width / 32) * 32)

        ratio_h = float(new_height) / height
        ratio_w = float(new_width) / width

        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img, [ratio_h, ratio_w]



    def resize_image_db(self, img):
        height, width, _ = img.shape
        if height > self.short_size:
            new_height = 1280
            new_width = 1280
        else:
            new_height = int(math.ceil(height / 32) * 32)
            new_width = int(math.ceil(new_height / height * width / 32) * 32)

        ratio_h = float(new_height) / height
        ratio_w = float(new_width) / width

        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img, [ratio_h, ratio_w]
