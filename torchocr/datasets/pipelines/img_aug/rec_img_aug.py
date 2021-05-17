# coding=utf-8  
# @Time   : 2021/1/27 14:47
# @Auto   : zzf-jeff

'''
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/ppocr/data/imaug/rec_img_aug.py
'''

from torchocr.datasets.builder import PIPELINES
from .tia_augment import *
from .rec_common_augment import *

import random


@PIPELINES.register_module()
class RecAug(object):
    def __init__(self, aug_prob=0.5, use_tia=True, **kwargs):
        self.aug_prob = aug_prob
        self.use_tia = use_tia

    def __call__(self, data):
        img = data['image']
        img = warp(img, 10, self.use_tia, self.aug_prob)
        data['image'] = img
        return data


class Config:
    """
    Config
    """

    def __init__(self, use_tia):
        self.anglex = random.random() * 30
        self.angley = random.random() * 15
        self.anglez = random.random() * 10
        self.fov = 42
        self.r = 0
        self.shearx = random.random() * 0.3
        self.sheary = random.random() * 0.05
        self.borderMode = cv2.BORDER_REPLICATE
        self.use_tia = use_tia

    def make(self, w, h, ang):
        """
        make
        """
        self.anglex = random.random() * 5 * flag()
        self.angley = random.random() * 5 * flag()
        self.anglez = -1 * random.random() * int(ang) * flag()
        self.fov = 42
        self.r = 0
        self.shearx = 0
        self.sheary = 0
        self.borderMode = cv2.BORDER_REPLICATE
        self.w = w
        self.h = h

        self.perspective = self.use_tia
        self.stretch = self.use_tia
        self.distort = self.use_tia

        self.crop = True
        self.affine = False
        self.reverse = True
        self.noise = True
        self.jitter = True
        self.blur = True
        self.color = True


def warp(img, ang, use_tia=True, prob=0.4):
    """
    warp
    """
    h, w, _ = img.shape
    config = Config(use_tia=use_tia)
    config.make(w, h, ang)
    new_img = img

    if config.distort:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = tia_distort(new_img, random.randint(3, 6))

    if config.stretch:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = tia_stretch(new_img, random.randint(3, 6))

    if config.perspective:
        if random.random() <= prob:
            new_img = tia_perspective(new_img)

    if config.crop:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            new_img = random_top_crop(new_img)

    if config.blur:
        if random.random() <= prob:
            new_img = gauss_blur(new_img)

    if config.color:
        if random.random() <= prob:
            new_img = random_hsv(new_img)

    if config.noise:
        if random.random() <= prob:
            new_img = add_gauss_noise(new_img)

    return new_img
