# coding=utf-8  
# @Time   : 2021/1/7 10:11
# @Auto   : zzf-jeff

from torchocr.datasets.pipelines.compose import PIPELINES
import cv2
import numpy as np
import math
import sys
import os


def pad_img(image, top, bottom, left, right, color):
    padded_image = cv2.copyMakeBorder(image, top, bottom,
                                      left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image


def normalize(image, resize_image_shape):
    resized_image = image.astype('float32')

    if resize_image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255

    resized_image -= 0.5
    resized_image /= 0.5

    return resized_image


def resize_img(img, image_shape):
    H, W = image_shape[1:]
    h, w = img.shape[:2]
    new_w = int((float(H) / h) * w)
    if (new_w > W):
        resized_image = cv2.resize(img, (W, H))
    else:
        img = cv2.resize(img, (new_w, H))
        resized_image = pad_img(img, 0, 0, 0, W - new_w, color=(0, 0, 0))
    resized_image = normalize(resized_image, image_shape)

    return resized_image


@PIPELINES.register_module()
class RecResizeImg(object):
    def __init__(self,
                 image_shape,
                 infer_mode=False,
                 character_type='ch',
                 **kwargs):
        """rec img resize -->32,100

        :param image_shape:
        :param infer_mode:
        :param character_type:
        :param kwargs:
        """
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_type = character_type

    def __call__(self, data):
        try:
            if data is None:
                return None
            img = data['image']
            if self.infer_mode and self.character_type == "ch":
                norm_img = resize_img(img, self.image_shape)
            else:
                norm_img = resize_img(img, self.image_shape)
            data['image'] = norm_img
            return data
        except Exception as e:
            file_name = os.path.basename(__file__).split(".")[0]
            print('{} --> '.format(file_name), e)
            return None
