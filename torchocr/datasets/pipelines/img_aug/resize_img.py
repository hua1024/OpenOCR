# coding=utf-8  
# @Time   : 2021/1/7 10:11
# @Auto   : zzf-jeff
from torchocr.datasets.pipelines.compose import PIPELINES
from torchvision import transforms
import cv2
import numpy as np
import math
import sys


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


def resize_norm_img(img, image_shape):
    """ resize img and padding Normalize
    根据image_shape resize图片，不够的地方padding 0
    :param img:
    :param image_shape:
    :return:
    """
    # todo:之后接的Normalize操作是否有必要解耦
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def resize_norm_img_chinese(img, image_shape):
    # todo: chinese ?
    imgC, imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    max_wh_ratio = 0
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(32 * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


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
        img = data['image']
        if self.infer_mode and self.character_type == "ch":
            norm_img = resize_norm_img_chinese(img, self.image_shape)
        else:
            norm_img = resize_norm_img(img, self.image_shape)
        data['image'] = norm_img
        return data


## todo: 如何兼容多种检测不同的resize操作
@PIPELINES.register_module()
class DetResizeForTest(object):
    def __init__(self, image_shape, **kwargs):
        self.image_shape = image_shape

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape
        img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img, short_size=736):
        height, width, _ = img.shape
        if height < width:
            new_height = short_size
            new_width = new_height / height * width
        else:
            new_width = short_size
            new_height = new_width / width * height
        new_height = int(round(new_height / 32) * 32)
        new_width = int(round(new_width / 32) * 32)

        ratio_h = float(new_height) / height
        ratio_w = float(new_width) / width

        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img, [ratio_h, ratio_w]
