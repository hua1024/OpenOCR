# coding=utf-8  
# @Time   : 2021/5/15 18:36
# @Auto   : zzf-jeff

'''
text recognition base augment func
'''
import cv2
import random
import numpy as np


def flag():
    """
    flag
    """
    return 1 if random.random() > 0.5000001 else -1


def gauss_blur(img):
    '''高斯模糊

    :param img:
    :return:
    '''
    h, w = img.shape[:2]
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img


def random_top_crop(img):
    """随机上下裁剪1-5个像素

    :param img:
    :return:
    """
    h, w = img.shape[:2]
    top_min = 1
    top_max = 5
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = img.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]

    return crop_img


def random_hsv(img):
    """随机变换hsv

    :param img:
    :return:
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    delta = 0.001 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img


def add_gauss_noise(img, mean=0, var=0.1):
    """添加高斯噪声

    :param image:
    :param mean:
    :param var:
    :return:
    """
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    new_img = img + 0.5 * noise
    new_img = np.clip(new_img, 0, 255)
    new_img = np.uint8(new_img)
    return new_img
