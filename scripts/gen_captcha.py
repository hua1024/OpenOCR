# coding=utf-8  
# @Time   : 2020/8/19 9:30
# @Auto   : zzf-jeff


# -*- coding: UTF-8 -*-
from captcha.image import ImageCaptcha  # pip install captcha
from PIL import Image
import random
import time
import os
import argparse
import sys

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

# 生成验证码信息
ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

TRAIN_DATASET_PATH = os.path.join('/zzf/data/captcha/data', 'train')
TEST_DATASET_PATH = os.path.join('/zzf/data/captcha/data', 'test')
PREDICT_DATASET_PATH = os.path.join('/zzf/data/captcha/data', 'pred')


def random_captcha():
    captcha_text = []
    for i in range(MAX_CAPTCHA):
        c = random.choice(ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image


if __name__ == '__main__':
    # data-model : 选择生成验证码的用途，包括train,test,pred
    # data-number : 生成验证码的数目
    parser = argparse.ArgumentParser(description="gen captcha data")
    parser.add_argument('--data-model', type=str, choices=['train', 'test', 'pred'], default="test")
    parser.add_argument('--data-number', type=int, default=30000)
    args = parser.parse_args()

    if args.data_model == 'train':
        path = TRAIN_DATASET_PATH
    elif args.data_model == 'test':
        path = TEST_DATASET_PATH
    else:
        path = PREDICT_DATASET_PATH
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(args.data_number):
        now = str(int(time.time()))
        text, image = gen_captcha_text_and_image()
        filename = text + '_' + now + '.jpg'
        image.save(path + os.path.sep + filename)
        print('saved %d : %s' % (i + 1, filename))
