# coding=utf-8  
# @Time   : 2020/12/29 17:35
# @Auto   : zzf-jeff
from .compose import PIPELINES

import cv2
import numpy as np
from torchvision import transforms



@PIPELINES.register_module()
class Normalize():
    def __init__(self, mean, std, to_rgb=False):
        # self.mean = np.array(mean, dtype=np.float32)
        # self.std = np.array(std, dtype=np.float32)
        # self.to_rgb = to_rgb
        self.mean = mean
        self.std = std

    def __call__(self, datas):
        transform = transforms.Normalize(mean=self.mean, std=self.std)
        datas['img'] = transform(datas['img'])
        return datas


@PIPELINES.register_module()
class ToTensor(object):
    """Image ToTensor -->numpy2Tensor
    Args:
        image ：原始图片
    Returns:
        result :  Over ToTensor
    """

    def __call__(self, datas):
        transform = transforms.ToTensor()
        datas['img'] = transform(datas['img'])
        return datas

# @PIPELINES.register_module()
# class Fliplr(object):
#     """
#
#     """
#     def __init__(self, p):
#         self.p = p
#
#     def __call__(self, datas):
#         transform = transforms.ToTensor()
#         datas['img'] = transform(datas['img'])
#         return datas
#
#     def may_augment_annotation(self, aug, data, shape):
#         if aug is None:
#             return data
#
#         line_polys = []
#         for poly in data['text_polys']:
#             new_poly = self.may_augment_poly(aug, shape, poly)
#             line_polys.append(new_poly)
#         data['text_polys'] = np.array(line_polys)
#         return data
#
#     def may_augment_poly(self, aug, img_shape, poly):
#         keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
#         keypoints = aug.augment_keypoints(
#             [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
#         poly = [(p.x, p.y) for p in keypoints]
#         return poly
