# coding=utf-8  
# @Time   : 2020/12/29 17:35
# @Auto   : zzf-jeff
from torchocr.datasets.pipelines.compose import PIPELINES

from torchvision import transforms
import cv2
import numpy as np


@PIPELINES.register_module()
class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        if len(self.keep_keys):
            data_dict = {}
            for k, v in data.items():
                if k  in self.keep_keys:
                    data_dict[k] = v
            return data_dict
        else:
            return data


@PIPELINES.register_module()
class Normalize():
    def __init__(self, mean, std, to_rgb=False):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        transform = transforms.Normalize(mean=self.mean, std=self.std)
        data['image'] = transform(data['image'])
        return data


@PIPELINES.register_module()
class ToTensor(object):
    """Image ToTensor -->numpy2Tensor
    Args:
        image ：原始图片
    Returns:
        result :  Over ToTensor
    """

    def __call__(self, data):
        transform = transforms.ToTensor()
        data['img'] = transform(data['img'])
        return data


@PIPELINES.register_module()
class DecodeImage(object):
    def __init__(self, img_mode='RGB', channel_first=False, **kwargs):
        """Decode image,read img

        :param img_mode: GRAY or RGB
        :param channel_first: hwc-->chw
        :param kwargs:
        """
        self.img_mode = img_mode
        self.channel_first = channel_first

    def __call__(self, data):
        img = data['image']
        assert isinstance(img, np.ndarray), 'img not cv2 type'
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]  # BGR-->RGB
        if self.channel_first:
            img = img.transpose((2, 0, 1))
        data['image'] = img
        return data

@PIPELINES.register_module()
class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='hwc', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        data['image'] = (
            img.astype('float32') * self.scale - self.mean) / self.std
        return data

@PIPELINES.register_module()
class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        data['image'] = img.transpose((2, 0, 1))
        return data