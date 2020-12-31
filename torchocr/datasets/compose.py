# coding=utf-8  
# @Time   : 2020/11/10 11:10
# @Auto   : zzf-jeff

from torchocr.datasets.builder import PIPELINES
from collections.abc import Sequence
from torchocr.utils.registry import build_from_cfg


@PIPELINES.register_module()
class Compose(object):
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            # callable() 函数用于检查一个对象是否是可调用的
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got {}'.format(type(transform)))

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {}'.format(t)

        format_string += '\n)'
        return format_string


