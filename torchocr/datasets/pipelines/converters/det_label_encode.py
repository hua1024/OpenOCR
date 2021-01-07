# coding=utf-8  
# @Time   : 2021/1/7 15:02
# @Auto   : zzf-jeff


'''
主要包括读取label坐标,以及忽略部分不适用的文字
'''

import numpy as np

from ..compose import PIPELINES


@PIPELINES.register_module()
class DetLabelEncode(object):
    def __init__(self, ignore_tags, **kwargs):
        self.ignore_tags = ignore_tags

    def __call__(self, data):
        label = data['label']
        boxes, txts, txt_tags = [], [], []
        for i in range(0, len(label)):
            gt = label[i].split(',')
            box = [(int(gt[i]), int(gt[i + 1])) for i in range(0, 8, 2)]
            txt = ''.join(gt[8:])
            if txt in self.ignore_tags:
                txt_tags.append(True)
            else:
                txt_tags.append(False)

            boxes.append(box)
            txts.append(txt)

        data['polys'] = np.array(boxes)
        data['texts'] = txts
        data['ignore_tags'] = np.array(txt_tags)
        return data
