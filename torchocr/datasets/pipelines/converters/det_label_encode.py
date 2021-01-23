# coding=utf-8  
# @Time   : 2021/1/7 15:02
# @Auto   : zzf-jeff


'''
主要包括读取label坐标,以及忽略部分不适用的文字
todo : 添加越界检查
'''

import numpy as np
import os
from ..compose import PIPELINES


@PIPELINES.register_module()
class DetLabelEncode(object):
    def __init__(self, ignore_tags, **kwargs):
        self.ignore_tags = ignore_tags

    def expand_points_num(self, boxes):
        # 对于部分多边形多点的，要expand点到shape一致
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)

        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes

    def __call__(self, data):
        try:
            if data is None:
                return None
            label = data['label']
            boxes, txts, txt_tags = [], [], []
            for i in range(0, len(label)):
                gt = label[i].split(',')
                if not len(gt) >= 8:
                    continue
                box = [[int(gt[i]), int(gt[i + 1])] for i in range(0, len(gt), 2)]

                txt = 'text'
                # txt = gt[8:]
                if txt in self.ignore_tags:
                    txt_tags.append(True)
                else:
                    txt_tags.append(False)
                boxes.append(box)
                txts.append(txt)

            boxes = self.expand_points_num(boxes)

            data['polys'] = np.array(boxes, dtype='float32')
            data['texts'] = txts
            data['ignore_tags'] = np.array(txt_tags, dtype='float32')
            return data

        except Exception as e:
            file_name = os.path.basename(__file__).split(".")[0]
            print('{} --> '.format(file_name), e)
            return None
