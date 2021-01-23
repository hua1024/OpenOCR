# coding=utf-8  
# @Time   : 2020/12/25 10:45
# @Auto   : zzf-jeff

from torchocr.datasets.builder import DATASET
from torchocr.datasets.base import BaseDataset
from abc import ABCMeta, abstractmethod
import numpy as np
import cv2
import json

@DATASET.register_module()
class DetJsonDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self, ann_file, pipeline, **kwargs):
        super().__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_file):
        infos = self.read_txt(ann_file)
        data_infos = []
        for (img_path, gt_path) in infos:
            labels = self.get_bboxs(gt_path)
            img = cv2.imread(img_path)
            data_infos.append({'img_path': img_path, 'image': img, 'label': labels})
        return data_infos

    def get_bboxs(self, gt_path):
        labels = []
        with open(gt_path, 'r',encoding='utf-8') as file:
            instances = json.loads(file.read())
            for instance in instances:
                pts = instance['points']
                labels.append(pts)
        return labels


    def read_txt(self, txt_path):
        '''
        读取txt文件的标注信息，格式为
        xxx/a/1.png,a
        xxx/a/2.png,a
        Args:
            txt_path: train/valid/test data txt or json
        Returns:
            imgs：list, all data info
        '''
        with open(txt_path, 'r', encoding='utf-8') as f:
            infos = list(map(lambda line: line.strip().split(','), f))
        return infos


