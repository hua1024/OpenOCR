# coding=utf-8  
# @Time   : 2020/12/25 10:45
# @Auto   : zzf-jeff

from torchocr.datasets.builder import DATASET
from torchocr.datasets.base import BaseDataset
from abc import ABCMeta, abstractmethod
import numpy as np
import cv2
import json
from tqdm import tqdm


@DATASET.register_module()
class DetJsonDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self, ann_file, pipeline, mode, data_root, **kwargs):
        if 'split_type' in kwargs:
            self.split_type = kwargs['split_type']
        else:
            self.split_type = ','
        super().__init__(ann_file, pipeline, mode, data_root, **kwargs)

    def load_annotations(self, ann_file):
        infos = self.read_txt(ann_file, self.split_type)
        data_infos = []
        for (img_path, gt_path) in tqdm(infos):
            labels, texts = self.get_bboxs(gt_path)
            data_infos.append({'img_path': img_path, 'label': labels, 'text': texts})
        return data_infos

    def get_bboxs(self, gt_path):
        labels = []
        texts = []
        with open(gt_path, 'r', encoding='utf-8') as file:
            instances = json.loads(file.read())
            for instance in instances:
                pts = instance['points']
                pts = pts.split(',')
                texts.append(['ocr'])
                labels.append(pts)
        return labels, texts

    def read_txt(self, txt_path, split_type):
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
            infos = list(map(lambda line: line.strip().split(split_type), f))
        return infos


@DATASET.register_module()
class AyxDetJsonDataset(DetJsonDataset):

    def load_annotations(self, ann_file):
        data_infos = []
        infos = super(AyxDetJsonDataset, self).read_txt(ann_file, self.split_type)
        for info in infos:
            txt_file, use_flag = info
            if int(use_flag) == 1:
                # 调用父类的方法
                data_infos += super(AyxDetJsonDataset, self).load_annotations(txt_file)

        return data_infos
