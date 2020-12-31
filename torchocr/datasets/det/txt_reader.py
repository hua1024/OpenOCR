# coding=utf-8  
# @Time   : 2020/12/25 10:45
# @Auto   : zzf-jeff

import math
from ..builder import DET_DATASET
from .base import DetBaseDataset
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
import numpy as np
import cv2


@DET_DATASET.register_module()
class DetTextDataset(DetBaseDataset, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_annotations(self, ann_file):
        infos = self.read_txt(ann_file)
        data_infos = []
        for (img_path, gt_path) in infos:
            text_polys, ignore_tags, texts = self.get_bboxs(gt_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data_infos.append({'img_path': img_path, 'img': img, 'text_polys': text_polys,
                               'texts': texts, 'ignore_tags': ignore_tags})
        return data_infos

    def get_bboxs(self, gt_path):
        polys = []
        tags = []
        texts = []
        with open(gt_path, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '').strip('\n')
                gt = line.split(',')
                if self.ignore_tags in gt[-1]:
                    tags.append(True)
                else:
                    tags.append(False)
                box = [(int(gt[i]), int(gt[i + 1])) for i in range(0, 8, 2)]
                polys.append(box)
                texts.append('')
            return np.array(polys), tags, texts

    def read_txt(self, txt_path):
        '''
        读取txt文件的标注信息，格式为
        xxx/a/1.png,a
        xxx/a/2.png,a
        Args:
            txt_path: train/valid/test data txt
        Returns:
            imgs：list, all data info
        '''
        with open(txt_path, 'r', encoding='utf-8') as f:
            infos = list(map(lambda line: line.strip().split(','), f))
        return infos

    @abstractmethod
    def prepare_data(self, datas):
        pass
