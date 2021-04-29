# coding=utf-8  
# @Time   : 2020/12/25 10:45
# @Auto   : zzf-jeff

from torchocr.datasets.builder import DATASET
from torchocr.datasets.base import BaseDataset
from abc import ABCMeta, abstractmethod
import numpy as np
import cv2
import os
from tqdm import tqdm


@DATASET.register_module()
class DetTextICDAR15(BaseDataset, metaclass=ABCMeta):
    def __init__(self, ann_file, pipeline, mode, data_root, **kwargs):
        super().__init__(ann_file, pipeline, mode, data_root, **kwargs)

    def load_annotations(self, ann_file):
        infos = self.read_txt(ann_file)
        data_infos = []
        for (img_path, gt_path) in tqdm(infos):
            labels, texts = self.get_bboxs(gt_path)
            data_infos.append({'img_path': img_path, 'label': labels, 'text': texts})
        return data_infos

    def get_bboxs(self, gt_path):
        """

        :param gt_path:
        :return: labels [['2','3','4','5']]
                texts [['sss']]
        """
        labels = []
        texts = []
        with open(gt_path, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '').strip('\n')
                gt = line.split(',')
                labels.append(gt[:8])
                texts.append(gt[8:])
            return labels, texts

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


@DATASET.register_module()
class DetTextDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self, ann_file, pipeline, mode, data_root, **kwargs):
        super().__init__(ann_file, pipeline, mode, data_root, **kwargs)

    def load_annotations(self, ann_file):
        infos = self.read_txt(ann_file)
        data_infos = []
        for (img_path, gt_path) in tqdm(infos):
            if self.data_root:
                img_path = os.path.join(self.data_root, img_path)
                gt_path = os.path.join(self.data_root, gt_path)

            labels, texts = self.get_bboxs(gt_path)
            data_infos.append({'img_path': img_path, 'label': labels, 'text': texts})
        return data_infos

    def get_bboxs(self, gt_path, is_voc=False):
        labels = []
        texts = []
        with open(gt_path, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '').strip('\n')
                gt = line.split(',')
                labels.append(gt[:8])
                texts.append(['ocr'])
            return labels, texts

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


@DATASET.register_module()
class RecTextDataset(BaseDataset, metaclass=ABCMeta):
    def __init__(self, ann_file, pipeline, mode, data_root, **kwargs):
        super().__init__(ann_file, pipeline, mode, data_root, **kwargs)

    def load_annotations(self, ann_file):
        infos = self.read_txt(ann_file)

        data_infos = []
        for info in tqdm(infos):
            if len(info) != 2:
                continue
            img_path, label = info[0], info[1]
            if self.data_root:
                img_path = os.path.join(self.data_root, img_path)

            data_infos.append({'img_path': img_path, 'label': label})
        return data_infos

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
            infos = list(map(lambda line: line.strip().split(' ',1), f))
        return infos
