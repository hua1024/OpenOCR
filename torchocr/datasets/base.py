# coding=utf-8  
# @Time   : 2020/12/29 15:20
# @Auto   : zzf-jeff
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod
import copy
from torchocr.datasets.pipelines.compose import Compose
import numpy as np


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, ann_file, pipeline, **kwargs):
        """DetDataset的基类
        Args:
            pipeline : 数据增强模块
            ann_file : 构建自己数据集时使用的文件索引
        """
        super().__init__()

        self.ann_file = ann_file
        self.pipeline = Compose(pipeline)
        self.data_infos = self.load_annotations(self.ann_file)

    @abstractmethod
    def load_annotations(self, ann_file):
        pass

    def __len__(self):
        return len(self.data_infos)

    def aug_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __getitem__(self, idx):
        # todo : data出问题的情况下，重新找，写法上不严谨
        # 如果当前的index有问题，需要继续随机的找一个
        data = self.aug_data(idx)
        if data is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return data
