# coding=utf-8  
# @Time   : 2020/12/29 15:20
# @Auto   : zzf-jeff
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod
import copy
from torchocr.datasets.pipelines.compose import Compose


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
        data = self.aug_data(idx)
        return data
