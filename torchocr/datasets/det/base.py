# coding=utf-8  
# @Time   : 2020/12/29 15:20
# @Auto   : zzf-jeff
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod
import copy
from ..compose import Compose


class DetBaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, ann_file, data_root, pipeline, ignore_tags, img_prefix='', test_mode=False):
        """DetDataset的基类
        Args:
            pipeline : 数据增强模块
            ann_file : 构建自己数据集时使用的文件索引
        """
        super().__init__()

        self.ann_file = ann_file
        self.data_root = data_root
        self.ignore_tags = ignore_tags
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_infos = self.load_annotations(self.ann_file)


    @abstractmethod
    def load_annotations(self, ann_file):
        pass

    def __len__(self):
        return len(self.data_infos)

    @abstractmethod
    def prepare_data(self, datas):
        pass

    def aug_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        results = self.prepare_data(results)
        return self.pipeline(results)

    def __getitem__(self, idx):
        return self.aug_data(idx)
