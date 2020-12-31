# coding=utf-8  
# @Time   : 2020/12/29 16:49
# @Auto   : zzf-jeff


import math
from ..builder import DET_DATASET
from .txt_reader import DetTextDataset
from .random_crop import EastRandomCropData
from .make_border_map import MakeBorderMap
from .make_shrink_map import MakeShrinkMap


@DET_DATASET.register_module()
class DBDataset(DetTextDataset):
    def __init__(self, pre_params, **kwargs):
        super().__init__(**kwargs)

        shrink_ratio = pre_params['shrink_ratio']
        thresh_min = pre_params['thresh_min']
        thresh_max = pre_params['thresh_max']
        min_text_size = pre_params['min_text_size']
        size = pre_params['size']
        max_tries = pre_params['max_tries']
        min_crop_side_ratio = pre_params['min_crop_side_ratio']

        self.random_crop = EastRandomCropData(size=size, max_tries=max_tries, min_crop_side_ratio=min_crop_side_ratio)
        self.make_border_map = MakeBorderMap(shrink_ratio=shrink_ratio, thresh_min=thresh_min, thresh_max=thresh_max)
        self.make_shrink_map = MakeShrinkMap(min_text_size=min_text_size, shrink_ratio=shrink_ratio)

    # 争对不同算法进行预处理
    def prepare_data(self, datas):
        """prepare_data 进行db的预处理

        :param datas:
            {'img_path': img_path, 'img': img, 'text_polys': text_polys,
                'texts': '', 'ignore_tags': ignore_tags}
        :return:
            {'img_path': img_path, 'img': img, 'text_polys': text_polys,
                'texts': '', 'ignore_tags': ignore_tags,'threshold_map':threshold_map,
                'threshold_mask':threshold_mask,'shrink_map':shrink_map,
                'shrink_mask':shrink_mask}
        """
        datas = self.random_crop(datas)
        datas = self.make_border_map(datas)
        datas = self.make_shrink_map(datas)

        # threshold_map = datas['threshold_map']
        # threshold_mask = datas['threshold_mask']
        # shrink_mask = datas['shrink_mask']
        # shrink_map = datas['shrink_map']
        # img = datas['img']
        # points = datas['text_polys']
        # print(img.shape)
        #
        # import cv2
        # for point in points:
        #     point = point.astype(int)
        #     cv2.polylines(img, [point], True,(255,255,0))
        #
        #
        # cv2.imwrite('img.png', img)
        # cv2.imwrite('threshold_mask.png', threshold_mask)
        # cv2.imwrite('threshold_map.png', threshold_map)
        # cv2.imwrite('threshold_mask.png', threshold_mask)
        # cv2.imwrite('shrink_mask.png', shrink_mask)
        # cv2.imwrite('shrink_map.png', shrink_map)


        # print(datas)

        return datas
