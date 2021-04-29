# coding=utf-8  
# @Time   : 2021/1/27 14:47
# @Auto   : zzf-jeff


class RecAug(object):
    def __init__(self, aug_prob=0.4, **kwargs):
        self.aug_prob = aug_prob

    def __call__(self, data):
        img = data['image']
        img = warp(img, 10, self.use_tia, self.aug_prob)
        data['image'] = img
        return data
