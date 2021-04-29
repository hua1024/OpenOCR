# coding=utf-8  
# @Time   : 2020/12/4 14:51
# @Auto   : zzf-jeff


import Levenshtein
import numpy as np
from .builder import METRICS

@METRICS.register_module()
class RecMetric(object):
    def __init__(self, main_indicator='acc', **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        char_num = 0

        norm_edit_dis = 0.0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            norm_edit_dis += Levenshtein.distance(pred, target) / max(
                len(pred), len(target))
            if pred == target:
                correct_num += 1

            all_num += 1

        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis

        return {
            'acc': correct_num / all_num,
            'norm_edit_dis': 1 - norm_edit_dis / all_num
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = self.correct_num / self.all_num
        norm_edit_dis = 1 - self.norm_edit_dis / self.all_num
        self.reset()
        return {'acc': acc, 'norm_edit_dis': norm_edit_dis}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
