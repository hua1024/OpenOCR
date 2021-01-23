# coding=utf-8  
# @Time   : 2021/1/13 14:27
# @Auto   : zzf-jeff


import cv2
import numpy as np
from .builder import POSTPROCESS
import torch
from .pse import pse_decode


@POSTPROCESS.register_module()
class PSEPostProcess():
    def __init__(self, thresh, min_score, min_kernel_area, scale, is_poly=False):
        self.thresh = thresh
        self.min_kernel_area = min_kernel_area
        self.scale = scale
        self.is_poly = is_poly
        self.min_score = min_score

    def __call__(self, pred, data_batch):
        result_batch = []

        if isinstance(data_batch, dict):
            shape_list = data_batch['shape']
        else:
            shape_list = data_batch

        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]

            # if isinstance(src_h, torch.Tensor):
            #     src_h = src_h.numpy()
            # if isinstance(src_w, torch.Tensor):
            #     src_w = src_w.numpy()

            pred_single = pred[batch_index]
            if not self.is_poly:
                preds, boxes_list = pse_decode(pred_single, self.scale, self.thresh)

                if len(boxes) > 0:
                    boxes = np.array(boxes)
                    boxes[:, :, 0] = boxes[:, :, 0] * ratio_w
                    boxes[:, :, 1] = boxes[:, :, 1] * ratio_h

                result_batch.append({'points': boxes, 'scores': scores})

        return result_batch
