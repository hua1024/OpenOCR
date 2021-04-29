# coding=utf-8  
# @Time   : 2021/1/8 15:47
# @Auto   : zzf-jeff
import random
import cv2
import numpy as np
import math
import torch
from torchocr.datasets.builder import PIPELINES


@PIPELINES.register_module()
class C2TDProcessTrain():
    def __init__(self, crop_size, scale=0.25):
        super().__init__()
        self.crop_size = crop_size
        self.scale = scale

    def draw_contours(self, img, contours, idx=-1, color=1, border_width=1):
        #     img = img.copy()
        cv2.drawContours(img, contours, idx, color, border_width)
        return img

    def points_to_contour(self, points):
        contours = [[list(p)] for p in points]
        return np.asarray(contours, dtype=np.int32)

    def points_to_contours(self, points):
        return np.asarray([self.points_to_contour(points)])

    def find_contours(self, mask, method=None):
        if method is None:
            method = cv2.CHAIN_APPROX_SIMPLE
        mask = np.asarray(mask, dtype=np.uint8)
        mask = mask.copy()
        try:
            contours, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP,
                                           method=method)
        except:
            _, contours, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP,
                                              method=method)
        return contours

    def cal_gt_for_single_image(self, normed_xs, normed_ys, labels):

        text_label = 1
        ignore_label = -1
        background_label = 0
        bbox_border_width = 1
        pixel_cls_border_weight_lambda = 1.0


        # 创建对应的heat map作为label, 1/4 map_size
        score_map_shape = [int(self.crop_size[0] * self.scale), int(self.crop_size[1] * self.scale)]
        h, w = score_map_shape

        mask = np.zeros(score_map_shape, dtype=np.int32)
        pixel_cls_weight = np.zeros(score_map_shape, dtype=np.float32)
        pixel_cls_label = np.ones(score_map_shape, dtype=np.int32) * background_label
        heat_map = np.zeros((h, w, 4), dtype=np.float32)

        # validate the args
        assert np.ndim(normed_xs) == 2
        assert np.shape(normed_xs)[-1] == 4
        assert np.shape(normed_xs) == np.shape(normed_ys)
        assert len(normed_xs) == len(labels)

        num_positive_bboxes = np.sum(np.asarray(labels) == text_label)

        # rescale normalized xys to absolute values
        xs = normed_xs * w
        ys = normed_ys * h

        ## get the masks of all bboxes
        bbox_masks = []
        pos_mask = mask.copy()

        for bbox_idx, (bbox_xs, bbox_ys) in enumerate(zip(xs, ys)):
            if labels[bbox_idx] == background_label:
                continue
            bbox_mask = mask.copy()
            bbox_points = zip(bbox_xs, bbox_ys)

            bbox_contours = self.points_to_contours(bbox_points)
            self.draw_contours(bbox_mask, bbox_contours, idx=-1,
                               color=1, border_width=-1)
            bbox_masks.append(bbox_mask)

            if labels[bbox_idx] == text_label:
                pos_mask += bbox_mask
                # # 对其中的每一个点进行标注
                ori_ymin = int(bbox_ys[1] / self.scale)
                ori_ymax = int(bbox_ys[3] / self.scale)

                min_bbox_ys = int(min(bbox_ys.tolist()))
                max_bbox_ys = int(max(bbox_ys.tolist()))
                min_bbox_xs = int(min(bbox_xs.tolist()))
                max_bbox_xs = int(max(bbox_xs.tolist()))

                for p_y in range(min_bbox_ys, max_bbox_ys):
                    for p_x in range(min_bbox_xs, max_bbox_xs):
                        heat_map[p_y, p_x, 1] = ori_ymin - p_y
                        heat_map[p_y, p_x, 2] = ori_ymax - p_y

        # treat overlapped in-bbox pixels as negative,
        # and non-overlapped  ones as positive
        pos_mask = np.asarray(pos_mask == 1, dtype=np.int32)
        num_positive_pixels = np.sum(pos_mask)

        ## add all bbox_maskes, find non-overlapping pixels
        sum_mask = np.sum(bbox_masks, axis=0)
        not_overlapped_mask = sum_mask == 1

        ## gt and weight calculation
        for bbox_idx, bbox_mask in enumerate(bbox_masks):
            bbox_label = labels[bbox_idx]
            if bbox_label == ignore_label:
                # for ignored bboxes, only non-overlapped pixels are encoded as ignored
                bbox_ignore_pixel_mask = bbox_mask * not_overlapped_mask
                pixel_cls_label += bbox_ignore_pixel_mask * ignore_label
                continue

            if labels[bbox_idx] == background_label:
                continue
            # from here on, only text boxes left.

            # for positive bboxes, all pixels within it and pos_mask are positive
            bbox_positive_pixel_mask = bbox_mask * pos_mask
            # background or text is encoded into cls gt
            pixel_cls_label += bbox_positive_pixel_mask * bbox_label

            # let N denote num_positive_pixels
            # weight per pixel = N /num_positive_bboxes / n_pixels_in_bbox
            # so all pixel weights in this bbox sum to N/num_positive_bboxes
            # and all pixels weights in this image sum to N, the same
            # as setting all weights to 1
            num_bbox_pixels = np.sum(bbox_positive_pixel_mask)
            if num_bbox_pixels > 0:
                per_bbox_weight = num_positive_pixels * 1.0 / num_positive_bboxes
                per_pixel_weight = per_bbox_weight / num_bbox_pixels
                pixel_cls_weight += bbox_positive_pixel_mask * per_pixel_weight

            ## the border of bboxes might be distored because of overlapping
            ## so recalculate it, and find the border mask
            new_bbox_contours = self.find_contours(bbox_positive_pixel_mask)
            bbox_border_mask = mask.copy()
            self.draw_contours(bbox_border_mask, new_bbox_contours, -1,
                               color=1, border_width=bbox_border_width * 2 + 1)
            bbox_border_mask *= bbox_positive_pixel_mask
            bbox_border_cords = np.where(bbox_border_mask)

            ## give more weight to the border pixels if configured
            pixel_cls_weight[bbox_border_cords] *= pixel_cls_border_weight_lambda


        heat_map[:, :, 0] = pixel_cls_label
        heat_map[:, :, 3] = pixel_cls_weight

        return heat_map

    def __call__(self, data):
        im = data['image']
        text_polys = data['polys']
        text_tags = data['ignore_tags']

        h, w = im.shape[:2]
        # cal norm point
        norm_x = text_polys[:, :, 0] / w
        norm_y = text_polys[:, :, 1] / h
        # ignore_tags -1 is ignore,1 is text
        text_tags = np.where(np.array(text_tags) <= 0, 1, -1)

        heat_map = self.cal_gt_for_single_image(norm_x, norm_y, text_tags)

        data['image'] = im
        data['heat_map'] = torch.from_numpy(heat_map.transpose((2, 0, 1)))

        return data


