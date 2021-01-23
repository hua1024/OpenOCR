# coding=utf-8  
# @Time   : 2021/1/8 15:47
# @Auto   : zzf-jeff
import random
import cv2
import numpy as np
import math
import torch
from torchocr.datasets.builder import PIPELINES



def cal_bbox_cn(x_min, y_min, x_max, y_max, bboxs):
    bboxs_target = []
    for bbox in bboxs:
        # 高度方面，只要出现1/3在外面就不检测， 宽度方面， 只要超过一个h就要检测
        tol_pix_h = (bbox[3] - bbox[1])
        tol_pix_w = bbox[2] - bbox[0]

        c_xmin = bbox[0] - x_min
        c_ymin = bbox[1] - y_min
        c_xmax = bbox[2] - x_min
        c_ymax = bbox[3] - y_min

        # 不检测的类型， 高度方面，有1/3在外面
        # if not (-tol_pix_h < c_ymin < y_max-y_min+tol_pix_h and -tol_pix_h < c_ymax < y_max-y_min + tol_pix_h):
        #     continue  #pixel_based_text2

        # 宽度判断
        in_xmin = max(0, c_xmin)
        in_xmax = min(x_max - x_min, c_xmax)
        if in_xmax - in_xmin <= 10:  # 小于0完全在外面
            continue
        if in_xmax - in_xmin < (tol_pix_h / 2) and tol_pix_w > tol_pix_h:  # pixel_based_text2
            continue

        # 高度判断
        in_ymin = max(0, c_ymin)
        in_ymax = min(y_max - y_min, c_ymax)
        if in_ymax - in_ymin <= 10:
            continue

        if in_ymax - in_ymin < (tol_pix_h / 3 * 2):  # pixel_based_text2
            continue

        if c_xmin < 0:
            c_xmin = 0.
        if c_ymin < 0:
            c_ymin = 0.
        if c_xmax > x_max - x_min:
            c_xmax = x_max - x_min
        if c_ymax > y_max - y_min:
            c_ymax = y_max - y_min
        bboxs_target.append([c_xmin, c_ymin, c_xmax, c_ymax])
    return bboxs_target



def points_to_contour(points):
    contours = [[list(p)] for p in points]
    return np.asarray(contours, dtype=np.int32)


def points_to_contours(points):
    return np.asarray([points_to_contour(points)])


def draw_contours(img, contours, idx=-1, color=1, border_width=1):
    #     img = img.copy()
    cv2.drawContours(img, contours, idx, color, border_width)
    return img


def create_heat_map_patch(xmin, ymin, xmax, ymax):
    """
    :param xmin: 除以4之后的小数点坐标
    :param ymin:
    :param xmax:
    :param ymax:
    :return:
    """
    xmax = xmax - xmin
    ymax = ymax - ymin

    # 先生成 ymax * 1 * 1 大小的片段， 再扩充成 ymax * xmax * 3 大小的feature map
    line_pach = np.zeros((ymax, 1), dtype=float)

    if ymax % 2 == 0:
        # ymax为偶，长度是为奇数
        centre = int(ymax / 2)

        step_len = 1. / centre / 2
        for step, p in enumerate(range(centre, -1, -1)):  # 取不到 -1
            line_pach[p, 0] = 1. - step * step_len

        for step, p in enumerate(range(centre, ymax, 1)):
            line_pach[p, 0] = 1. - step * step_len
    else:
        # ymax为奇数，长度是为偶数
        centre = math.floor(ymax / 2)
        step_len = 1. / centre / 2
        for step, p in enumerate(range(centre, -1, -1)):  # 取不到 -1
            line_pach[p, 0] = 1. - step * step_len

        for step, p in enumerate(range(centre + 1, ymax, 1)):
            line_pach[p, 0] = 1. - step * step_len

    return np.tile(line_pach, (1, xmax))


@PIPELINES.register_module()
class C2TDProcessTrain():
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size

    def __call__(self, data):
        im = data['image']
        text_polys = data['polys']
        text_tags = data['ignore_tags']

        # 创建对应的heat map作为label, 1/4 map_size
        factor = 4.0
        heat_map = np.zeros((int(self.crop_size[1] // factor), int(self.crop_size[0] // factor), 4), dtype=np.float32)
        mask = np.zeros((int(self.crop_size[1] // factor), int(self.crop_size[0] // factor)), dtype=np.float32)
        pixel_cls_weight = np.zeros((int(self.crop_size[1] // factor), int(self.crop_size[0] // factor)),
                                    dtype=np.float32)
        pos_mask = mask.copy()
        bbox_masks = []

        for bbox in text_polys:
            bbox_4 = [int(cond / factor) for cond in bbox]
            xmin, ymin, xmax, ymax = bbox_4  # 缩放到1/4之后的坐标
            bbox_mask = mask.copy()
            bbox_xs = [xmin, xmax, xmax, xmin]
            bbox_ys = [ymin, ymin, ymax, ymax]
            bbox_points = zip(bbox_xs, bbox_ys)
            bbox_contours = points_to_contours(bbox_points)
            draw_contours(bbox_mask, bbox_contours, idx=-1, color=1, border_width=-1)
            bbox_masks.append(bbox_mask)
            pos_mask += bbox_mask
            heat_map_corp = create_heat_map_patch(xmin, ymin, xmax, ymax)
            heat_map[ymin: ymax, xmin: xmax, 0] = heat_map_corp
            # 对其中的每一个点进行标注
            for p_y in range(ymin, ymax):
                for p_x in range(xmin, xmax):
                    heat_map[p_y, p_x, 1] = bbox[1] - (p_y + 0.5) * factor
                    heat_map[p_y, p_x, 2] = bbox[3] - (p_y + 0.5) * factor

        pos_mask = np.asarray(pos_mask == 1, dtype=np.int32)
        num_positive_pixels = np.sum(pos_mask)
        for bbox_idx, bbox_mask in enumerate(bbox_masks):
            bbox_positive_pixel_mask = bbox_mask * pos_mask
            num_bbox_pixels = np.sum(bbox_positive_pixel_mask)
            if num_bbox_pixels > 0:
                per_bbox_weight = num_positive_pixels * 1.0 / len(bbox_masks)
                per_pixel_weight = per_bbox_weight / num_bbox_pixels
                pixel_cls_weight += bbox_positive_pixel_mask * per_pixel_weight

        heat_map[:, :, 3] = pixel_cls_weight

        data['image'] = im
        data['heat_map'] = torch.from_numpy(heat_map)

        return data
