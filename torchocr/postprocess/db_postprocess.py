# coding=utf-8  
# @Time   : 2020/12/29 14:10
# @Auto   : zzf-jeff

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from .builder import POSTPROCESS
import torch


@POSTPROCESS.register_module()
class DBPostProcess():
    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=1.5,
                 is_polygon=False,
                 dilation_kernel=None):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.is_polygon = is_polygon
        self.dilation_kernel = dilation_kernel

    def __call__(self, pred, data_batch):
        '''
        batch: (image, polygons, ignore_tags
        h_w_list: 包含[h,w]的数组
        pred:
            binary: text region segmentation map, with shape (N, 1,H, W)
        '''
        # 取出channel的数据 [n,1,h,w]-->[n,h,w]

        pred = pred.detach().cpu().numpy()
        pred = pred[:, 0, :, :]
        segmentation = self.binarize(pred)
        result_batch = []
        if isinstance(data_batch, dict):
            shape_list = data_batch['shape']
        else:
            shape_list = data_batch

        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]

            # tmp func,model type has bug
            if isinstance(src_h, torch.Tensor):
                src_h = src_h.numpy()
            if isinstance(src_w, torch.Tensor):
                src_w = src_w.numpy()
            # dilate
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]

            if not self.is_polygon:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                       src_w, src_h)
            else:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], mask,
                                                          src_w, src_h)

            result_batch.append({'points': boxes, 'scores': scores})

        return result_batch

    def binarize(self, pred):
        return pred > self.thresh

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        # 中心(x,y), (宽,高), 旋转角度
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]

        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        # poly和bbox得到的结果不一致,poly的score高于bbox

        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        # cv2.imwrite('ssa.png',(bitmap * 255).astype(np.uint8)[ymin:ymax + 1, xmin:xmax + 1])
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''
        # bitmap = _bitmap.cpu().numpy()  # The first channel
        # pred = pred.cpu().detach().numpy()
        bitmap = _bitmap
        height, width = bitmap.shape
        # cv2.imwrite('thresh_map.png',(bitmap * 255).astype(np.uint8))
        try:
            contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
        except:
            img, contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        scores = []
        # 容易出现大于max_candidates的map,目前进行排序处理
        if len(contours) > self.max_candidates:
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        for contour in contours[:self.max_candidates]:
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points, dtype=np.int32).reshape(-1, 2)
            # cal score using np.mean
            score = self.box_score_fast(pred, points)
            if score < self.box_thresh:
                continue

            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)

        return boxes, scores

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        # bitmap = _bitmap.cpu().numpy()  # The first channel
        # pred = pred.cpu().detach().numpy()
        bitmap = _bitmap
        height, width = bitmap.shape
        boxes = []
        scores = []
        # cv2.imwrite('thresh_map.png', (bitmap * 255).astype(np.uint8))
        try:
            contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
        except:
            img, contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)

        # 容易出现大于max_candidates的map,目前进行排序处理
        if len(contours) > self.max_candidates:
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        for contour in contours[:self.max_candidates]:

            epsilon = 0.001 * cv2.arcLength(contour, True)
            # contour_len = cv2.arcLength(contour, True)
            # epsilon = 0.001 * contour_len if contour_len > 300 else 0.01 * contour_len
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, points)

            if score < self.box_thresh:
                continue
            # shape -->[1,n,2]
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            # shape -->[n,2]
            box = box.reshape(-1, 2)
            if len(box) == 0:
                continue
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)

            boxes.append(box.astype(np.int16))
            scores.append(score)

        return boxes, scores
