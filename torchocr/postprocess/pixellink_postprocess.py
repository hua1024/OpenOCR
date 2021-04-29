# coding=utf-8  
# @Time   : 2020/12/29 14:10
# @Auto   : zzf-jeff

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from .builder import POSTPROCESS
import torch
import torch.nn.functional as F


@POSTPROCESS.register_module()
class PixelLinkPostProcess():
    def __init__(self, neighbor_num, pixel_conf, link_conf, min_area, min_height):
        import pyximport
        pyximport.install()
        from .pixel_link_decode import decode_image_by_join
        self.decode_image_by_join = decode_image_by_join
        self.neighbor_num = neighbor_num
        self.link_conf = link_conf
        self.pixel_conf = pixel_conf
        self.min_area = min_area
        self.min_height = min_height

    def decode_image(self, pixel_scores, link_scores,
                     pixel_conf_threshold, link_conf_threshold):

        mask = self.decode_image_by_join(pixel_scores, link_scores,
                                         pixel_conf_threshold, link_conf_threshold, self.neighbor_num)
        return mask

    def decode_batch(self, pixel_cls_scores, pixel_link_scores):

        batch_size = pixel_cls_scores.shape[0]
        batch_mask = []
        for image_idx in range(batch_size):
            image_pos_pixel_scores = pixel_cls_scores[image_idx, :, :]
            image_pos_link_scores = pixel_link_scores[image_idx, :, :, :]
            mask = self.decode_image(
                image_pos_pixel_scores, image_pos_link_scores,
                self.pixel_conf, self.link_conf
            )
            batch_mask.append(mask)

        return np.asarray(batch_mask, np.int32)

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

    def min_area_rect(self, cnt):
        """
        Args:
            xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
            ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
                Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
        Return:
            the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta].
        """
        rect = cv2.minAreaRect(cnt)
        cx, cy = rect[0]
        w, h = rect[1]
        theta = rect[2]
        box = [cx, cy, w, h, theta]
        return box, w * h

    def rect_to_xys(self, rect, image_shape):
        """Convert rect to xys, i.e., eight points
        The `image_shape` is used to to make sure all points return are valid, i.e., within image area
        """
        h, w = image_shape[0:2]

        def get_valid_x(x):
            if x < 0:
                return 0
            if x >= w:
                return w - 1
            return x

        def get_valid_y(y):
            if y < 0:
                return 0
            if y >= h:
                return h - 1
            return y

        rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
        points = cv2.boxPoints(rect)
        points = np.int0(points)
        for i_xy, (x, y) in enumerate(points):
            x = get_valid_x(x)
            y = get_valid_y(y)
            points[i_xy, :] = [x, y]
        points = np.reshape(points, -1)
        return points

    def mask_to_bboxes(self, mask, image_shape):
        image_h, image_w = image_shape[0:2]

        bboxes = []
        max_bbox_idx = mask.max()
        mask = cv2.resize(mask, (image_w, image_h),
                          interpolation=cv2.INTER_NEAREST)

        for bbox_idx in range(1, max_bbox_idx + 1):
            bbox_mask = mask == bbox_idx
            cnts = self.find_contours(bbox_mask)
            if len(cnts) == 0:
                continue
            cnt = cnts[0]
            rect, rect_area = self.min_area_rect(cnt)

            w, h = rect[2:-1]
            if min(w, h) < self.min_height:
                continue

            if rect_area < self.min_area:
                continue

            xys = self.rect_to_xys(rect, image_shape)
            bboxes.append(xys)

        return bboxes

    def to_bboxes(self, image_shape, pixel_pos_scores, link_pos_scores):
        link_pos_scores = np.transpose(link_pos_scores, (0, 2, 3, 1))
        mask = self.decode_batch(pixel_pos_scores, link_pos_scores)[0, ...]
        bboxes = self.mask_to_bboxes(mask, image_shape)
        return mask, bboxes

    def __call__(self, pred, data_batch):
        """pixel link post process

        :param pred:
        :param data_batch:
        :return:
        """

        result_batch = []
        if isinstance(data_batch, dict):
            shape_list = data_batch['shape']
        else:
            shape_list = data_batch

        # for batch_index in range(pred.shape[0]):
        src_h, src_w, ratio_h, ratio_w = shape_list[0]
        if isinstance(src_h, torch.Tensor):
            src_h = src_h.numpy()
        if isinstance(src_w, torch.Tensor):
            src_w = src_w.numpy()

        score = pred
        shape = score.shape

        pixel_pos_scores = F.softmax(score[:, 0:2, :, :], dim=1)[:, 1, :, :]
        # FIXME the dimention should be changed
        link_scores = score[:, 2:, :, :].view(shape[0], 2, self.neighbor_num, shape[2], shape[3])
        link_pos_scores = F.softmax(link_scores, dim=1)[:, 1, :, :, :]

        pixel_pos_scores = pixel_pos_scores.cpu().detach().numpy()
        link_pos_scores = link_pos_scores.cpu().detach().numpy()

        mask, bboxes = self.to_bboxes(shape[2:], pixel_pos_scores, link_pos_scores)

        res_scores = []
        res_boxes = []

        for box in bboxes:
            box = box.reshape(4, 2)
            box = np.array(box, dtype=np.float32)
            # 默认对应1/4的map
            box *= 4
            box[:, 0] = np.clip(box[:, 0], 0, src_w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, src_h - 1)
            res_boxes.append(box.astype(np.int16))
            res_scores.append(self.pixel_conf)
            result_batch.append({'points': res_boxes, 'scores': res_scores})

        return result_batch
