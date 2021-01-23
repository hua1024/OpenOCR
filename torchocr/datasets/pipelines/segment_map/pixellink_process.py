# coding=utf-8  
# @Time   : 2021/1/22 15:16
# @Auto   : zzf-jeff


import numpy as np
import cv2
import torch
import pyclipper
from torchocr.datasets.builder import PIPELINES
from torchocr.datasets.pipelines.img_aug.random_crop import EastRandomCropData


@PIPELINES.register_module()
class PixelLinkProcessTrain():

    def __init__(self,
                 size,
                 max_tries=50,
                 min_crop_side_ratio=0.1,
                 keep_ratio=True,
                 num_neighbours=4):
        """PixelLink label process for train

        :param size: crop img size .e.g=[640,640]
        :param max_tries:
        :param min_crop_side_ratio:
        :param keep_ratio:
        """
        self.crop_func = EastRandomCropData(size, max_tries, min_crop_side_ratio, keep_ratio)
        self.crop_size = size
        self.num_neighbours = num_neighbours

    def cal_norm_xy(self, polys, img_size):
        """

        :param polys:
        :param img_size:
        :return:
        """

        # print(polys[:, 0].reshape(-1, 1))

        norm_x = np.hstack((polys[:, 0].reshape(-1, 1), polys[:, 2].reshape(-1, 1),
                            polys[:, 2].reshape(-1, 1), polys[:, 0].reshape(-1, 1)))
        norm_x = np.array(norm_x, dtype=np.float32)

        # print(norm_x)

        norm_x /= img_size[1]
        norm_y = np.hstack((polys[:, 1].reshape(-1, 1), polys[:, 1].reshape(-1, 1),
                            polys[:, 3].reshape(-1, 1), polys[:, 3].reshape(-1, 1)))

        norm_y = np.array(norm_y, dtype=np.float32)
        norm_y /= img_size[0]

        labels = np.ones(norm_x.shape[0], dtype=np.int32)

        return norm_x, norm_y, labels

    def get_neighbours_8(self, x, y):
        """
        Get 8 neighbours of point(x, y)
        8 neighbors:
            0 1 2
            7 - 3
            6 5 4
        """
        return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
                (x - 1, y), (x + 1, y), \
                (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

    def get_neighbours_4(self, x, y):
        """
        Get 4 neighbours of point(x, y)
        4 neighbors: ? 怪怪的
              4
            1 - 2
              3
        """
        return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]

    def get_neighbours(self, x, y, num_neighbours):
        ''' link neighbours

        :param x:
        :param y:
        :param num_neighbours:
        :return:
        '''
        if num_neighbours == 4:
            return self.get_neighbours_4(x, y)
        else:
            return self.get_neighbours_8(x, y)

    def cal_gt_for_single_image(self, normed_xs, normed_ys, labels, config):
        """
        Args:
            xs, ys: both in shape of (N, 4),
                and N is the number of bboxes,
                their values are normalized to [0,1]
            labels: shape = (N,), only two values are allowed:
                                                            -1: ignored
                                                            1: text
        Return:
            pixel_cls_label
            pixel_cls_weight
            pixel_link_label
            pixel_link_weight
        """
        score_map_shape = self.crop_size
        pixel_cls_weight_method = 'PIXEL_CLS_WEIGHT_bbox_balanced'
        h, w = score_map_shape
        text_label = 1
        ignore_label = -1
        background_label = 0
        num_neighbours = self.num_neighbours
        bbox_border_width = 1
        pixel_cls_border_weight_lambda = 1.0

        # validate the args
        assert np.ndim(normed_xs) == 2
        assert np.shape(normed_xs)[-1] == 4
        assert np.shape(normed_xs) == np.shape(normed_ys)
        assert len(normed_xs) == len(labels)

        num_positive_bboxes = np.sum(np.asarray(labels) == text_label)
        # rescale normalized xys to absolute values
        xs = normed_xs * w
        ys = normed_ys * h

        # initialize ground truth values
        mask = np.zeros(score_map_shape, dtype=np.int32)
        pixel_cls_label = np.ones(score_map_shape, dtype=np.int32) * background_label
        pixel_cls_weight = np.zeros(score_map_shape, dtype=np.float32)

        pixel_link_label = np.zeros((h, w, num_neighbours), dtype=np.int32)
        pixel_link_weight = np.ones((h, w, num_neighbours), dtype=np.float32)

        # find overlapped pixels, and consider them as ignored in pixel_cls_weight
        # and pixels in ignored bboxes are ignored as well
        # That is to say, only the weights of not ignored pixels are set to 1

        ## get the masks of all bboxes
        bbox_masks = []
        pos_mask = mask.copy()
        for bbox_idx, (bbox_xs, bbox_ys) in enumerate(zip(xs, ys)):
            if labels[bbox_idx] == background_label:
                continue

            bbox_mask = mask.copy()

            bbox_points = zip(bbox_xs, bbox_ys)
            bbox_contours = points_to_contours(bbox_points)
            draw_contours(bbox_mask, bbox_contours, idx=-1,
                          color=1, border_width=-1)

            bbox_masks.append(bbox_mask)

            if labels[bbox_idx] == text_label:
                pos_mask += bbox_mask

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
            # print(type(bbox_positive_pixel_mask), type(bbox_label))
            pixel_cls_label += bbox_positive_pixel_mask * bbox_label

            # for the pixel cls weights, only positive pixels are set to ones
            if pixel_cls_weight_method == "PIXEL_CLS_WEIGHT_all_ones":
                pixel_cls_weight += bbox_positive_pixel_mask
            elif pixel_cls_weight_method == "PIXEL_CLS_WEIGHT_bbox_balanced":
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
            else:
                raise (ValueError, 'pixel_cls_weight_method not supported:%s' \
                       % (pixel_cls_weight_method))

            ## calculate the labels and weights of links
            ### for all pixels in  bboxes, all links are positive at first
            bbox_point_cords = np.where(bbox_positive_pixel_mask)
            pixel_link_label[bbox_point_cords] = 1

            ## the border of bboxes might be distored because of overlapping
            ## so recalculate it, and find the border mask
            new_bbox_contours = find_contours(bbox_positive_pixel_mask)
            bbox_border_mask = mask.copy()
            draw_contours(bbox_border_mask, new_bbox_contours, -1,
                          color=1, border_width=bbox_border_width * 2 + 1)
            bbox_border_mask *= bbox_positive_pixel_mask
            bbox_border_cords = np.where(bbox_border_mask)

            ## give more weight to the border pixels if configured
            pixel_cls_weight[bbox_border_cords] *= pixel_cls_border_weight_lambda

            ### change link labels according to their neighbour status
            border_points = zip(*bbox_border_cords)

            def in_bbox(nx, ny):
                return bbox_positive_pixel_mask[ny, nx]

            for y, x in border_points:
                neighbours = get_neighbours(x, y, config)
                for n_idx, (nx, ny) in enumerate(neighbours):
                    if not is_valid_cord(nx, ny, w, h) or not in_bbox(nx, ny):
                        pixel_link_label[y, x, n_idx] = 0

        pixel_cls_weight = np.asarray(pixel_cls_weight, dtype=np.float32)
        pixel_link_weight *= np.expand_dims(pixel_cls_weight, axis=-1)

        return pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight

    def __call__(self, data):
        if data is None:
            return None

        data = self.crop_func(data)
        img = data['image']
        text_polys = data['polys']
        text_tags = data['ignore_tags']

        # print(text_polys)
        # print('crop')
        # print(text_polys.shape)
        norm_x, norm_y, labels = self.cal_norm_xy(text_polys, self.crop_size)
        pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight = \
            self.cal_gt_for_single_image(norm_x, norm_y, labels)

        return data
