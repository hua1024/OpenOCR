# coding=utf-8  
# @Time   : 2021/1/9 14:55
# @Auto   : zzf-jeff


import math
import cv2
import numpy as np
import torch

from torchocr.datasets.builder import PIPELINES


# todo : 很多处理都跟其他segment的方式重复了,感觉独立处理共用好一点
@PIPELINES.register_module()
class EASTProcessTrain(object):
    def __init__(self,
                 length=512,
                 scale=0.25,
                 min_text_size=10,
                 **kwargs):
        self.length = length
        self.scale = scale
        self.min_text_size = min_text_size

    def cal_distance(self, x1, y1, x2, y2):
        '''calculate the Euclidean distance'''
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def move_points(self, vertices, index1, index2, r, coef):
        '''move the two points to shrink edge
        Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
            index1  : offset of point1
            index2  : offset of point2
            r       : [r1, r2, r3, r4] in paper
            coef    : shrink ratio in paper
        Output:
            vertices: vertices where one edge has been shinked
        '''
        index1 = index1 % 4
        index2 = index2 % 4
        x1_index = index1 * 2 + 0
        y1_index = index1 * 2 + 1
        x2_index = index2 * 2 + 0
        y2_index = index2 * 2 + 1

        r1 = r[index1]
        r2 = r[index2]
        length_x = vertices[x1_index] - vertices[x2_index]
        length_y = vertices[y1_index] - vertices[y2_index]
        length = self.cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
        if length > 1:
            ratio = (r1 * coef) / length
            vertices[x1_index] += ratio * (-length_x)
            vertices[y1_index] += ratio * (-length_y)
            ratio = (r2 * coef) / length
            vertices[x2_index] += ratio * length_x
            vertices[y2_index] += ratio * length_y
        return vertices


    def shrink_poly(self, vertices, coef=0.3):
        '''shrink the text region
        Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
            coef    : shrink ratio in paper
        Output:
            v       : vertices of shrinked text region <numpy.ndarray, (8,)>
        '''
        x1, y1, x2, y2, x3, y3, x4, y4 = vertices
        r1 = min(self.cal_distance(x1, y1, x2, y2), self.cal_distance(x1, y1, x4, y4))
        r2 = min(self.cal_distance(x2, y2, x1, y1), self.cal_distance(x2, y2, x3, y3))
        r3 = min(self.cal_distance(x3, y3, x2, y2), self.cal_distance(x3, y3, x4, y4))
        r4 = min(self.cal_distance(x4, y4, x1, y1), self.cal_distance(x4, y4, x3, y3))
        r = [r1, r2, r3, r4]

        # obtain offset to perform move_points() automatically
        if self.cal_distance(x1, y1, x2, y2) + self.cal_distance(x3, y3, x4, y4) > \
                self.cal_distance(x2, y2, x3, y3) + self.cal_distance(x1, y1, x4, y4):
            offset = 0  # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
        else:
            offset = 1  # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

        v = vertices.copy()
        v = self.move_points(v, 0 + offset, 1 + offset, r, coef)
        v = self.move_points(v, 2 + offset, 3 + offset, r, coef)
        v = self.move_points(v, 1 + offset, 2 + offset, r, coef)
        v = self.move_points(v, 3 + offset, 4 + offset, r, coef)
        return v

    def get_boundary(self, vertices):
        '''get the tight boundary around given vertices
        Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
        Output:
            the boundary
        '''
        x1, y1, x2, y2, x3, y3, x4, y4 = vertices
        x_min = min(x1, x2, x3, x4)
        x_max = max(x1, x2, x3, x4)
        y_min = min(y1, y2, y3, y4)
        y_max = max(y1, y2, y3, y4)
        return x_min, x_max, y_min, y_max

    def cal_error(self, vertices):
        '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
        calculate the difference between the vertices orientation and default orientation
        Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
        Output:
            err     : difference measure
        '''
        x_min, x_max, y_min, y_max = self.get_boundary(vertices)
        x1, y1, x2, y2, x3, y3, x4, y4 = vertices
        err = self.cal_distance(x1, y1, x_min, y_min) + self.cal_distance(x2, y2, x_max, y_min) + \
              self.cal_distance(x3, y3, x_max, y_max) + self.cal_distance(x4, y4, x_min, y_max)
        return err

    def rotate_vertices(self, vertices, theta, anchor=None):
        '''rotate vertices around anchor
        Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
            theta   : angle in radian measure
            anchor  : fixed position during rotation
        Output:
            rotated vertices <numpy.ndarray, (8,)>
        '''
        v = vertices.reshape((4, 2)).T
        if anchor is None:
            anchor = v[:, :1]
        rotate_mat = self.get_rotate_mat(theta)
        res = np.dot(rotate_mat, v - anchor)
        return (res + anchor).T.reshape(-1)

    def find_min_rect_angle(self, vertices):
        '''find the best angle to rotate poly and obtain min rectangle
        Input:
            vertices: vertices of text region <numpy.ndarray, (8,)>
        Output:
            the best angle <radian measure>
        '''
        angle_interval = 1
        angle_list = list(range(-90, 90, angle_interval))
        area_list = []
        for theta in angle_list:
            rotated = self.rotate_vertices(vertices, theta / 180 * math.pi)
            x1, y1, x2, y2, x3, y3, x4, y4 = rotated
            temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                        (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
            area_list.append(temp_area)

        sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
        min_error = float('inf')
        best_index = -1
        rank_num = 10
        # find the best angle with correct orientation
        for index in sorted_area_index[:rank_num]:
            rotated = self.rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
            temp_error = self.cal_error(rotated)
            if temp_error < min_error:
                min_error = temp_error
                best_index = index

        return angle_list[best_index] / 180 * math.pi

    def get_rotate_mat(self, theta):
        '''positive theta value means rotate clockwise'''
        return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])



    def rotate_all_pixels(self, rotate_mat, anchor_x, anchor_y, length):
        '''get rotated locations of all pixels for next stages
        Input:
            rotate_mat: rotatation matrix
            anchor_x  : fixed x position
            anchor_y  : fixed y position
            length    : length of image
        Output:
            rotated_x : rotated x positions <numpy.ndarray, (length,length)>
            rotated_y : rotated y positions <numpy.ndarray, (length,length)>
        '''
        x = np.arange(length)
        y = np.arange(length)
        x, y = np.meshgrid(x, y)
        x_lin = x.reshape((1, x.size))
        y_lin = y.reshape((1, x.size))
        coord_mat = np.concatenate((x_lin, y_lin), 0)
        rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                        np.array([[anchor_x], [anchor_y]])
        rotated_x = rotated_coord[0, :].reshape(x.shape)
        rotated_y = rotated_coord[1, :].reshape(y.shape)
        return rotated_x, rotated_y

    def __call__(self, data):

        img = data['image']
        text_polys = data['polys']
        text_tags = data['ignore_tags']
        if img is None:
            return None
        if text_polys.shape[0] == 0:
            return None
        # 随机裁剪和增强独立出去了
        ori_h, ori_w = img.shape[:2]

        ignored_polys = []
        polys = []
        index = np.arange(0, self.length, int(1 / self.scale))
        index_x, index_y = np.meshgrid(index, index)
        score_map = np.zeros((int(ori_h * self.scale), int(ori_w * self.scale), 1), np.float32)
        geo_map = np.zeros((int(ori_h * self.scale), int(ori_w * self.scale), 5), np.float32)
        ignored_map = np.zeros((int(ori_h * self.scale), int(ori_w * self.scale), 1), np.float32)

        for i, (vertice, tag) in enumerate(zip(text_polys, text_tags)):
            vertice = vertice.flatten()
            if tag:
                ignored_polys.append(np.around(self.scale * vertice.reshape((4, 2))).astype(np.int32))
                continue

            ## 先进行了shrink 在*0.25，得到1/4的输出
            poly = np.around(self.scale * self.shrink_poly(vertice).reshape((4, 2))).astype(np.int32)
            polys.append(poly) # scaled & shrinked

            # if the poly is too small, then ignore it during training
            poly_h = min(
                np.linalg.norm(poly[0] - poly[3]),
                np.linalg.norm(poly[1] - poly[2]))
            poly_w = min(
                np.linalg.norm(poly[0] - poly[1]),
                np.linalg.norm(poly[2] - poly[3]))
            if min(poly_h, poly_w) < self.min_text_size:
                ignored_polys.append(np.around(self.scale * vertice.reshape((4, 2))).astype(np.int32))

            # score map
            temp_mask = np.zeros(score_map.shape[:-1], np.float32)
            cv2.fillPoly(temp_mask, [poly], 1)

            theta = self.find_min_rect_angle(vertice)
            rotate_mat = self.get_rotate_mat(theta)
            rotated_vertices = self.rotate_vertices(vertice, theta)
            x_min, x_max, y_min, y_max = self.get_boundary(rotated_vertices)
            rotated_x, rotated_y = self.rotate_all_pixels(rotate_mat, vertice[0], vertice[1], self.length)

            d1 = rotated_y - y_min
            d1[d1 < 0] = 0
            d2 = y_max - rotated_y
            d2[d2 < 0] = 0
            d3 = rotated_x - x_min
            d3[d3 < 0] = 0
            d4 = x_max - rotated_x
            d4[d4 < 0] = 0

            geo_map[:, :, 0] += d1[index_y, index_x] * temp_mask
            geo_map[:, :, 1] += d2[index_y, index_x] * temp_mask
            geo_map[:, :, 2] += d3[index_y, index_x] * temp_mask
            geo_map[:, :, 3] += d4[index_y, index_x] * temp_mask
            geo_map[:, :, 4] += theta * temp_mask

        cv2.fillPoly(ignored_map, ignored_polys, 1)
        cv2.fillPoly(score_map, polys, 1)

        data['image'] = img
        data['score_map'] = torch.Tensor(score_map).permute(2, 0, 1)
        data['geo_map'] = torch.Tensor(geo_map).permute(2, 0, 1)
        data['training_mask'] = torch.Tensor(ignored_map).permute(2, 0, 1)

        return data
