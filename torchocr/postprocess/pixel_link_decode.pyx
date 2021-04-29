import numpy as np

def get_neighbours_4(x, y):
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]

def get_neighbours_8(x, y):
    """
    Get 8 neighbours of point(x, y)
    """
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
            (x - 1, y), (x + 1, y), \
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >= 0 and x < w and y >= 0 and y < h;

def decode_image_by_join(pixel_scores, link_scores,
                         pixel_conf_threshold, link_conf_threshold, neighbor_num):
    pixel_mask = pixel_scores >= pixel_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    points = list(zip(*np.where(pixel_mask)))
    h, w = np.shape(pixel_mask)
    group_mask = dict.fromkeys(points, -1)  # 并查集算法中的MAKE-SET(x)，创建一个只有元素x的集合，且x不应出现在其他的集合中
    def find_parent(point):

        return group_mask[point]

    def set_parent(point, parent):
        group_mask[point] = parent

    def is_root(point):
        return find_parent(point) == -1

    def find_root(point):
        root = point
        update_parent = False
        swap_point = []

        while not is_root(root):
            swap_point.append(root)
            root = find_parent(root)
            update_parent = True
        # for acceleration of find_root  路径压缩, 经过的点的父节点设置为根节点, 成两级结构
        if update_parent:
            for _ in swap_point:
                set_parent(_, root)
        return root

    def join(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)
        if root1 != root2:
            set_parent(root1, root2)

    def get_all():
        root_map = {}
        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]
        mask = np.zeros_like(pixel_mask, dtype=np.int16)
        for point in points:
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask

    for point in points:
        y, x = point
        if neighbor_num == 4:
            neighbours = get_neighbours_4(x, y)
        else:
            neighbours = get_neighbours_8(x, y)
        for n_idx, (nx, ny) in enumerate(neighbours):
            if is_valid_cord(nx, ny, w, h):

                link_value = link_mask[y, x, n_idx]  # and link_mask[ny, nx, reversed_idx]
                pixel_cls = pixel_mask[ny, nx]
                if link_value and pixel_cls:
                    join(point, (ny, nx))

    return get_all()
