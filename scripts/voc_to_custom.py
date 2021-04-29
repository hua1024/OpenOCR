# coding=utf-8  
# @Time   : 2021/2/24 10:53
# @Auto   : zzf-jeff

import os
import argparse
import glob


def transform_voc_data(args):
    origin_path = args.origin_path
    transform_path = args.transform_path
    if not os.path.exists(transform_path):
        os.makedirs(transform_path)
    origin_label_list = glob.glob(os.path.join(origin_path, '*.txt'))
    for file in origin_label_list:
        basename = os.path.basename(file)
        new_path = os.path.join(transform_path, basename)
        with open(file, 'r', encoding='utf-8') as fr:
            with open(new_path, 'a+', encoding='utf-8') as fw:
                lines = fr.readlines()
                for line in lines:
                    line = line.strip('\n')
                    result = line.split(',')[:4]
                    result = list(map(int, result))
                    tmp_h = result[3] - result[1]
                    tmp_w = result[2] - result[0]
                    x1, y1, x2, y2, x3, y3, x4, y4 = result[0], result[1], result[0] + tmp_w, result[1], result[2], \
                                                     result[3], result[0], result[1] + tmp_h
                    fw_str = ('{},' * 8).format(x1, y1, x2, y2, x3, y3, x4, y4)
                    fw.write(fw_str + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--origin_path', nargs='?', type=str, default=None)
    parser.add_argument('--transform_path', nargs='?', type=str, default=None)
    # parser.add_argument('--mode', nargs='?', type=str, default='train')
    args = parser.parse_args()
    transform_voc_data(args)
