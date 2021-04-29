# coding=utf-8  
# @Time   : 2020/12/29 18:32
# @Auto   : zzf-jeff


import os
import argparse


def gen_train_file(args):
    label_path = args.label_path
    img_path = args.img_path
    files = os.listdir(img_path)

    if args.label_type == 'txt':
        with open(os.path.join(args.save_path, '{}_list.txt'.format(args.mode)), 'w+', encoding='utf-8') as fid:
            for file in files:
                label_str = '{},{}'.format(os.path.join(img_path, file),
                                           os.path.join(label_path, os.path.splitext(file)[0] + '.txt')) + '\n'
                fid.write(label_str)
    else:
        with open(os.path.join(args.save_path, '{}_list.txt'.format(args.mode)), 'w+', encoding='utf-8') as fid:
            for file in files:
                label_str = '{},{}'.format(os.path.join(img_path, file),
                                           os.path.join(label_path, os.path.splitext(file)[0] + '.json')) + '\n'
                fid.write(label_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--label_path', nargs='?', type=str, default=None)
    parser.add_argument('--img_path', nargs='?', type=str, default=None)
    parser.add_argument('--save_path', nargs='?', type=str, default=None)
    parser.add_argument('--label_type', nargs='?', type=str, default='txt')
    parser.add_argument('--mode', nargs='?', type=str, default=None)
    args = parser.parse_args()
    gen_train_file(args)
