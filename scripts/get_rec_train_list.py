# coding=utf-8  
# @Time   : 2020/12/29 18:32
# @Auto   : zzf-jeff


import os
import argparse


def gen_train_file(args):
    label_path = args.label_path
    img_path = args.img_path
    files = os.listdir(img_path)

    with open(os.path.join(args.save_path, 'train_list.txt'), 'w+', encoding='utf-8') as fid:
        for file in files:
            basename = os.path.basename(file)
            label = basename.split('_')[0]

            label_str = '{},{}'.format(os.path.join(img_path, file),label) + '\n'
            fid.write(label_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--label_path', nargs='?', type=str, default='/zzf/data/captcha/data/train')
    parser.add_argument('--img_path', nargs='?', type=str, default='/zzf/data/captcha/data/train')
    parser.add_argument('--save_path', nargs='?', type=str, default='/zzf/data/captcha/data')
    args = parser.parse_args()
    gen_train_file(args)
