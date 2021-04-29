# coding=utf-8  
# @Time   : 2021/2/3 18:01
# @Auto   : zzf-jeff

'''
modify for
https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py
'''
import argparse
import os
import lmdb
import cv2
from tqdm import tqdm
import numpy as np
import pathlib


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(data_list, lmdb_save_path, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        data_list  : a list contains img_path\tlabel
        lmdb_save_path : LMDB output path
        checkValid : if true, check the validity of every image
    """
    os.makedirs(lmdb_save_path, exist_ok=True)
    # #1099511627776所需要的磁盘空间的最小值，1T
    env = lmdb.open(lmdb_save_path, map_size=1099511627776)
    cache = {}
    cnt = 1
    for imagePath, label in tqdm(data_list, desc='make dataset, save to {}'.format(lmdb_save_path)):
        # print(imagePath, label)
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--label_path', nargs='?', type=str, default=None)
    parser.add_argument('--image_path', nargs='?', type=str, default=None)
    parser.add_argument('--lmdb_save_path', nargs='?', type=str, default=None)
    args = parser.parse_args()

    image_path = args.image_path
    label_file = args.label_path
    lmdb_save_path = args.lmdb_save_path

    os.makedirs(lmdb_save_path, exist_ok=True)
    data_list = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc='load data from {}'.format(label_file)):
            line = line.strip('\n').split(' ')
            if len(line) > 1:
                img_path = os.path.join(image_path, line[0])
                label = line[1]
                data_list.append((img_path, label))

    createDataset(data_list, lmdb_save_path)
