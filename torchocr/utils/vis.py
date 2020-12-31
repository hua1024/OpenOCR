# coding=utf-8  
# @Time   : 2020/12/30 10:27
# @Auto   : zzf-jeff
import numpy as np
import cv2

def show_img(imgs, title='img'):
    from matplotlib import pyplot as plt
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    imgs = np.expand_dims(imgs, axis=0)

    for i, img in enumerate(imgs):
        plt.figure()
        plt.title('{}_{}'.format(title, i))
        plt.imshow(img, cmap=None if color else 'gray')
    plt.savefig('{}.png'.format(title))
    # cv2.imwrite('{}.png'.format(title),imgs)