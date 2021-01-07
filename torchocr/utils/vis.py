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


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    import cv2
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path