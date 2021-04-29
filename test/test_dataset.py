# coding=utf-8  
# @Time   : 2020/12/29 18:08
# @Auto   : zzf-jeff
import sys
import torch
from torchsummary import summary

sys.path.append('./')
from torchocr.datasets import build_dataloader, build_dataset
import torch
from torch.utils.data import DataLoader

train_pipeline = [
    dict(type='DecodeImage', img_mode='BGR', channel_first=False),
    dict(type='DetLabelEncode', ignore_tags=['*', '###']),
    dict(type='IaaAugment',
         augmenter_args=[
             dict(type='Fliplr', args=dict(p=0.5)),
             dict(type='Affine', args=dict(rotate=[-10, 10])),
             dict(type='Resize', args=dict(size=[0.5, 3.0])),
         ]
         ),
    dict(type='PixelLinkProcessTrain',
         size=[512, 512], max_tries=50, min_crop_side_ratio=0.1, keep_ratio=True,
         num_neighbours=4,
         ),
    dict(type='NormalizeImage', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='ToCHWImage'),
    dict(type='KeepKeys',
         keep_keys=['image', 'cls_label', 'cls_weight', 'link_label', 'link_weight']),
]

# train_pipeline = [
#     dict(type='DecodeImage', img_mode='BGR', channel_first=False),
#     dict(type='DetLabelEncode', ignore_tags=['*', '###']),
#     dict(type='IaaAugment',
#          augmenter_args=[
#              dict(type='Fliplr', args=dict(p=0.5)),
#              dict(type='Affine', args=dict(rotate=[-10, 10])),
#              dict(type='Resize', args=dict(size=[0.5, 3])),
#          ]
#          ),
#     dict(type='EastRandomCropData',
#          size=[640, 640], max_tries=50, min_crop_side_ratio=0.1, keep_ratio=True),
#     dict(type='MakeBorderMap', shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7),
#     dict(type='MakeShrinkMap', min_text_size=8, shrink_ratio=0.4),
#     dict(type='NormalizeImage', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     dict(type='ToCHWImage'),
#     dict(type='KeepKeys', keep_keys=['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask']),
# ]

train_data = dict(
    dataset=dict(
        type='DetTextDataset',
        ann_file=r'/media/newData/user/zzf/data/icdar2015/test/test_list.txt',
        pipeline=train_pipeline,
    ),
    loader=dict(
        batch_size=1,
        num_workers=0,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        pin_memory=False,
    )
)

dataset = build_dataset(cfg=train_data['dataset'])

from collections import defaultdict


def collate(batch):
    ## 对于分割等有多个map图的,没有totensor,也要stack
    if len(batch) == 0:
        return None
    clt = defaultdict(list)
    for i, dic in enumerate(batch):
        clt['idx'].append(torch.tensor(i))
        for k, v in dic.items():
            clt[k].append(v)

    for k, v in clt.items():
        if isinstance(clt[k][0], (torch.Tensor)):
            clt[k] = torch.stack(v, 0)

    # collate = default_collate(batch)
    return clt


train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True,
                          collate_fn=None)

from torchocr.utils.vis import show_img

for i, data in enumerate(train_loader):
    img = data['image']
    cls_weight = data['cls_weight']
    cls_label = data['cls_label']
    link_label = data['link_label']
    link_weight = data['link_weight']
    # geo_map = data['geo_map']
    # training_mask = data['training_mask']
    #
    # print(cls_label)
    print(img.shape)
    print(cls_weight.shape)
    print(cls_label.shape)
    print(link_label.shape)
    print(link_weight.shape)
    #
    # show_img((cls_weight[0].numpy()), title='shrink_map')
    # show_img(cls_label[0].numpy(), title='img')
    # link_label = link_label.transpose(2, 3).transpose(1, 2)
    # print(link_label.shape)

    break
