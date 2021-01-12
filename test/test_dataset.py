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
    dict(type='EASTProcessTrain',
         image_shape=[512, 512], background_ratio=0.125, min_crop_side_ratio=0.1, min_text_size=10),
    dict(type='KeepKeys', keep_keys=['image', 'score_map', 'geo_map', 'training_mask']),
]

train_data = dict(
    dataset=dict(
        type='DetTextDataset',
        ann_file=r'/zzf/data/icdar2015/test/test_list.txt',
        pipeline=train_pipeline,
    ),
    loader=dict(
        batch_size=1,
        num_workers=4,
        workers_per_gpu=1,
        shuffle=True,
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


for idx in range(100):
    for i, data in enumerate(train_loader):

        img = data['image']
        score_map = data['score_map']
        geo_map = data['geo_map']
        training_mask = data['training_mask']

        print(img.shape)
        print(score_map.shape)
        print(geo_map.shape)
        print(training_mask.shape)

        show_img(img[0].numpy().transpose(1, 2, 0), title='img')
        show_img(score_map[0].numpy().transpose(1, 2, 0), title='score_map')
        break

