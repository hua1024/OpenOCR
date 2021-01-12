# coding=utf-8  
# @Time   : 2020/12/1 14:55
# @Auto   : zzf-jeff

import sys
import torch
from torchsummary import summary
import os
sys.path.append('./')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torchocr.models import build_model
from torchocr.models import build_head

device = torch.device('cuda:0')

det_data = torch.randn(1, 3, 640, 640).to(device)

db_model = dict(
    type='DetectionModel',
    transform=None,
    backbone=dict(
        type='DetResNet',
        in_channels=3,
        depth=50
    ),
    neck=dict(
        type='EASTWithUnet',
        in_channels=[256, 512, 1024, 2048],
        out_channels=128
    ),
    # head=dict(
    #     type='C2TDHead',
    #     in_channels=3,
    # ),
    head=None

)

det_model = build_model(cfg=db_model)
# print(det_model)
det_model = det_model.to(device)
y = det_model(det_data)

print(y.shape)
