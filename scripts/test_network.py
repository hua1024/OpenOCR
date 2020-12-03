# coding=utf-8  
# @Time   : 2020/12/1 14:55
# @Auto   : zzf-jeff

import sys
import torch
from torchsummary import summary

sys.path.append('./')

from torchocr.models import build_backbone
from torchocr.models import build_head
from torchocr.models import build_loss
from torchocr.converter import build_converter
from torchocr.models import build_rec
from torchocr.optimizers import build_optimizer
from torchocr.lr_schedulers import build_lr_scheduler

cfg_model = dict(
    type='RecResNet',
    in_channels=1,
    num_classes=100,
    depth=50
)

cfg_head = dict(
    type='CRNNHead',
    use_lstm=True,
    lstm_num=3,
    use_conv=False,
    use_attention=False,
    in_channel=2048,
    hidden_channel=1024,
    classes=10
)

cfg_loss = dict(
    type='CRNNHead',
    use_lstm=True,
    lstm_num=3,
    use_conv=False,
    use_attention=False,
    in_channel=2048,
    hidden_channel=1024,
    classes=10
)

cfg_converter = dict(
    type='CTCConverter',
    alphabet_path='test.txt'
)

rec_model = dict(
    type='OCRRecognition',
    backbone=dict(
        type='RecResNet',
        in_channels=1,
        num_classes=100,
        depth=50
    ),
    neck=None,
    head=dict(
        type='CRNNHead',
        use_lstm=True,
        lstm_num=3,
        use_conv=False,
        use_attention=False,
        in_channel=2048,
        hidden_channel=1024,
        classes=10
    )
)

cfg_optimizer = dict(
    type='SGDDecay',
    params=dict(
        weight_decay=1e4,
        momentum=0.9
    )
)

cfg_lr = dict(
    type='StepLR',
    params=dict(
        step_size=10,
        gamma=0.1
    )
)

device = torch.device('cuda:0')
input_data = torch.randn(1, 1, 32, 100).to(device)

# model = build_backbone(cfg=cfg_model).to(device)
# crnn_head = build_head(cfg=cfg_head).to(device)
# print(model)
# x = model(input_data)
# y = crnn_head(x)
# print(y.shape)

# converter = build_converter(cfg=cfg_converter)
# test = '123'
# print(converter.decode(torch.IntTensor([2,3,4]),torch.IntTensor([1,1,1])))
# print(converter.encode(test))
model = build_rec(cfg=rec_model)
model = model.to(device)
y = model(input_data)
optimizer = build_optimizer(cfg_optimizer)(model, 0.01)
lr_scheduler = build_lr_scheduler(cfg_lr)(optimizer)
print(optimizer)
print(lr_scheduler)
print(y.shape)

# print(model)
