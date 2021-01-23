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
from torchocr.converters import build_converter
from torchocr.models import build_rec
from torchocr.models import build_det
from torchocr.optimizers import build_optimizer
from torchocr.lr_schedulers import build_lr_scheduler

model = dict(
    type='CRNN',
    backbone=dict(
        type='RecResNet',
        in_channels=1,
        depth=18
    ),
    neck=dict(
        type='EncodeWithLSTM',
        num_lstm=2,
        in_channels=512,
        hidden_channel=256
    ),
    head=dict(
        type='CTCHead',
        in_channels=512,
        n_class=10
    )
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.99, weight_decay=5e-4)  # 优化器 默认SGD
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2),
                        detect_anomaly=False)  # 优化器配置项 grad_clip:梯度剪切，防止梯度爆炸或消失 detect_anomaly :该参数在debug模式下打开，可帮助调试代码，默认关闭，影响模型训练速度。

cfg_loss = dict(
    type='CTCLoss'
)

cfg_converter = dict(
    type='CTCConverter',
    alphabet_path='scripts/test.txt'
)

lr = dict(
    type='StepLR',
    step_size=10,
    gamma=0.1
)

device = torch.device('cuda:0')
input_data = torch.randn(1, 1, 32, 100).to(device)

# model = build_backbone(cfg=cfg_model).to(device)
# crnn_head = build_head(cfg=cfg_head).to(device)
# print(model)
# x = model(input_data)
# y = crnn_head(x)
# print(y.shape)

#

model = build_rec(cfg=model)
print(model)
model = model.to(device)
y = model(input_data)
print(y.shape)
optimizer = build_optimizer(optimizer,model)
lr_scheduler = build_lr_scheduler(lr)(optimizer)
print(optimizer)
print(lr_scheduler)
loss = build_loss(cfg_loss)
print(loss)
converter = build_converter(cfg=cfg_converter)
test = '123'
print(converter.decode(torch.IntTensor([2, 3, 4]), torch.IntTensor([1, 1, 1])))
print(converter.encode(test))


