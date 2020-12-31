# coding=utf-8  
# @Time   : 2020/12/29 18:08
# @Auto   : zzf-jeff
import sys
import torch
from torchsummary import summary

sys.path.append('./')
from torchocr.datasets import build_det_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchocr.utils.vis import show_img
from torchocr.models import build_det
from torchocr.optimizers import build_optimizer
from torchocr.lr_schedulers import build_lr_scheduler

train_pipeline = [
    # dict(type='Fliplr', p=0.5),
    # dict(type='Affine', rotate=[-10, 10]),
    # dict(type='Resize', size=[0.5, 3]),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

db_model = dict(
    type='DBNet',
    backbone=dict(
        type='DetResNet',
        in_channels=3,
        depth=50
    ),
    neck=dict(
        type='DB_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256
    ),
    head=dict(
        type='DBHead',
        in_channels=256,
        k=50
    ),
    loss=dict(
        type='DBLoss',
    ),
    postprocess=dict(
        type='DBPostProcess'
    )

)

data_cfg = dict(
    type='DBDataset',
    data_root=r'data/mscoco',
    ann_file=r'/zzf/data/icdar2015/test/test_list.txt',
    img_prefix='',
    pipeline=train_pipeline,
    ignore_tags='#',
    # label制作的参数
    pre_params=dict(
        shrink_ratio=0.4,
        thresh_min=0.3,
        thresh_max=0.7,
        min_text_size=8,
        size=(640, 640),
        max_tries=50,
        min_crop_side_ratio=0.1,
    )
)


lr = dict(
    type='StepLR',
    step_size=10,
    gamma=0.1
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.99, weight_decay=5e-4)  # 优化器 默认SGD



device = torch.device('cuda:0')
det_model = build_det(cfg=db_model)
print(det_model)
det_model = det_model.to(device)

optimizer = build_optimizer(optimizer,det_model)
lr_scheduler = build_lr_scheduler(lr)(optimizer)

det_model.train()

dataset = build_det_dataset(cfg=data_cfg)
train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)


from collections import OrderedDict
import torch.distributed as dist
def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value if _loss is not None)
    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key and 'Y' not in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.detach().item()

    return loss, log_vars

for idx in range(100):
    for i, data in enumerate(train_loader):

        for key, value in data.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(device)

        optimizer.zero_grad()
        loss_dict = det_model(data)
        loss_dict['loss'].backward()
        optimizer.step()
        loss, log_vars = parse_losses(loss_dict)
        print(log_vars)
        # Update learning rate
        lr_scheduler.step()
        # img = data['img']
        # shrink_label = data['shrink_map']
        # threshold_label = data['threshold_map']
        # show_img(img[0].numpy().transpose(1, 2, 0), title='img')
        # show_img((shrink_label[0].to(torch.float)).numpy(), title='shrink_label')
        # break

state = {
    'state_dict': det_model.state_dict(),
    'lr': optimizer.param_groups[0]['lr'],
    'optimizer': optimizer.state_dict(),
}

torch.save(state, 'testdb_model')