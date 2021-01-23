# coding=utf-8  
# @Time   : 2020/12/29 18:08
# @Auto   : zzf-jeff

algorithm = 'PixelLink-4s'

model = dict(
    type='DetectionModel',
    transform=None,
    backbone=dict(
        type='DetResNet',
        in_channels=3,
        depth=50
    ),
    neck=dict(
        type='PixelWithUnet',
        in_channels=[256, 512, 1024, 2048],
        num_neighbours=4
    ),
    head=dict(
        type='PixelHead',
        num_neighbours=4
    ),
)

# todo:train_pipeline 这块对于map图的处理会不会太死了
# IaaAugment这里直接延用原作代码
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
         keep_keys=['image', 'pixel_cls_label', 'pixel_cls_weight', 'pixel_link_label', 'pixel_link_weight']),
]

test_pipeline = [
    dict(type='DecodeImage', img_mode='BGR', channel_first=False),
    dict(type='DetLabelEncode', ignore_tags=['*', '###']),
    dict(type='KeepKeys', keep_keys=['image']),
]

train_data = dict(
    dataset=dict(
        type='DetTextDataset',
        ann_file=r'/zzf/data/icdar2015/train/train_list.txt',
        pipeline=train_pipeline,
    ),
    loader=dict(
        batch_size=4,
        num_workers=4,
        collate_fn=None,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )
)

test_data = dict(
    dataset=dict(
        type='DetTextDataset',
        ann_file=r'/zzf/data/icdar2015/test/test_list.txt',
        pipeline=test_pipeline,
    ),
    loader=dict(
        batch_size=1,
        num_workers=4,
        shuffle=False,
        collate_fn=None,
        drop_last=True,
        pin_memory=False,
    )
)

# 学习率优化设置 默认StepLR
lr_scheduler = dict(type='StepLR', step_size=50, gamma=0.1)
# lr_scheduler = dict(type='CosineWarmup', warm_up_epochs=3, epochs=50)
# 优化器设置 默认SGD
optimizer = dict(type='SGD', lr=0.001, momentum=0.99, weight_decay=5e-4)
# loss设置
loss = dict(type='EASTLoss')
# 后处理设置
postprocess = dict(
    type='DBPostProcess',
    thresh=0.3,
    box_thresh=0.5,
    max_candidates=1000,
    unclip_ratio=1.5
)
metric = dict(type='PolygonMetric', main_indicator='hmean')

options = dict(
    gpu_ids='2',  # gup ids,
    total_epochs=1000,  # 训练epoch大小,
    work_dir=None,  # 模型保存文件目录，包含日志文件
    load_from=None,  # 用于加载已训练完模型，用于用较低学习率微调网络
    resume_from='',  # 用于程序以外中断，继续训练
    is_eval=False,
    eval_batch_step=[0, 100],
    # 验证集配置，根据统计指标计算，默认给保存最好的模型
    checkpoint_interval_epoch=50,  # 模型保存策略，默认每个epoch都保存
    save_best_checkpoint=True,
    print_batch_step=50,  # step为单位
    log_smooth_window=20,
    seed=2021,  # 随机种子
)
