# coding=utf-8  
# @Time   : 2020/12/29 18:08
# @Auto   : zzf-jeff

algorithm = 'c2td'

model = dict(
    type='DetectionModel',
    transform=None,
    backbone=dict(
        type='VGGPixelWithDilation',
        in_channels=3,
    ),
    neck=dict(
        type='C2TDWithUnet',
        in_channels=[256, 512, 512, 1024],
        out_channels=3
    ),
    head=dict(
        type='C2TDHead',
        in_channels=3,
    ),
)

# todo:train_pipeline 这块对于map图的处理会不会太死了
# IaaAugment这里直接延用原作代码
train_pipeline = [
    dict(type='DecodeImage', img_mode='BGR', channel_first=False),
    dict(type='DetLabelEncode', ignore_tags=['*', '###']),
    dict(type='IaaAugment',
         augmenter_args=[
             # dict(type='Fliplr', args=dict(p=0.5)),
             # dict(type='Affine', args=dict(rotate=[-10, 10])),
             dict(type='Resize', args=dict(size=[0.5, 2])),
         ]
         ),
    dict(type='EastRandomCropData',
         size=[640, 640], max_tries=50, min_crop_side_ratio=0.1, keep_ratio=True),
    dict(type='C2TDProcessTrain', crop_size=[640, 640], scale=0.25),
    dict(type='NormalizeImage', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='ToCHWImage'),
    dict(type='KeepKeys', keep_keys=['image', 'heat_map']),
]

test_pipeline = [
    dict(type='DecodeImage', img_mode='BGR', channel_first=False),
    dict(type='DetLabelEncode', ignore_tags=['*', '###']),
    dict(type='DetResizeForTest', image_shape=[736, 1280],mode=''),
    dict(type='NormalizeImage', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='ToCHWImage'),
    dict(type='KeepKeys', keep_keys=['image', 'shape', 'polys', 'ignore_tags']),
]

train_data = dict(
    dataset=dict(
        type='DetTextICDAR15',
        # ann_file=r'/media/newData/user/zzf/data/icdar2015/train/train_list.txt',
        ann_file=r'/media/newData/user/zzf/data/icdar2015/test/test_list.txt',
        pipeline=train_pipeline,
        mode='train',
        data_root=None
    ),
    loader=dict(
        batch_size=8,  # dist时，实际上的gpu应该是number*batch_size
        num_workers=4,  # dist时，num_workers注意不要超出主机cpu
        shuffle=True,  # dist，已经在load_data时做了shuffle
        drop_last=True,
        pin_memory=False,
        collate_fn=None,
    )
)

test_data = dict(
    dataset=dict(
        type='DetTextICDAR15',
        ann_file=r'/media/newData/user/zzf/data/icdar2015/test/test_list.txt',
        pipeline=test_pipeline,
        mode='test',
        data_root=None
    ),
    loader=dict(
        batch_size=1,
        num_workers=0,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
        collate_fn=None,
    )
)

# 学习率优化设置 默认StepLR
# lr_scheduler = dict(type='StepLR', step_size=50, gamma=0.1)
lr_scheduler = dict(type='CosineWarmup', warm_up_epochs=3, epochs=800)
# 优化器设置 默认SGD
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=1e-4)
# loss设置
loss = dict(
    type='C2TDLoss',
    thresh=0.5,
    neg_pos=3
)
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
    gpu_ids='3',  # gup ids,
    total_epochs=800,  # 训练epoch大小,
    work_dir=None,  # 模型保存文件目录，包含日志文件
    load_from=None,  # 用于加载已训练完模型，用于用较低学习率微调网络
    resume_from=None,  # 用于程序以外中断，继续训练
    is_eval=False,
    eval_batch_step=[4000, 5000],
    # 验证集配置，根据统计指标计算，默认给保存最好的模型
    checkpoint_interval_epoch=50,  # 模型保存策略，默认每个epoch都保存
    save_best_checkpoint=True,
    print_batch_step=50,  # step为单位
    log_smooth_window=20,
    seed=2021,  # 随机种子
)
