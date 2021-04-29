# coding=utf-8  
# @Time   : 2020/12/29 18:08
# @Auto   : zzf-jeff

algorithm = 'DB'  # 算法名称

# 模型结构
model = dict(
    type='DetectionModel',
    transform=None,
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
)

# IaaAugment这里直接沿用原作代码
train_pipeline = [
    dict(type='DecodeImage', img_mode='RGB', channel_first=False),
    dict(type='DetLabelEncode', ignore_tags=['*', '###']),
    dict(type='IaaAugment',
         augmenter_args=[
             dict(type='Fliplr', args=dict(p=0.5)),
             dict(type='Affine', args=dict(rotate=[-10, 10])),
             dict(type='Resize', args=dict(size=[0.5, 2])),
         ]
         ),
    dict(type='EastRandomCropData',
         size=[640, 640], max_tries=50, min_crop_side_ratio=0.1, keep_ratio=True),
    dict(type='MakeBorderMap', shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7),
    dict(type='MakeShrinkMap', min_text_size=8, shrink_ratio=0.4),
    dict(type='NormalizeImage', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='ToCHWImage'),
    dict(type='KeepKeys', keep_keys=['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask']),
]

test_pipeline = [
    dict(type='DecodeImage', img_mode='RGB', channel_first=False),
    dict(type='DetLabelEncode', ignore_tags=['*', '###']),
    dict(type='DetResizeForTest', short_size=736, mode='db'),
    dict(type='NormalizeImage', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='ToCHWImage'),
    dict(type='KeepKeys', keep_keys=['image', 'shape', 'polys', 'ignore_tags']),
]

train_data = dict(
    dataset=dict(
        type='DetTextICDAR15',
        ann_file=r'/zzf/data/ocr_det_data/icdar2015/train/train_list.txt',
        # ann_file=r'/media/newData/user/zzf/data/ocr_det_data/icdar2015/test/test_list.txt',
        pipeline=train_pipeline,
        mode='train',
        data_root=None,
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
        ann_file=r'/zzf/data/ocr_det_data/icdar2015/test/test_list.txt',
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
# lr_scheduler = dict(type='DecayLearningRate', lr=0.007, epochs=1200)
lr_scheduler = dict(type='CosineWarmup', warm_up_epochs=5, epochs=1200)
# lr_scheduler = dict(type='MultiStepLR', milestones=[30,50,80], gamma=0.1)
# lr_scheduler = dict(type='MultiStepWarmup', milestones=[300, 500, 800], gamma=0.1, warm_up_epochs=3)

# 优化器设置 默认SGD
# optimizer = dict(type='Adam', lr=0.001, beta1=0.9, beta2=0.99, weight_decay=1e-5)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=1e-4)

# loss设置
loss = dict(
    type='DBLoss',
    alpha=1.0,
    beta=10,
    ohem_ratio=3
)
# 后处理设置
postprocess = dict(
    type='DBPostProcess',
    thresh=0.3,
    box_thresh=0.5,
    max_candidates=1000,
    unclip_ratio=1.5,
    is_polygon=False
)
# 测试设置
metric = dict(
    type='PolygonMetric',
    main_indicator='hmean'
)

# 训练超参
options = dict(
    gpu_ids='0',  # gup ids,
    total_epochs=1200,  # 训练epoch大小,
    work_dir=None,  # 模型保存文件目录，包含日志文件
    load_from=None,  # 用于加载已训练完模型，用于用较低学习率微调网络
    resume_from=None,  # 用于程序以外中断，继续训练
    is_eval=True,
    eval_batch_step=[3000, 3000],
    # 验证集配置，根据统计指标计算，默认给保存最好的模型
    checkpoint_interval_epoch=100,  # 模型保存策略，默认每个epoch都保存
    save_best_checkpoint=True,
    print_batch_step=50,  # step为单位
    log_smooth_window=20,
    seed=2021,  # 随机种子
)
