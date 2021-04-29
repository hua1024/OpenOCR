# coding=utf-8  
# @Time   : 2020/12/29 18:08
# @Auto   : zzf-jeff

algorithm = 'PixelLink-4s'

model = dict(
    type='DetectionModel',
    transform=None,
    backbone=dict(
        type='VGGPixelWithDilation',
        in_channels=3,
    ),
    neck=dict(
        type='PixelWithUnet',
        in_channels=[256, 512, 512, 1024],
        num_neighbours=8
    ),
    head=dict(
        type='PixelHead',
        num_neighbours=8
    ),
)

train_pipeline = [
    dict(type='DecodeImage', img_mode='RGB', channel_first=False),
    dict(type='DetLabelEncode', ignore_tags=['*', '###']),
    dict(type='IaaAugment',
         augmenter_args=[
             dict(type='Fliplr', args=dict(p=0.5)),
             dict(type='Affine', args=dict(rotate=[-10, 10])),
             dict(type='Resize', args=dict(size=[0.5, 2.0])),
         ]
         ),
    dict(type='PixelLinkProcessTrain',
         size=[640, 640], max_tries=50, min_crop_side_ratio=0.1, keep_ratio=True,
         num_neighbours=8, scale=0.25
         ),
    dict(type='NormalizeImage', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='ToCHWImage'),
    dict(type='KeepKeys',
         keep_keys=['image', 'cls_label', 'cls_weight', 'link_label', 'link_weight']),
]

test_pipeline = [
    dict(type='DecodeImage', img_mode='RGB', channel_first=False),
    dict(type='DetLabelEncode', ignore_tags=['*', '###']),
    dict(type='DetResizeForTest', image_shape=[736, 1280], mode=''),
    dict(type='NormalizeImage', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='ToCHWImage'),
    dict(type='KeepKeys', keep_keys=['image', 'shape', 'polys', 'ignore_tags']),
]

train_data = dict(
    dataset=dict(
        type='DetTextICDAR15',
        ann_file=r'/zzf/data/icdar2015/train/train_list.txt',
        # ann_file=r'/zzf/data/icdar2015/test/test_list.txt',
        pipeline=train_pipeline,
        mode='train',
        data_root=None
    ),
    loader=dict(
        batch_size=4,
        num_workers=2,
        collate_fn=None,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )
)

test_data = dict(
    dataset=dict(
        type='DetTextICDAR15',
        ann_file=r'/zzf/data/icdar2015/test/test_list.txt',
        pipeline=test_pipeline,
        mode='test',
        data_root=None
    ),
    loader=dict(
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=None,
        drop_last=True,
        pin_memory=False,
    )
)

# 学习率优化设置 默认StepLR
# lr_scheduler = dict(type='StepLR', step_size=50, gamma=0.1)
lr_scheduler = dict(type='CosineWarmup', warm_up_epochs=4, epochs=800)
# 优化器设置 默认SGD
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=1e-4)
# loss设置
loss = dict(
    type='PixelLinkLoss',
    num_neighbours=8,
)
# 后处理设置
postprocess = dict(
    type='PixelLinkPostProcess',
    neighbor_num=8,
    pixel_conf=0.7,
    link_conf=0.5,
    min_area=300,
    min_height=10
)
metric = dict(type='PolygonMetric', main_indicator='hmean')

options = dict(
    gpu_ids='0',  # gup ids,
    total_epochs=800,  # 训练epoch大小,
    work_dir=None,  # 模型保存文件目录，包含日志文件
    load_from=None,  # 用于加载已训练完模型，用于用较低学习率微调网络
    resume_from=None,  # 用于程序以外中断，继续训练
    is_eval=True,
    eval_batch_step=[2000, 2000],
    # 验证集配置，根据统计指标计算，默认给保存最好的模型
    checkpoint_interval_epoch=50,  # 模型保存策略，默认每个epoch都保存
    save_best_checkpoint=True,
    print_batch_step=50,  # step为单位
    log_smooth_window=20,
    seed=6,  # 随机种子
)
