# coding=utf-8  
# @Time   : 2020/12/29 18:08
# @Auto   : zzf-jeff

model = dict(
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
    )
)


train_pipeline = [
    # dict(type='Fliplr', p=0.5),
    # dict(type='Affine', rotate=[-10, 10]),
    # dict(type='Resize', size=[0.5, 3]),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
val_pipeline = [
    # dict(type='Fliplr', p=0.5),
    # dict(type='Affine', rotate=[-10, 10]),
    # dict(type='Resize', size=[0.5, 3]),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
test_pipeline = [
    # dict(type='Fliplr', p=0.5),
    # dict(type='Affine', rotate=[-10, 10]),
    # dict(type='Resize', size=[0.5, 3]),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]




data = dict(
    batch_size=64,
    workers_per_gpu=1,
    train = dict(
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
            min_crop_side_ratio=0.1
        )
    ),
    val=dict(
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
            min_crop_side_ratio=0.1
        )
    ),
    test=dict(
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
            min_crop_side_ratio=0.1
        )
    ),
)
