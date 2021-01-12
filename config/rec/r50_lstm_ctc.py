# coding=utf-8  
# @Time   : 2020/12/29 18:08
# @Auto   : zzf-jeff

algorithm = 'CRNN'

model = dict(
    type='RecognitionModel',
    transform=None,
    backbone=dict(
        type='RecResNet',
        in_channels=3,
        depth=18
    ),
    neck=dict(
        type='EncodeWithLSTM',
        num_lstm=2,
        in_channels=512,
        hidden_channel=256,
    ),
    head=dict(
        type='CTCHead',
        in_channels=512,
        n_class=63,
    ),
)
train_pipeline = [
    dict(type='DecodeImage', img_mode='BGR', channel_first=False),
    dict(type='RecResizeImg', image_shape=[3, 32, 100], infer_mode=False, character_type='ch'),
    dict(type='CTCLabelEncode', max_text_length=25, character_dict_path='test/test.txt', character_type='ch',
         use_space_char=False),
    dict(type='KeepKeys', keep_keys=['image', 'label', 'length']),
]
test_pipeline = [
    dict(type='DecodeImage', img_mode='BGR', channel_first=False),
    dict(type='RecResizeImg', image_shape=[3, 32, 100], infer_mode=False, character_type='ch'),
    dict(type='CTCLabelEncode', max_text_length=25, character_dict_path='test/test.txt', character_type='ch',
         use_space_char=False),
    dict(type='KeepKeys', keep_keys=['image', 'label', 'length']),
]


train_data = dict(
    dataset=dict(
        type='RecTextDataset',
        ann_file=r'/zzf/data/captcha/data/train_list.txt',
        pipeline=train_pipeline,
    ),
    loader=dict(
        batch_size=32,
        num_workers=4,
        workers_per_gpu=1,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )
)

test_data = dict(
    dataset=dict(
        type='RecTextDataset',
        ann_file=r'/zzf/data/captcha/data/test_list.txt',
        pipeline=test_pipeline,
    ),
    loader=dict(
        batch_size=1,
        num_workers=4,
        workers_per_gpu=1,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
    )
)




# 学习率优化设置 默认StepLR
# lr_scheduler = dict(type='StepLR', step_size=5, gamma=0.1)
lr_scheduler = dict(type='CosineWarmup', warm_up_epochs=3, epochs=50)
# 优化器设置 默认SGD
optimizer = dict(type='SGD', lr=0.001, momentum=0.99, weight_decay=5e-4)
# loss设置
loss = dict(type='CTCLoss')
# 后处理设置
postprocess = dict(
    type='CTCLabelDecode',
    character_dict_path='test/test.txt',
    character_type='ch',
    use_space_char=False
)
metric = dict(type='RecMetric', main_indicator='acc')

options = dict(
    device='1',  # gup ids,
    total_epochs=50,  # 训练epoch大小,
    work_dir=None,  # 模型保存文件目录，包含日志文件
    load_from=None,  # 用于加载已训练完模型，用于用较低学习率微调网络
    resume_from=None,  # 用于程序以外中断，继续训练
    is_eval=True,
    eval_batch_step=[0, 5000],
    # 验证集配置，根据统计指标计算，默认给保存最好的模型
    checkpoint_interval_epoch=10,  # 模型保存策略，默认每个epoch都保存
    save_best_checkpoint=True,
    print_batch_step=50,  # step为单位
    log_smooth_window=20,
    seed=2021,  # 随机种子
)
