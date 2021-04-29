# coding=utf-8  
# @Time   : 2020/12/29 18:08
# @Auto   : zzf-jeff

algorithm = 'CRNN'

model = dict(
    type='RecognitionModel',
    transform=None,
    backbone=dict(
        type='RecMobileNetV3',
        in_channels=3,
        mode='large'
    ),
    neck=dict(
        type='EncodeWithLSTM',
        num_lstm=2,
        in_channels=960,
        hidden_channel=96,
    ),
    head=dict(
        type='CTCHead',
        in_channels=192,
        n_class=None,
    ),
)
train_pipeline = [
    dict(type='DecodeImage', img_mode='RGB', channel_first=False),
    dict(type='RecResizeImg', image_shape=[3, 32, 200], infer_mode=False, character_type='ch'),
    dict(type='CTCLabelEncode', max_text_length=50, character_dict_path='test/gsc_dict.txt',
         character_type='ch',
         use_space_char=False),
    dict(type='KeepKeys', keep_keys=['image', 'label', 'length']),
]
test_pipeline = [
    dict(type='DecodeImage', img_mode='RGB', channel_first=False),
    dict(type='RecResizeImg', image_shape=[3, 32, 200], infer_mode=False, character_type='ch'),
    dict(type='CTCLabelEncode', max_text_length=50, character_dict_path='test/gsc_dict.txt',
         character_type='ch',
         use_space_char=False),
    dict(type='KeepKeys', keep_keys=['image', 'label', 'length']),
]

train_data = dict(
    dataset=dict(
        type='RecTextDataset',
        ann_file=r'/zzf/data/ocr_rec_data/aiyunxiao/gushici_train_val/train.txt',
        pipeline=train_pipeline,
        mode='train',
        data_root='/zzf/data/ocr_rec_data/aiyunxiao/gushici_train_val'
    ),
    loader=dict(
        batch_size=32,
        num_workers=4,
        collate_fn=None,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )
)

test_data = dict(
    dataset=dict(
        type='RecTextDataset',
        ann_file=r'/zzf/data/ocr_rec_data/aiyunxiao/gushici_train_val/val.txt',
        pipeline=test_pipeline,
        mode='test',
        data_root='/zzf/data/ocr_rec_data/aiyunxiao/gushici_train_val',
    ),
    loader=dict(
        batch_size=16,
        num_workers=4,
        collate_fn=None,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
    )
)

# 学习率优化设置 默认StepLR
# lr_scheduler = dict(type='StepLR', step_size=5, gamma=0.1)
# lr_scheduler = dict(type='MultiStepWarmup', milestones=[15, 25, 40], gamma=0.1, warm_up_epochs=5)
lr_scheduler = dict(type='CosineWarmup', warm_up_epochs=3, epochs=50)
# 优化器设置 默认SGD
optimizer = dict(type='Adam', lr=0.001, beta1=0.9, beta2=0.99, weight_decay=1e-5)

# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=1e-4)
# loss设置
loss = dict(type='CTCLoss')
# 后处理设置
postprocess = dict(
    type='CTCLabelDecode',
    character_dict_path='test/gsc_dict.txt',
    character_type='ch',
    use_space_char=False
)

metric = dict(type='RecMetric', main_indicator='acc')

options = dict(
    gpu_ids='0,1',  # gup ids,
    total_epochs=50,  # 训练epoch大小,
    work_dir=None,  # 模型保存文件目录，包含日志文件
    load_from=None,  # 用于加载已训练完模型，用于用较低学习率微调网络
    resume_from=None,  # 用于程序以外中断，继续训练
    is_eval=True,
    eval_batch_step=[0, 10000],
    # 验证集配置，根据统计指标计算，默认给保存最好的模型
    checkpoint_interval_epoch=1,  # 模型保存策略，默认每个epoch都保存
    save_best_checkpoint=True,
    print_batch_step=50,  # step为单位
    log_smooth_window=20,
    seed=128,  # 随机种子
)
