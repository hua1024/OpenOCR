# coding=utf-8  
# @Time   : 2020/12/4 10:28
# @Auto   : zzf-jeff

model = dict(
    type='CRNN',
    backbone=dict(
        type='RecResNet',
        in_channels=1,
        depth=50
    ),
    neck=dict(
        type='DecodeWithLSTM',
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

hyp = dict(
    seed=1111,
    gpu_id='0',
    max_iterations=100000,
    init_lr=0.001,

    optimizer=dict(
        type='SGDDecay',
        params=dict(
            weight_decay=0.0001,
            momentum=0.9
        )
    ),
    criterion=dict(
        type='CTCLoss'
    ),
    lr_scheduler=dict(
        type='StepLR',
        params=dict(
            step_size=10,
            gamma=0.1
        )
    )
)

