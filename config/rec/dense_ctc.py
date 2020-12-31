# coding=utf-8  
# @Time   : 2020/12/18 17:09
# @Auto   : zzf-jeff


model = dict(
    type='CRNN',
    backbone=dict(
        type='RecResNet',
        in_channels=1,
        num_classes=100,
        depth=50
    ),
    neck=None,
    head=dict(
        type='CRNNHead',
        use_lstm=True,
        lstm_num=3,
        use_conv=False,
        use_attention=False,
        in_channel=2048,
        hidden_channel=1024,
        classes=10
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

train = dict(

)

test = dict(

)
