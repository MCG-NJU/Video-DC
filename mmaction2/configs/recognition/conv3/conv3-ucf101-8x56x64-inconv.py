_base_ = [
    '../../_base_/default_runtime.py'
]

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='MyConv3d',
        in_size=(56, 64),
        conv_cfg=dict(type='Conv3d'),
        norm_type='InstanceNorm3d',
        norm_eval=False,
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=128,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

dataset_type = 'RawframeDataset'
data_root = './data/ucf101/rawframes'
data_root_val = './data/ucf101/rawframes'
ann_file_train = './data/ucf101/ucf101_train_split_1_rawframes.txt'
ann_file_val = './data/ucf101/ucf101_val_split_1_rawframes.txt'
ann_file_test = './data/ucf101/ucf101_val_split_1_rawframes.txt'
clip_len = 8
frame_interval=8
file_client_args = dict(io_backend='disk')
train_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop'),
    # dict(
    #     type='MultiScaleCrop',
    #     input_size=112,
    #     scales=(1, 0.8),
    #     random_crop=False,
    #     max_wh_scale_gap=0),
    dict(type='Resize', scale=(56, 64), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=clip_len,
        frame_interval=frame_interval,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=56),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=clip_len,
        frame_interval=frame_interval,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='ThreeCrop', crop_size=64),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=150, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=10),
    dict(
        type='MultiStepLR',
        begin=10,
        end=150,
        by_epoch=True,
        milestones=[90, 130],
        gamma=0.1)
]

# train_cfg = dict(
#     type='EpochBasedTrainLoop', max_epochs=256, val_begin=1, val_interval=5)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

# # learning policy
# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=34),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=222,
#         eta_min=0,
#         by_epoch=True,
#         begin=34,
#         end=256)
# ]

# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=1e-4),
#     clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=5))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
