_base_ = [
    '../../_base_/models/slowonly_r18.py', '../../_base_/default_runtime.py', '../../_base_/schedules/sgd_150e_warmup.py'
]

# model settings
model = dict(
    cls_head=dict(
        num_classes=101
    )
)

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/data0/chenyang/ViD/mmaction2/data/ucf101/rawframes'
data_root_val = '/data0/chenyang/ViD/mmaction2/data/ucf101/rawframes'
ann_file_train = '/data0/chenyang/ViD/mmaction2/data/ucf101/ucf101_train_split_1_rawframes.txt'
ann_file_val = '/data0/chenyang/ViD/mmaction2/data/ucf101/ucf101_val_split_1_rawframes.txt'
clip_len = 8
frame_interval = 8
# file_client_args = dict(io_backend='disk')

train_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),
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
    dict(type='Resize', scale=(-1, 128)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=clip_len,
        frame_interval=frame_interval,
        num_clips=10, # 10
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 128)),
    dict(type='ThreeCrop', crop_size=128),
    # dict(type='CenterCrop', crop_size=112),
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
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator


# runtime settings
default_hooks = dict(checkpoint=dict(interval=4, max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (30 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=240)
