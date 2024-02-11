_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# modify model's num_classes
model = dict(bbox_head=dict(num_classes=3,))

# dataset settings
dataset_type = 'CocoDataset'
classes = ('rust', 'scratch', 'spot')
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(4000, 2400), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(4000, 2400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            img_prefix='data/images/',
            classes=classes,
            ann_file='data/annotations/train2017.json',
            pipeline=train_pipeline)),
    val=dict(
        img_prefix='data/images/',
        classes=classes,
        ann_file='data/annotations/val2017.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='data/images/',
        classes=classes,
        ann_file='data/annotations/test2017.json',
        pipeline=test_pipeline))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

# # optimizer LR
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(_delete_=True)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=64)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[24, 33])

# -------------------修改训练的epochs数-----------------
runner = dict(
    type='EpochBasedRunner',  # 将使用的 runner 的类别 (例如 IterBasedRunner 或 EpochBasedRunner)。
    max_epochs=36)

# 日志打印配置
log_config = dict(
    interval=20,  # default: 50, set 20
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# set checkpoints save every: 5 epochs
checkpoint_config = dict(interval=5)

