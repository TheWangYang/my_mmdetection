_base_ = './yolov3_d53_mstrain-608_273e_coco.py'

# model settings
model = dict(bbox_head=dict(num_classes=3,))

# dataset settings
# set the classes according to mydet
classes = ('rust', 'scratch', 'spot')
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(3000, 1800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3000, 1800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# set the dataset path using mydet dataset path
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        img_prefix='data/images/',
        classes=classes,
        ann_file='data/annotations/train2017.json',
        pipeline=train_pipeline),
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


# # optimizer LR
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=64)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[218, 246])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=273)

evaluation = dict(interval=10, metric=['bbox'])

# set checkpoints save every: 5 epochs
checkpoint_config = dict(interval=20)


