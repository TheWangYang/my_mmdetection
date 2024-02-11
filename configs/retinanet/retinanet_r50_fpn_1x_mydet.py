_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# --------------modify config file-----------------
# modify model's num_classes property to adopt train set' class number
model = dict(bbox_head=dict(num_classes=3,))

# --------------modify settings in config file about dataset--------------
# ---------------modify dataset class number-------------
dataset_type = 'CocoDataset'
classes = ('rust', 'scratch', 'spot')

# --------------modify images size and so on include data augumentation-----------------
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,# images' number in per gpu: 1
    workers_per_gpu=1, # data worker loader number in per gpu: 1
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


# optimizer
# set lr = 0.005 can run, set 0.008 can run, 0.01 too large
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# user should not change this value
auto_scale_lr = dict(enable=True, base_batch_size=16)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[24, 33])

# -------------------modify train epochs-----------------
runner = dict(
    type='EpochBasedRunner',  # can be modified in [IterBasedRunner, EpochBasedRunner]
    max_epochs=36)

# -------------------print log every 20 intervals---------------------
# yapf:disable
log_config = dict(
    interval=20,  # default: 50, set 20 for each epoch(interval = interval in (trainset length / batchsize))
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# set checkpoints save every: 5 epochs
checkpoint_config = dict(interval=5)
