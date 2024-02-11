_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# --------------修改配置文件-----------------
# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=3),
    ))

# --------------修改数据集相关设置--------------
# ---------------修改数据集类别-------------
dataset_type = 'CocoDataset'
classes = ('rust', 'scratch', 'spot')

# --------------修改训练和测试的图像尺寸大小-----------------
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(3750, 2500), keep_ratio=True),
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
        img_scale=(3750, 2500),
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
    samples_per_gpu=1,# 每块GPU上的图片数量为1
    workers_per_gpu=1, # 设置每块GPU上的图像加载器为1
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
# -------------------修改训练的epochs数-----------------
runner = dict(
    type='EpochBasedRunner',  # 将使用的 runner 的类别 (例如 IterBasedRunner 或 EpochBasedRunner)。
    max_epochs=35)

    # -------------------修改运行时每个epoch中多少batch打印一下日志---------------------
# yapf:disable
log_config = dict(
    interval=20,  # default: 50, set 20
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])