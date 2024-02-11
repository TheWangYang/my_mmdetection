_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]


# 修改模型结构
model = dict(
    roi_head=dict(
        type='PISARoIHead',
        bbox_head=dict(
            num_classes=3,# 修改模型的类别数量为3（满足自定义数据集要求）
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            sampler=dict(
                type='ScoreHLRSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
                k=0.5,
                bias=0.),
            isr=dict(k=2, bias=0),
            carl=dict(k=1, bias=0.2))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0)))


# 修改输入图片的尺寸
dataset_type = 'CocoDataset'
classes = ('rust', 'scratch', 'spot')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1500, 1000), keep_ratio=True),
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
        img_scale=(1500, 1000),
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
    samples_per_gpu=2,# 每块GPU上的图片数量为2，对于1000x1500来说
    workers_per_gpu=2, # 设置每块GPU上的图像加载器为2，对于1000x1500来说
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