_base_ = ['../_base_/schedules/schedule_1x.py','../_base_/datasets/wider_face.py','../_base_/default_runtime.py']
# model settings
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    pad_size_divisor=32)
pretrained = '/home/gzhu2023/gzhu2023/ckr/mmdetection/checkpoints/internimage_s_1k_224.pth'
model = dict(
    type='YOLOV3',
    data_preprocessor=data_preprocessor,
    # backbone=dict(
    #     type='Darknet',
    #     depth=53,
    #     out_indices=(3, 4, 5),
    #     init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    backbone=dict(
        # _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=80,
        depths=[4, 4, 21, 4],
        groups=[5, 10, 20, 40],
        mlp_ratio=4.,
        drop_path_rate=0.3,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=True,
        with_cp=False,
        out_indices=(1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='YOLOTina',
        num_scales=3,
        band_kernel_size = 11,
        in_channels=[640, 320, 160],
        out_channels=[516, 258, 126]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=1,
        in_channels=[516, 258, 126],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            # base_sizes=[[(76.8, 64), (96.76, 80.63), (121.91, 101.59)],
            #             [(38.40, 32), (48.38, 40.32), (60.96, 50.80)],
            #             [(19.2, 16), (24.19, 20.16), (30.48, 25.40)]],
            base_sizes=[[(64, 76.8), (80.63, 96.76), (101.59, 121.91)],
                        [(32, 38.40), (40.32, 48.38), (50.80, 60.96)],
                        [(16, 19.2), (20.16, 24.19), (25.40, 30.48)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))
# dataset settings
dataset_type = 'WIDERFaceDataset'
data_root = 'data/WIDERFace/'
# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=data_preprocessor['mean'],
        to_rgb=data_preprocessor['bgr_to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='RandomResize', scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(608, 608), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='VOC2007/ImageSets/Main/train.txt',
        # data_prefix=dict(sub_data_root='VOC2007/'),
        filter_cfg=dict(
            filter_empty_gt=True, min_size=32, bbox_min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='VOC2007/ImageSets/Main/val.txt',
        # data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

# val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
# test_evaluator = val_evaluator
val_evaluator = dict(
    # TODO: support WiderFace-Evaluation for easy, medium, hard cases
    type='VOCMetric',
    metric='mAP',
    eval_mode='11points')
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=273, val_interval=7)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=2000),
    dict(type='MultiStepLR', by_epoch=True, milestones=[218, 246], gamma=0.1)
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=7, save_best='auto'))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
