# This code implements the training pipeline using MMPretrain framework.
# Assumptions:
# - Data is preprocessed into a JSON file 'data/ego_hand/train.json' (list of dicts), each entry with:
#   - 'image_path': str (full path to image)
#   - 'hamer_feats': list of floats (flattened normalized joints_2d (42), bbox (4), box_center/size (3) -> 49 floats)
#   - 'label': int (0 for non-ego, 1 for ego)
# - Similar for val.json and test.json.
# - Images are accessible at the paths in JSON.
# - For simplicity, we omit detection_confidence and hand_type from inputs.
# - Model: ResNet50 for image, MLP for hamer_feats, concat and linear classifier.
# - Place files in appropriate dirs: datasets/ego_hand_dataset.py, models/ego_classifier.py, configs/ego_hand/ego_classifier_cfg.py

# Notes:
# - For binary classification, you can adjust head to num_classes=1 with BCE loss if preferred.
# - Handle class imbalance: Add class_weight to loss or use sampler.
# - Augmentations sync for hamer_feats (e.g., flip): Implement custom transform if needed.
# - Register custom classes: Ensure files are in mmpretrain/ subdirs and imported properly.

# Config file for training

dataset_type = 'EgoHandDataset'
data_root = 'data/ego_hand/'  # Adjust to your data root

# Pipeline for image processing (hamer_feats not transformed)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label', 'hamer_feats'])  # Collect includes hamer_feats
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(224, 224)),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label', 'hamer_feats'])
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',  # Your JSON file
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        pipeline=val_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = val_dataloader  # Same as val for simplicity

# Model config
model = dict(
    type='EgoClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),  # Global avg pool after stage 4
        style='pytorch',
    ),
    neck=dict(type='GlobalAveragePooling'),  # To get (B, 2048)
    head=dict(
        type='LinearClsHead',
        num_classes=2,  # Binary, but use 2 classes for softmax/cross-entropy
        in_channels=2048 + 256,  # Image feats + hamer feats
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)

# Optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

# Learning policy
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[30, 60, 90],
    gamma=0.1,
)

# Train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Runtime
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)
log_level = 'INFO'
load_from = None
resume = False