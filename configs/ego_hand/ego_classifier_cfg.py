# This code implements the training pipeline using MMPretrain framework.
# Assumptions:
# - Data is preprocessed into a JSON file 'data/ego_hand/train.json' (list of dicts), each entry with:
#   - 'image_path': str (full path to image)
#   - 'hamer_feats': list of floats (flattened normalized joints_2d (42), bbox (4), box_center/size (3) -> 49 floats)
#   - 'label': int (0 for non-ego, 1 for ego)
# - Similar for valid.json and test.json.
# - Images are accessible at the paths in JSON.
# - For simplicity, we omit detection_confidence and hand_type from inputs.
# - Model: ResNet50 for image, MLP for hamer_feats, concat and linear classifier.
# - Place files in appropriate dirs: datasets/ego_hand_dataset.py, models/ego_classifier.py, configs/ego_hand/ego_classifier_cfg.py

# Notes:
# - For binary classification, you can adjust head to num_classes=1 with BCE loss if preferred.
# - Handle class imbalance: Multiple approaches available (see below).
# - Augmentations sync for hamer_feats (e.g., flip): Implement custom transform if needed.
# - Register custom classes: Ensure files are in mmpretrain/ subdirs and imported properly.

# CLASS IMBALANCE HANDLING OPTIONS:
# 1. Class weights in loss (IMPLEMENTED BELOW): class_weight=[2.0, 1.0] for emphasizing class 0
# 2. Weighted sampler: Replace DefaultSampler with WeightedRandomSampler
# 3. Focal Loss: Replace CrossEntropyLoss with FocalLoss
# 4. Combination: Use both weighted sampling + class weights

# Config file for training

dataset_type = 'EgoHandDataset'
data_root = 'data/ego_hand/'  # Adjust to your data root

work_dir = 'work_dirs/ego_hand'

# Pipeline for image processing (hamer_feats not transformed)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='PackInputs', algorithm_keys=['hamer_feats'])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(224, 224)),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='PackInputs', algorithm_keys=['hamer_feats'])
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train_0.json',
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'valid_0.json',
        pipeline=val_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        pipeline=val_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# Option 3: Focal Loss (uncomment to replace CrossEntropyLoss)
# Good for extreme imbalance, focuses on hard examples
# loss=dict(
#     type='FocalLoss',
#     alpha=0.75,      # Weight for rare class (class 0)
#     gamma=2.0,       # Focusing parameter (higher = more focus on hard examples)
#     loss_weight=1.0
# ),

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
        loss=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
            class_weight=[6.0, 0.5]  # [weight_for_class_0, weight_for_class_1] Higher weight for negative class (class 0)
        ),
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

# Evaluator for validation and testing
val_evaluator = dict(type='Accuracy', topk=(1,))
test_evaluator = val_evaluator

# Train, valid, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Runtime settings with wandb integration and proper visualization
default_scope = 'mmpretrain'

# Configure default hooks with best checkpoint saving
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),  # More frequent logging
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,  # Keep only 3 checkpoints to save disk space
        save_best='auto',  # Automatically save best checkpoint based on validation metric
        rule='greater'     # For accuracy, higher is better
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='VisualizationHook',
        enable=True,  # Enable visualization
        interval=500,  # Visualize every 500 samples
        show=False,    # Don't show images during training
    ),
)

# Configure visualizer with wandb backend
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='ego-hand-classification',
                name='ego_classifier_fold_{fold}',  # Will be formatted for each fold
                group='5fold-cross-validation',
                tags=['ego-hand', 'multimodal', 'hamer', 'resnet50'],
                notes='Binary classification of ego-hand using ResNet50 + HAMER features'
            )
        )
    ]
)

# Configure environment
env_cfg = dict(
    cudnn_benchmark=True,  # Enable for better performance with fixed input sizes
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Set log level and other runtime settings
log_level = 'INFO'
log_processor = dict(window_size=20)  # Smooth logging over 20 iterations
load_from = None
resume = False

# Set random seed for reproducibility
randomness = dict(seed=42, deterministic=False)