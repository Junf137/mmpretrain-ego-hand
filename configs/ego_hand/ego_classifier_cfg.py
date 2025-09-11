
# Config file for training
dataset_type = 'EgoHandDataset'
data_root = 'data/ego_hand/'  # Adjust to your data root
mean_std_file = 'data/ego_hand/hamer_mean_std.json'

work_dir = 'work_dirs/ego_hand'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(224, 224)),
    dict(type='EgoSyncedHorizontalFlip', prob=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='LoadHamerFeats'),
    dict(type='StandardizeHamerFeats', mean_std_file=mean_std_file),
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
        ann_file=data_root + 'train.json',
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DistributedWeightedSampler'),
    persistent_workers=True,
    pin_memory=True
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'valid.json',
        pipeline=val_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
    pin_memory=True
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
    persistent_workers=True,
    pin_memory=True
)

# Model config
model = dict(
    type='EgoClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=(3,),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch',
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=4,  # 4-class: ego left/right, exo left/right
        in_channels=2048 + 1024,
        loss=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
        ),
        topk=(1, 2),
    ),
)

# Optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'hamer_encoder': dict(lr_mult=5.0),
            'head': dict(lr_mult=5.0),
        }
    ),
    clip_grad=dict(max_norm=5.0, norm_type=2)
)

# Learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', T_max=100, by_epoch=True)
]

# Evaluator for validation and testing
val_evaluator = [
    dict(type='Accuracy', topk=(1, 2)),
    dict(type='SingleLabelMetric', items=['precision', 'recall', 'f1-score'], average='macro')
]
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
        save_best='f1-score/macro',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='VisualizationHook',
        enable=True,  # Enable visualization
        interval=500,  # Visualize every 500 samples
        show=False,    # Don't show images during training
    ),
)

custom_hooks = [
    dict(type='EarlyStoppingHook', monitor='f1-score/macro', rule='greater',
         patience=10, min_delta=0.0)
]

# Configure visualizer with wandb backend
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='ego-hand-classification',
                name='ego_classifier',
                tags=['ego-hand', 'multimodal', 'hamer', 'resnet50'],
                notes='4-class classification of ego-hand using ResNet50 + HAMER features'
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
