_base_ = ['./mae_vit-large-p16_8xb512-amp-coslr-800e_ego1m.py']

# ---- load a local ImageNet-1k MAE checkpoint (same arch/patch!) ----
resume = False  # we are NOT resuming a previous ego1m run
load_from = 'checkpoints/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k_20220825-df72726a.pth'

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=0.75e-4 * 4096 / 256,  # ~0.5x your scratch LR
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))


# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,  # adapted for pretrained checkpoint
        by_epoch=True,
        begin=0,
        end=10,  #
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=390,  # adapted for pretrained checkpoint
        by_epoch=True,
        begin=10,  # adapted for pretrained checkpoint
        end=400,  # adapted for pretrained checkpoint
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=400)  # e.g., 100–400e of domain-adaptive pretrain
