#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""5-Fold Cross Validation Training Script for Ego-Hand Classification.

This script trains the ego-hand classifier using 5-fold cross validation with
pre-split data files. Each fold uses its own train/validation split and saves
the best checkpoint for continuation in subsequent folds.
"""

import argparse
import copy
import os
import os.path as osp
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(description='5-Fold Cross Validation Training')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work-dir',
        help='the root dir to save logs and models for all folds',
        default='./work_dirs/ego_hand_5fold'
    )
    parser.add_argument(
        '--fold',
        type=int,
        help='specific fold to train (0-4). If not specified, trains all folds',
        default=None
    )
    parser.add_argument(
        '--resume-fold',
        type=int,
        help='fold to resume from (0-4)',
        default=0
    )
    parser.add_argument(
        '--conda-env',
        help='conda environment to use for training',
        default='vit_pose'
    )
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training'
    )
    parser.add_argument(
        '--wandb-project',
        help='W&B project name',
        default='ego-hand-classification'
    )
    parser.add_argument(
        '--data-root',
        help='data root directory',
        default='data/ego_hand/'
    )

    return parser.parse_args()


def setup_fold_config(base_cfg: Config, fold: int, args) -> Config:
    """Setup configuration for a specific fold."""
    cfg = copy.deepcopy(base_cfg)

    # Set fold-specific work directory
    cfg.work_dir = osp.join(args.work_dir, f'fold_{fold}')
    mkdir_or_exist(cfg.work_dir)

    # Update dataset paths for this fold
    cfg.train_dataloader.dataset.ann_file = osp.join(args.data_root, f'train_{fold}.json')
    cfg.val_dataloader.dataset.ann_file = osp.join(args.data_root, f'valid_{fold}.json')
    cfg.test_dataloader.dataset.ann_file = osp.join(args.data_root, f'valid_{fold}.json')

    # Set fold-specific random seed
    cfg.randomness.seed = args.seed + fold

    # Configure wandb for this fold
    if hasattr(cfg, 'visualizer') and cfg.visualizer is not None:
        for backend in cfg.visualizer.vis_backends:
            if backend.type == 'WandbVisBackend':
                backend.init_kwargs.name = f'ego_classifier_fold_{fold}'
                backend.init_kwargs.project = args.wandb_project
                backend.init_kwargs.tags.append(f'fold_{fold}')

    # Enable AMP if requested
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    return cfg


def find_best_checkpoint(work_dir: str) -> Optional[str]:
    """Find the best checkpoint in a fold's work directory."""
    best_ckpt_path = osp.join(work_dir, 'best_accuracy_top1_epoch_*.pth')
    import glob
    best_ckpts = glob.glob(best_ckpt_path)
    if best_ckpts:
        # Return the most recent best checkpoint
        best_ckpts.sort(key=os.path.getmtime, reverse=True)
        return best_ckpts[0]
    return None


def save_fold_results(fold: int, work_dir: str, results: Dict) -> None:
    """Save fold results to JSON file."""
    results_file = osp.join(work_dir, f'fold_{fold}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def train_fold(cfg: Config, fold: int, args) -> Dict:
    """Train a single fold and return results."""
    print(f'=== Training Fold {fold} ===')

    # Check if we should resume from a previous fold's best checkpoint
    if fold > 0:
        prev_fold_work_dir = osp.join(args.work_dir, f'fold_{fold-1}')
        best_prev_ckpt = find_best_checkpoint(prev_fold_work_dir)
        if best_prev_ckpt:
            print(f'Loading best checkpoint from fold {fold-1}: {best_prev_ckpt}')
            cfg.load_from = best_prev_ckpt

    # Build and start training
    runner = Runner.from_cfg(cfg)
    runner.train()

    # Find and return the best checkpoint info
    best_ckpt = find_best_checkpoint(cfg.work_dir)

    # Load validation results if available
    results = {
        'fold': fold,
        'work_dir': cfg.work_dir,
        'best_checkpoint': best_ckpt,
        'config_seed': cfg.randomness.seed,
    }

    # Try to extract final validation metrics
    try:
        log_file = osp.join(cfg.work_dir, 'vis_data', 'scalars.json')
        if osp.exists(log_file):
            with open(log_file, 'r') as f:
                scalars = json.load(f)
                # Extract best validation accuracy
                if 'val/accuracy_top1' in scalars:
                    val_accs = scalars['val/accuracy_top1']
                    if val_accs:
                        results['best_val_accuracy'] = max([item[1] for item in val_accs])
                        results['final_val_accuracy'] = val_accs[-1][1]
    except Exception as e:
        print(f'Warning: Could not extract validation metrics: {e}')

    save_fold_results(fold, args.work_dir, results)
    print(f'=== Completed Fold {fold} ===')

    return results


def create_summary_report(all_results: List[Dict], work_dir: str) -> None:
    """Create a summary report of all folds."""
    summary = {
        'experiment': 'ego-hand-5fold-cv',
        'total_folds': len(all_results),
        'fold_results': all_results,
        'summary_statistics': {}
    }

    # Calculate summary statistics
    val_accuracies = []
    for result in all_results:
        if 'best_val_accuracy' in result:
            val_accuracies.append(result['best_val_accuracy'])

    if val_accuracies:
        import numpy as np
        summary['summary_statistics'] = {
            'mean_val_accuracy': float(np.mean(val_accuracies)),
            'std_val_accuracy': float(np.std(val_accuracies)),
            'min_val_accuracy': float(np.min(val_accuracies)),
            'max_val_accuracy': float(np.max(val_accuracies)),
            'individual_accuracies': val_accuracies
        }

    # Save summary
    summary_file = osp.join(work_dir, 'cross_validation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print('=== Cross Validation Summary ===')
    if val_accuracies:
        print(f'Mean Validation Accuracy: {summary["summary_statistics"]["mean_val_accuracy"]:.4f} ± {summary["summary_statistics"]["std_val_accuracy"]:.4f}')
        print(f'Individual Fold Accuracies: {val_accuracies}')
    print(f'Summary saved to: {summary_file}')


def main():
    args = parse_args()

    # Load base configuration
    cfg = Config.fromfile(args.config)
    print(f'Loaded config from: {args.config}')

    # Create main work directory
    mkdir_or_exist(args.work_dir)

    # Determine which folds to train
    if args.fold is not None:
        folds_to_train = [args.fold]
        print(f'Training single fold: {args.fold}')
    else:
        folds_to_train = list(range(args.resume_fold, 5))
        print(f'Training folds: {folds_to_train}')

    all_results = []

    # Train each fold
    for fold in folds_to_train:
        try:
            # Setup fold-specific configuration
            fold_cfg = setup_fold_config(cfg, fold, args)

            # Train this fold
            fold_results = train_fold(fold_cfg, fold, args)
            all_results.append(fold_results)

        except Exception as e:
            print(f'Error training fold {fold}: {e}')
            # Continue with next fold
            continue

    # Create final summary
    if all_results:
        create_summary_report(all_results, args.work_dir)

    print('=== 5-Fold Cross Validation Complete ===')


if __name__ == '__main__':
    main()
