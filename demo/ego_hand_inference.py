#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Simple inference script for ego-hand classification.

Takes JSON file with images and HAMER features, outputs predictions to JSON.
"""

import argparse
import json
import os.path as osp
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from mmcv.image import imread
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import RUNNERS

from mmpretrain.structures import DataSample

# Import utilities from ego_hands tools
import sys
from pathlib import Path
ego_hands_path = Path(__file__).parent.parent / 'tools' / 'ego_hands'
if str(ego_hands_path) not in sys.path:
    sys.path.insert(0, str(ego_hands_path))

try:
    from hamer_utilities import (
        get_four_class_label_name,
        get_ego_exo_from_four_class_label,
        get_left_right_from_four_class_label
    )
except ImportError as e:
    print(f"Warning: Could not import ego_hands utilities: {e}")
    # Provide fallback implementations
    def get_four_class_label_name(label: int) -> str:
        label_names = {0: 'ego_left', 1: 'ego_right', 2: 'exo_left', 3: 'exo_right'}
        return label_names.get(label, 'unknown')

    def get_ego_exo_from_four_class_label(label: int) -> bool:
        return label in [0, 1]

    def get_left_right_from_four_class_label(label: int) -> str:
        return 'left' if label in [0, 2] else 'right'


def load_model(config_path: str, checkpoint_path: str, device: str = 'cuda'):
    """Load the trained ego-hand classification model."""
    # Fix for PyTorch 2.6 weights_only=True default behavior
    original_torch_load = torch.load
    def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)

    torch.load = patched_torch_load

    try:
        # Load config and build model
        cfg = Config.fromfile(config_path)

        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)

        # Load checkpoint
        runner.load_checkpoint(checkpoint_path)
        model = runner.model.to(device)
        model.eval()

        # Build pipeline
        from mmengine.dataset import Compose
        from mmpretrain.registry import TRANSFORMS

        test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
        pipeline_cfg = [t for t in test_pipeline_cfg if t['type'] != 'LoadImageFromFile']
        pipeline = Compose([TRANSFORMS.build(t) for t in pipeline_cfg])

        return model, cfg, pipeline

    finally:
        # Restore original torch.load function
        torch.load = original_torch_load


def preprocess_data(image_path: str, hamer_feats: List[float], pipeline) -> dict:
    """Preprocess single sample."""
    # Load image
    img = imread(image_path)
    if img is None:
        raise ValueError(f'Failed to read image {image_path}.')

    # Prepare data
    data = dict(
        img=img,
        img_path=image_path,
        img_shape=img.shape[:2],
        ori_shape=img.shape[:2],
        hamer_feats=torch.tensor(hamer_feats, dtype=torch.float32)
    )

    # Apply pipeline
    data = pipeline(data)
    return data


def run_inference(model, data_list: List[dict], pipeline, device: str, batch_size: int = 32):
    """Run inference on data list."""
    results = []
    # 4-class system: 0=ego left, 1=ego right, 2=exo left, 3=exo right
    classes = ['ego_left', 'ego_right', 'exo_left', 'exo_right']

    print(f"Running inference on {len(data_list)} samples...")

    # Process in batches
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(data_list)-1)//batch_size + 1}")

        # Preprocess batch
        batch_data = []
        valid_samples = []
        for sample in batch:
            try:
                data = preprocess_data(sample['image_path'], sample['hamer_feats'], pipeline)
                batch_data.append(data)
                valid_samples.append(sample)
            except Exception as e:
                print(f"Warning: Skipping {sample['image_path']}: {e}")
                continue

        if not batch_data:
            continue

        # Stack inputs
        inputs = torch.stack([data['inputs'] for data in batch_data]).to(device)
        data_samples = []

        for data in batch_data:
            data_sample = data['data_samples']
            data_sample.hamer_feats = data_sample.hamer_feats.to(device)
            data_samples.append(data_sample)

        # Run inference
        with torch.no_grad():
            predictions = model(inputs, data_samples, mode='predict')

        # Convert to results
        for j, (sample, pred) in enumerate(zip(valid_samples, predictions)):
            pred_scores = pred.pred_score.cpu().numpy()
            pred_label = torch.argmax(pred.pred_score).item()
            pred_score = float(torch.max(pred.pred_score).item())

            # Extract ego/exo and left/right from 4-class label
            pred_class_name = get_four_class_label_name(pred_label)
            is_ego_hand = get_ego_exo_from_four_class_label(pred_label)
            predicted_hand_type = get_left_right_from_four_class_label(pred_label)

            result = {
                'image_path': sample['image_path'],
                'pred_label': pred_label,
                'pred_score': pred_score,
                'pred_class': pred_class_name,
                'pred_scores': pred_scores.tolist(),
                # 4-class specific fields
                'is_ego_hand': is_ego_hand,
                'predicted_hand_type': predicted_hand_type,
                # Keep original HAMER data for visualization
                'frame_idx': sample.get('frame_idx'),
                'track_id': sample.get('track_id'),
                'hand_type': sample.get('hand_type', predicted_hand_type),
                'is_right': predicted_hand_type == 'right',
                'confidence': sample.get('confidence', sample.get('detection_confidence', 0)),
                'joints_2d': sample.get('joints_2d', []),
                'bbox': sample.get('bbox', []),
                'id': sample.get('id', '')
            }


            results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description='Ego-Hand Classification Inference')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('json_file', help='Input JSON file with samples')
    parser.add_argument('--output', default='predictions.json', help='Output predictions file')
    parser.add_argument('--device', default='cuda', help='Device for inference')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    # Check input file
    if not osp.exists(args.json_file):
        print(f"Error: Input file not found: {args.json_file}")
        return

    # Load model
    print("Loading model...")
    model, cfg, pipeline = load_model(args.config, args.checkpoint, args.device)
    print(f"Model loaded on {args.device}")

    # Load input data
    with open(args.json_file, 'r') as f:
        data_list = json.load(f)
    print(f"Loaded {len(data_list)} hand samples from {args.json_file}")

    # Run inference
    results = run_inference(model, data_list, pipeline, args.device, args.batch_size)
    print(f"Inference completed on {len(results)} samples")

    # Save predictions
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Predictions saved to {args.output}")

    print("Done!")


if __name__ == '__main__':
    main()