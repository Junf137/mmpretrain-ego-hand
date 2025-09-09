#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Simple inference script for ego-hand classification.

Takes JSON file with images and HAMER features, outputs predictions and visualizations.
Groups results by frame and uses color coding for different hand types.
"""

import argparse
import json
import os.path as osp
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mmcv.image import imread
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import RUNNERS

from mmpretrain.structures import DataSample


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
    classes = ['non-ego', 'ego']

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

            result = {
                'image_path': sample['image_path'],
                'pred_label': pred_label,
                'pred_score': pred_score,
                'pred_class': classes[pred_label],
                'pred_scores': pred_scores.tolist(),
                # Keep original HAMER data for visualization
                'frame_idx': sample.get('frame_idx'),
                'track_id': sample.get('track_id'),
                'hand_type': sample.get('hand_type', 0),  # 0=left, 1=right
                'is_right': sample.get('is_right', False),
                'confidence': sample.get('confidence', sample.get('detection_confidence', 0)),
                'joints_2d': sample.get('joints_2d', []),
                'bbox': sample.get('bbox', []),
                'id': sample.get('id', '')
            }

            # Add ground truth if available
            if 'label' in sample:
                result['gt_label'] = sample['label']
                result['gt_class'] = classes[sample['label']]
                result['correct'] = pred_label == sample['label']

            results.append(result)

    return results


def group_by_frame(results: List[dict]) -> Dict[int, List[dict]]:
    """Group results by frame index."""
    frame_groups = defaultdict(list)
    for result in results:
        frame_idx = result.get('frame_idx')
        if frame_idx is not None:
            frame_groups[frame_idx].append(result)
    return dict(frame_groups)


def get_hand_color(result: dict) -> tuple:
    """Get color for hand based on prediction and handedness.

    Args:
        result (dict): Prediction result with hand info.

    Returns:
        tuple: BGR color (for OpenCV)
    """
    if result['pred_class'] == 'ego':
        return (0, 255, 0)  # Green for ego hands
    else:
        # Non-ego hands: blue for left, red for right
        if result['is_right']:
            return (0, 0, 255)  # Red for right non-ego
        else:
            return (255, 0, 0)  # Blue for left non-ego


def draw_hand_on_frame(image: np.ndarray, result: dict) -> np.ndarray:
    """Draw single hand detection on frame.

    Args:
        image (np.ndarray): Input image.
        result (dict): Hand detection result.

    Returns:
        np.ndarray: Image with hand drawn.
    """
    img_vis = image.copy()

    # Get color based on prediction and handedness
    color = get_hand_color(result)

    # Draw hand joints
    if result['joints_2d']:
        joints_2d = np.array(result['joints_2d'])  # (21, 2)

        # Hand connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        # Draw connections with hand color
        for start_idx, end_idx in connections:
            if start_idx < len(joints_2d) and end_idx < len(joints_2d):
                start_point = tuple(joints_2d[start_idx].astype(int))
                end_point = tuple(joints_2d[end_idx].astype(int))
                cv2.line(img_vis, start_point, end_point, color, 2)

        # Draw joints (no text labels as requested)
        for joint in joints_2d:
            center = tuple(joint.astype(int))
            cv2.circle(img_vis, center, 4, color, -1)
            cv2.circle(img_vis, center, 5, (255, 255, 255), 1)  # White border

    # Draw bounding box with hand info
    if result['bbox']:
        bbox = result['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

        # Add hand info text above bbox
        hand_side = 'Right' if result['is_right'] else 'Left'
        bbox_text = f"{hand_side} | {result['pred_class']} ({result['pred_score']:.2f})"
        text_y = max(y1 - 10, 20)

        # Background for text
        text_size = cv2.getTextSize(bbox_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img_vis, (x1, text_y - text_size[1] - 5),
                     (x1 + text_size[0] + 10, text_y + 5), (0, 0, 0), -1)
        cv2.rectangle(img_vis, (x1, text_y - text_size[1] - 5),
                     (x1 + text_size[0] + 10, text_y + 5), color, 2)

        cv2.putText(img_vis, bbox_text, (x1 + 5, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img_vis


def create_frame_visualization(image_path: str, hands_in_frame: List[dict], save_path: str):
    """Create visualization for all hands in a single frame.

    Args:
        image_path (str): Path to frame image.
        hands_in_frame (List[dict]): All hand detections in this frame.
        save_path (str): Where to save the visualization.
    """
    try:
        # Load image
        image = imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return

        img_vis = image.copy()

        # Draw all hands in the frame
        for hand_result in hands_in_frame:
            img_vis = draw_hand_on_frame(img_vis, hand_result)

        # Add frame summary info
        frame_idx = hands_in_frame[0]['frame_idx']
        summary_text = f"Frame {frame_idx} | {len(hands_in_frame)} hand(s)"

        # Count predictions
        ego_count = sum(1 for h in hands_in_frame if h['pred_class'] == 'ego')
        non_ego_count = len(hands_in_frame) - ego_count
        summary_text += f" | Ego: {ego_count}, Non-ego: {non_ego_count}"

        # Background for summary
        text_size = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(img_vis, (10, 10), (20 + text_size[0], 40 + text_size[1]), (0, 0, 0), -1)
        cv2.rectangle(img_vis, (10, 10), (20 + text_size[0], 40 + text_size[1]), (255, 255, 255), 2)

        cv2.putText(img_vis, summary_text, (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add legend
        legend_y_start = img_vis.shape[0] - 100
        legend_items = [
            ("Ego hand", (0, 255, 0)),
            ("Left non-ego", (255, 0, 0)),
            ("Right non-ego", (0, 0, 255))
        ]

        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y_start + i * 25
            cv2.circle(img_vis, (20, y_pos), 8, color, -1)
            cv2.circle(img_vis, (20, y_pos), 10, (255, 255, 255), 2)
            cv2.putText(img_vis, label, (40, y_pos + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Save the frame
        cv2.imwrite(save_path, img_vis)

    except Exception as e:
        print(f"Warning: Failed to create visualization for {image_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Ego-Hand Classification Inference')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('json_file', help='Input JSON file with samples')
    parser.add_argument('--output', default='predictions.json', help='Output predictions file')
    parser.add_argument('--vis-dir', default='./visualizations', help='Directory to save visualizations')
    parser.add_argument('--device', default='cuda', help='Device for inference')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-frames', type=int, default=4000, help='Maximum number of frame visualizations to create')

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

    # Calculate metrics if ground truth available
    if any('gt_label' in r for r in results):
        correct = sum(1 for r in results if r.get('correct', False))
        total = len(results)
        accuracy = correct / total * 100
        print(f"Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")

        # Per-hand-type accuracy
        left_results = [r for r in results if not r['is_right']]
        right_results = [r for r in results if r['is_right']]

        if left_results:
            left_correct = sum(1 for r in left_results if r.get('correct', False))
            left_acc = left_correct / len(left_results) * 100
            print(f"Left Hand Accuracy: {left_acc:.1f}% ({left_correct}/{len(left_results)})")

        if right_results:
            right_correct = sum(1 for r in right_results if r.get('correct', False))
            right_acc = right_correct / len(right_results) * 100
            print(f"Right Hand Accuracy: {right_acc:.1f}% ({right_correct}/{len(right_results)})")

    # Save predictions
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Predictions saved to {args.output}")

    # Create frame-based visualizations
    if args.vis_dir:
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(exist_ok=True, parents=True)

        print(f"Creating frame-based visualizations...")

        # Group results by frame
        frame_groups = group_by_frame(results)
        print(f"Found {len(frame_groups)} unique frames")

        # Sort frames by frame index
        sorted_frames = sorted(frame_groups.keys())[:args.max_frames]

        for frame_idx in sorted_frames:
            hands_in_frame = frame_groups[frame_idx]

            # Use first hand's image path (all hands in frame should have same image)
            image_path = hands_in_frame[0]['image_path']

            save_path = vis_dir / f"frame_{frame_idx:06d}.png"

            create_frame_visualization(image_path, hands_in_frame, str(save_path))

        print(f"Created {len(sorted_frames)} frame visualizations in {vis_dir}")

        # Create summary statistics
        create_summary_stats(frame_groups, vis_dir)

    print("Done!")


def create_summary_stats(frame_groups: Dict[int, List[dict]], output_dir: Path):
    """Create summary statistics visualization."""
    try:
        # Calculate statistics
        total_frames = len(frame_groups)
        total_hands = sum(len(hands) for hands in frame_groups.values())

        ego_hands = 0
        non_ego_hands = 0
        left_hands = 0
        right_hands = 0

        for hands in frame_groups.values():
            for hand in hands:
                if hand['pred_class'] == 'ego':
                    ego_hands += 1
                else:
                    non_ego_hands += 1

                if hand['is_right']:
                    right_hands += 1
                else:
                    left_hands += 1

        # Create summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Ego vs Non-ego distribution
        ego_counts = [ego_hands, non_ego_hands]
        ego_labels = ['Ego', 'Non-ego']
        colors = ['lightgreen', 'lightcoral']
        ax1.pie(ego_counts, labels=ego_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Ego vs Non-ego Distribution')

        # 2. Left vs Right distribution
        hand_counts = [left_hands, right_hands]
        hand_labels = ['Left', 'Right']
        colors = ['skyblue', 'lightpink']
        ax2.pie(hand_counts, labels=hand_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Left vs Right Hand Distribution')

        # 3. Hands per frame histogram
        hands_per_frame = [len(hands) for hands in frame_groups.values()]
        ax3.hist(hands_per_frame, bins=range(1, max(hands_per_frame) + 2),
                alpha=0.7, color='lightblue', edgecolor='black')
        ax3.set_title('Hands per Frame Distribution')
        ax3.set_xlabel('Number of hands')
        ax3.set_ylabel('Number of frames')
        ax3.grid(axis='y', alpha=0.3)

        # 4. Confidence distribution
        confidences = [hand['pred_score'] for hands in frame_groups.values() for hand in hands]
        ax4.hist(confidences, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.axvline(np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax4.set_title('Prediction Confidence Distribution')
        ax4.set_xlabel('Confidence')
        ax4.set_ylabel('Count')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        # Add overall statistics as text
        stats_text = f"""Summary Statistics:
Total Frames: {total_frames}
Total Hands: {total_hands}
Avg Hands/Frame: {total_hands/total_frames:.1f}

Ego Hands: {ego_hands} ({ego_hands/total_hands*100:.1f}%)
Non-ego Hands: {non_ego_hands} ({non_ego_hands/total_hands*100:.1f}%)

Left Hands: {left_hands} ({left_hands/total_hands*100:.1f}%)
Right Hands: {right_hands} ({right_hands/total_hands*100:.1f}%)"""

        plt.figtext(0.02, 0.02, stats_text, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Ego-Hand Classification Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.savefig(output_dir / 'summary_stats.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("Summary statistics saved")

    except Exception as e:
        print(f"Warning: Failed to create summary stats: {e}")


if __name__ == '__main__':
    main()