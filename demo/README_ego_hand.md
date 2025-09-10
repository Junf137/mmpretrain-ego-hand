# Ego-Hand Classification Inference

Frame-based inference script for ego-hand classification with color-coded hand visualization.

## Usage

```bash
python demo/ego_hand_inference.py \
    configs/ego_hand/ego_classifier_cfg.py \
    work_dirs/ego_hand_single/best_accuracy_top1_epoch_XX.pth \
    data/test/inference_data_list.json \
    --output predictions.json \
    --vis-dir ./visualizations \
    --batch-size 32 \
    --max-frames 100
```

## Input Format

JSON file with list of hand samples containing HAMER detection data:
```json
[
  {
    "image_path": "data/test/ultrawide_video_ep003/frame_000019.jpg",
    "hamer_feats": [49 float values],
    "frame_idx": 19,
    "track_id": 6,
    "hand_type": 0,
    "is_right": false,
    "confidence": 0.31,
    "joints_2d": [[x1, y1], [x2, y2], ...],
    "bbox": [x1, y1, x2, y2],
    "label": 0
  }
]
```

## Output Format

**Predictions file** (`predictions.json`):
```json
[
  {
    "image_path": "data/test/ultrawide_video_ep003/frame_000019.jpg",
    "pred_label": 1,
    "pred_score": 0.85,
    "pred_class": "ego",
    "pred_scores": [0.15, 0.85],
    "frame_idx": 19,
    "track_id": 6,
    "is_right": false,
    "confidence": 0.31,
    "gt_label": 1,
    "gt_class": "ego",
    "correct": true
  }
]
```

## Visualizations

**Frame-based visualizations** (`frame_XXXXXX.png`):
- Groups all hand detections per frame
- Color-coded hand visualization:
  - **🟢 Green**: Ego hands (predicted as ego)
  - **🔵 Blue**: Left non-ego hands
  - **🔴 Red**: Right non-ego hands
- Shows for each hand:
  - 21 hand keypoints with connections (no text labels)
  - Bounding box with hand info (Left/Right | ego/non-ego | confidence)
  - Frame summary (frame number, hand counts)
  - Color legend

**Summary statistics** (`summary_stats.png`):
- Overall ego/non-ego distribution
- Left/right hand distribution
- Hands per frame histogram
- Prediction confidence distribution
- Detailed statistics text

## Options

- `--output`: Output JSON file for predictions (default: `predictions.json`)
- `--vis-dir`: Directory for visualization images (default: `./visualizations`)
- `--device`: Use `cuda` or `cpu` (default: `cuda`)
- `--batch-size`: Batch size for inference (default: 32)
- `--max-frames`: Maximum number of frame visualizations (default: 100)

## Example

```bash
# Run inference with frame-based visualization
python demo/ego_hand_inference.py \
    configs/ego_hand/ego_classifier_cfg.py \
    work_dirs/ego_hand_single/best_accuracy_top1_epoch_XX.pth \
    data/test/inference_data_list.json \
    --output test_predictions.json \
    --vis-dir ./test_visualizations \
    --max-frames 50
```

This will:
1. Load your trained ego-hand classification model
2. Process all hand samples in the JSON file
3. Group results by frame (multiple hands per frame)
4. Save predictions to `test_predictions.json`
5. Create frame visualizations with color-coded hands:
   - Green for ego hands
   - Blue for left non-ego hands
   - Red for right non-ego hands
6. Generate summary statistics
7. Show per-hand-type accuracy if ground truth is available

## Color Coding System

- **🟢 Green hands**: Classified as ego hands
- **🔵 Blue hands**: Left hands classified as non-ego
- **🔴 Red hands**: Right hands classified as non-ego

Each hand shows:
- Precise HAMER keypoints without text labels
- Bounding box with hand type and prediction info
- Detection confidence from HAMER