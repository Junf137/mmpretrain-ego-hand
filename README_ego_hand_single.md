# 🤲 Ego-Hand Classification Pipeline

A simple, production-ready pipeline for binary ego-hand classification using ResNet50 + HAMER features.

## 🚀 Quick Start

### 1. Prepare Data

Structure your data directory as:
```
data/ego_hand/
├── train.json     # Training samples
├── valid.json     # Validation samples
└── images/        # Image files
```

JSON format:
```json
[
  {
    "image_path": "path/to/image.jpg",
    "hamer_feats": [49 float values],  // HAMER pose features
    "label": 0  // 0=non-ego, 1=ego
  },
  ...
]
```

### 2. Train the Model

```bash
# Basic training
./train_ego_single.sh

# With custom parameters
./train_ego_single.sh \
    --config configs/ego_hand/ego_classifier_cfg.py \
    --work-dir work_dirs/my_experiment \
    --data-root data/ego_hand/ \
    --amp  # Enable mixed precision
```

### 3. Calculate Optimal Class Weights (Optional)

```bash
python tools/weight_calc.py data/ego_hand/
```

Update the calculated weights in your config:
```python
loss=dict(
    type='CrossEntropyLoss',
    class_weight=[calculated_weight_0, calculated_weight_1]
)
```

## 🏗️ Architecture

**Multimodal Dual-Branch Design:**
- **Image Branch**: ResNet-50 → GlobalAvgPool → 2048D features
- **HAMER Branch**: MLP (49→512→256) → 256D features
- **Fusion**: Concatenate → 2304D → LinearClassifier → 2 classes

## 📊 Monitoring

The pipeline includes:
- **Weights & Biases** integration for experiment tracking
- **Automatic checkpointing** with best model saving
- **Real-time visualization** every 500 samples
- **Comprehensive logging** every 50 iterations

## 📁 Project Structure

```
configs/ego_hand/
├── ego_classifier_cfg.py      # Main config file

mmpretrain/
├── models/classifiers/
│   └── ego_classifier.py      # Custom multimodal classifier
└── datasets/
    └── ego_hand_dataset.py    # Custom dataset loader

tools/
├── train.py                   # Standard MMPretrain training
└── weight_calc.py             # Class weight calculation

demo/
├── ego_hand_inference.py      # Inference script
└── README_ego_hand.md         # Inference guide

train_ego_single.sh            # Simple training launcher
```

## ⚙️ Configuration

### Key Parameters

**Model Configuration:**
```python
model = dict(
    type='EgoClassifier',
    backbone=dict(type='ResNet', depth=50),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=2304,  # 2048 + 256
        loss=dict(type='CrossEntropyLoss', class_weight=[6.0, 0.5])
    )
)
```

**Training Parameters:**
- **Batch Size**: 32
- **Optimizer**: SGD (lr=0.01, momentum=0.9, weight_decay=1e-4)
- **Scheduler**: MultiStepLR (milestones=[30,60,90], gamma=0.1)
- **Max Epochs**: 100
- **Class Weights**: [6.0, 0.5] (emphasizing minority class)

## 🔧 Advanced Usage

### Resume Training
```bash
./train_ego_single.sh --resume
```

### Mixed Precision Training
```bash
./train_ego_single.sh --amp
```

### Custom Configuration
```bash
./train_ego_single.sh \
    --config path/to/your/config.py \
    --work-dir path/to/output \
    --seed 123
```

## 📈 Results

After training, find your results in:
- **Checkpoints**: `work_dirs/ego_hand_single/*.pth`
- **Logs**: `work_dirs/ego_hand_single/vis_data/`
- **W&B Dashboard**: Check your wandb project for real-time metrics
- **Best Model**: `work_dirs/ego_hand_single/best_accuracy_top1_epoch_XX.pth`

## 🎯 Inference

Use the trained model for prediction:
```bash
python demo/ego_hand_inference.py \
    configs/ego_hand/ego_classifier_cfg.py \
    work_dirs/ego_hand_single/best_accuracy_top1_epoch_XX.pth \
    data/test/inference_data.json \
    --output predictions.json \
    --vis-dir ./visualizations
```

## 🐛 Troubleshooting

**Missing Data Files**: Ensure `train.json` and `valid.json` exist in data directory
**Class Imbalance**: Run `python tools/weight_calc.py` to get optimal weights
**Memory Issues**: Reduce `batch_size` in config or use `--amp` flag
**Environment**: Make sure correct conda environment is activated

## 📚 Key Features

✅ **Simple Setup**: Single train/valid split, no complex cross-validation
✅ **Class Imbalance Handling**: Weighted loss with configurable class weights
✅ **Multimodal**: Combines visual + pose features for robust classification
✅ **Production Ready**: Comprehensive logging, checkpointing, and monitoring
✅ **Flexible**: Easy to customize and extend for different use cases

---

**Happy Training!** 🚀
