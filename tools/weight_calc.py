#!/usr/bin/env python3
"""
Calculate optimal class weights for 4-class ego-hand classification.
Classes: 0=ego_left, 1=ego_right, 2=exo_left, 3=exo_right

Usage: python tools/weight_calc.py [data_root]
"""
import json
import sys
import numpy as np
from collections import Counter
from pathlib import Path

def calculate_class_weights(train_file):
    """Calculate class weights from training data using multiple strategies."""
    if not Path(train_file).exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    with open(train_file, "r") as f:
        data = json.load(f)

    class_counts = Counter([item["label"] for item in data])
    total = sum(class_counts.values())
    n_classes = len(class_counts)

    # Ensure we have all 4 classes (fill missing with 0)
    for i in range(4):
        if i not in class_counts:
            class_counts[i] = 0

    class_names = ['ego_left', 'ego_right', 'exo_left', 'exo_right']

    # Strategy 1: Inverse frequency weights
    inv_freq_weights = []
    for i in range(n_classes):
        if class_counts[i] > 0:
            inv_freq_weights.append(total / (n_classes * class_counts[i]))
        else:
            inv_freq_weights.append(1.0)  # Default for missing classes

    # Strategy 2: Balanced weights (sklearn-style)
    balanced_weights = []
    for i in range(n_classes):
        if class_counts[i] > 0:
            balanced_weights.append(total / (2 * class_counts[i]))
        else:
            balanced_weights.append(1.0)

    # Strategy 3: Square root of inverse frequency (softer)
    sqrt_inv_weights = [np.sqrt(w) for w in inv_freq_weights]

    print("=== 4-Class Ego-Hand Weight Calculation ===")
    print(f"Training file: {train_file}")
    print(f"Total samples: {total}")
    print()

    print("Class Distribution:")
    for i, name in enumerate(class_names):
        count = class_counts[i]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {i}: {name:<10} - {count:6d} samples ({percentage:5.1f}%)")

    print()
    imbalance_ratio = max(class_counts.values()) / max(1, min(class_counts.values()))
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")

    print("\n=== Weighting Strategies ===")
    print("1. Inverse Frequency (strong correction):")
    print(f"   class_weight={[round(w, 3) for w in inv_freq_weights]}")

    print("2. Balanced (sklearn-style, medium correction):")
    print(f"   class_weight={[round(w, 3) for w in balanced_weights]}")

    print("3. Square Root Inverse (soft correction):")
    print(f"   class_weight={[round(w, 3) for w in sqrt_inv_weights]}")

    print("4. Equal weights (no correction):")
    print(f"   class_weight=[1.0, 1.0, 1.0, 1.0]")

    print("\n=== Recommendations ===")
    if imbalance_ratio < 2:
        print("✓ Dataset is relatively balanced. Consider equal weights or soft correction.")
        recommended = [1.0, 1.0, 1.0, 1.0]
    elif imbalance_ratio < 5:
        print("⚠ Moderate imbalance detected. Recommend square root inverse weights.")
        recommended = [round(w, 3) for w in sqrt_inv_weights]
    else:
        print("🚨 Severe imbalance detected. Consider balanced or inverse frequency weights.")
        recommended = [round(w, 3) for w in balanced_weights]

    print(f"Recommended: class_weight={recommended}")

    print("\n=== Additional Strategies ===")
    print("Consider also:")
    print("- Focal Loss: Good for extreme imbalance, focuses on hard examples")
    print("- Data augmentation: Oversample minority classes")
    print("- Weighted sampling: Use WeightedRandomSampler in dataloader")

    return recommended

if __name__ == "__main__":
    # Default data root
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/ego_hand/"
    train_file = Path(data_root) / "train.json"

    try:
        calculate_class_weights(train_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure {train_file} exists or provide correct data root path.")
        sys.exit(1)
