#!/usr/bin/env python3
"""
Calculate optimal class weights for ego-hand classification.
Usage: python tools/weight_calc.py [data_root]
"""
import json
import sys
from collections import Counter
from pathlib import Path

def calculate_class_weights(train_file):
    """Calculate class weights from training data."""
    if not Path(train_file).exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    with open(train_file, "r") as f:
        data = json.load(f)

    class_counts = Counter([item["label"] for item in data])
    total = sum(class_counts.values())
    n_classes = len(class_counts)

    # Calculate inverse frequency weights
    class_weights = [total / (n_classes * class_counts[i]) for i in range(n_classes)]

    print("=== Class Weight Calculation ===")
    print(f"Training file: {train_file}")
    print(f"Total samples: {total}")
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Class imbalance ratio: {max(class_counts.values()) / min(class_counts.values()):.2f}:1")
    print(f"Recommended class_weight: {class_weights}")
    print(f"Formatted for config: class_weight={class_weights}")

    return class_weights

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
