import json
from collections import Counter

with open("data/ego_hand/train_0.json", "r") as f:
    data = json.load(f)
class_counts = Counter([item["label"] for item in data])
total = sum(class_counts.values())
n_classes = len(class_counts)
class_weights = [total / (n_classes * class_counts[i]) for i in range(n_classes)]
print(f"Class distribution: {dict(class_counts)}")
print(f"Recommended class_weight: {class_weights}")
