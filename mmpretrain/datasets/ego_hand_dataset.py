# File: mmpretrain/datasets/ego_hand_dataset.py
import json
import torch
from mmpretrain.datasets import BaseDataset
from mmpretrain.registry import DATASETS
from mmcv.transforms import LoadImageFromFile, PackInputs, RandomResizedCrop, RandomFlip, Normalize

@DATASETS.register_module()
class EgoHandDataset(BaseDataset):
    """Custom dataset for ego-hand classification.

    Loads from JSON with image_path, hamer_feats (list of floats), label (0/1).
    """
    def __init__(self, ann_file, pipeline, data_prefix='', **kwargs):
        super().__init__(ann_file=ann_file, pipeline=pipeline, data_prefix=data_prefix, **kwargs)

    def load_data_list(self):
        with open(self.ann_file, 'r') as f:
            data_list = json.load(f)

        processed_list = []
        for item in data_list:
            data_info = {
                'img_path': item['image_path'],
                'gt_label': item['label'],
                'hamer_feats': torch.tensor(item['hamer_feats'], dtype=torch.float32)  # Shape: (49,)
            }
            processed_list.append(data_info)
        return processed_list

    def get_data_info(self, idx):
        data_info = super().get_data_info(idx)
        return data_info  # Already includes img_path, gt_label, hamer_feats

# Note: In pipeline, we'll process image, but hamer_feats pass through as is.