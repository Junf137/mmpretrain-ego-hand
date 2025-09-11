# Copyright (c) OpenMMLab. All rights reserved.
import json
import torch
from typing import List, Optional, Sequence, Union

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class EgoHandDataset(BaseDataset):
    """Custom dataset for ego-hand classification.

    This dataset loads from JSON annotation files containing:
    - image_path: Path to the image file
    - hamer_feats: List of 57 floats from HAMER model
    - label: Integer label (0=ego left, 1=ego right, 2=exo left, 3=exo right)

    Args:
        ann_file (str): Path to the JSON annotation file.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        **kwargs: Other arguments passed to BaseDataset.
    """

    # Define default metainfo for 4-class classification
    METAINFO = {
        "classes": ["ego_left", "ego_right", "exo_left", "exo_right"],
        "paper_info": {
            "author": "Custom Implementation",
            "title": "Ego-Hand Classification with HAMER Features",
            "container": "Custom Dataset",
        },
    }

    def __init__(
        self,
        ann_file: str,
        metainfo: Optional[dict] = None,
        data_root: str = "",
        data_prefix: Union[str, dict] = "",
        **kwargs
    ):
        # Set default metainfo if not provided
        if metainfo is None:
            metainfo = self.METAINFO

        super().__init__(ann_file=ann_file, metainfo=metainfo, data_root=data_root, data_prefix=data_prefix, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load data information from annotation file.

        Returns:
            List[dict]: A list of data information dicts containing:
                - img_path: Path to the image
                - gt_label: Ground truth label
                - hamer_feats: HAMER features as tensor
        """
        with open(self.ann_file, "r") as f:
            data_list = json.load(f)

        processed_list = []
        for item in data_list:
            data_info = {
                "img_path": item["image_path"],
                "gt_label": item["label"],
                "hamer_feats": torch.tensor(item["hamer_feats"], dtype=torch.float32),
            }
            processed_list.append(data_info)
        return processed_list
