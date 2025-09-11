import random
import torch
import numpy as np
import json
import mmcv
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS

@TRANSFORMS.register_module()
class EgoSyncedHorizontalFlip(BaseTransform):
    """Flip image and synchronize HAMER features and labels.
    Assumes HAMER layout:
      0..41  : joints_2d (21x2)
      42..45 : bbox [x1,y1,x2,y2]
      46..47 : box_center [cx, cy]
      48     : box_size (normed by diag)
      49     : inv_box_size
      50..51 : wrist_to_middle (vx, vy)
      52..53 : wrist_to_thumb (vx, vy)
      54     : hand_type (0=left, 1=right)
      55     : box_horizontal_span_ratio
      56     : box_vertical_span_ratio
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def transform(self, results):
        if np.random.rand() >= self.prob:
            return results

        img = results['img']  # HWC, BGR or RGB depending on pipeline
        # Flip image horizontally
        results['img'] = mmcv.imflip(img, direction='horizontal')

        feats = results['hamer_feats']

        if len(feats) != 57:
            raise ValueError(f"Expected 57-dimensional HAMER features, got {len(feats)}")

        feats = feats.clone()

        # Heuristic: if both bbox==0 and all joints==0 → consider invalid
        bbox_zero = (feats[42:46].abs().sum() == 0)
        joints_zero = (feats[:42].abs().sum() == 0)
        feats_valid = not (bbox_zero and joints_zero)

        # Only flip HAMER features if valid
        if feats_valid:
            # Flip 2D joint coordinates (x-coordinates only)
            joints = feats[:42].view(21, 2)  # 21 joints × 2 coordinates
            joints[:, 0] = 1.0 - joints[:, 0]  # flip x-coordinates
            feats[:42] = joints.flatten()

            # Flip bounding box coordinates
            x1, y1, x2, y2 = feats[42:46]
            feats[42] = 1.0 - x2  # new x1 = 1 - old x2
            feats[44] = 1.0 - x1  # new x2 = 1 - old x1
            # y coordinates unchanged (feats[43], feats[45])

            # Flip box center x-coordinate
            feats[46] = 1.0 - feats[46]  # cx flipped
            # cy unchanged (feats[47])

            # Flip direction vectors (x-components only)
            feats[50] = -feats[50]  # wrist_to_middle vx
            feats[52] = -feats[52]  # wrist_to_thumb vx
            # vy components unchanged (feats[51], feats[53])

            # Flip hand_type: left ↔ right
            feats[54] = 1.0 - feats[54]  # 0 ↔ 1

        # Always flip the LABEL (image truly flipped)
        # Flip 4-class labels: ego_left ↔ ego_right, exo_left ↔ exo_right
        lbl = int(results['gt_label'])
        if lbl == 0:      # ego_left → ego_right
            results['gt_label'] = 1
        elif lbl == 1:    # ego_right → ego_left
            results['gt_label'] = 0
        elif lbl == 2:    # exo_left → exo_right
            results['gt_label'] = 3
        elif lbl == 3:    # exo_right → exo_left
            results['gt_label'] = 2
        else:
            raise ValueError(f"Invalid label {lbl}. Expected 0-3 for 4-class classification.")

        results['hamer_feats'] = feats
        return results


@TRANSFORMS.register_module()
class StandardizeHamerFeats(BaseTransform):
    def __init__(self, mean_std_file, eps=1e-6):
        with open(mean_std_file, 'r') as f:
            mean_std = json.load(f)

        self.mean = torch.tensor(mean_std['mean'], dtype=torch.float32)
        self.std = torch.tensor(mean_std['std'], dtype=torch.float32)
        self.eps = eps
    def transform(self, results):
        feats = results['hamer_feats']
        results['hamer_feats'] = (feats - self.mean) / (self.std + self.eps)
        return results


@TRANSFORMS.register_module()
class LoadHamerFeats(BaseTransform):
    def transform(self, results):
        feats = results['hamer_feats']
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats, dtype=torch.float32)
        results['hamer_feats'] = feats
        return results
