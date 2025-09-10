import torch
import torch.nn as nn
from mmpretrain.registry import MODELS
from .image import ImageClassifier

@MODELS.register_module()
class EgoClassifier(ImageClassifier):
    """Custom multimodal classifier for ego-hand detection.

    Takes image and hamer_feats, encodes separately, concatenates, classifies.
    """
    def __init__(self, backbone, neck=None, head=None, **kwargs):
        super().__init__(backbone=backbone, neck=neck, head=head, **kwargs)

        # Custom hamer encoder: MLP for 57-dim input to 1024-dim
        self.hamer_encoder = nn.Sequential(
            nn.Linear(57, 256), nn.ReLU(inplace=True), nn.LayerNorm(256), nn.Dropout(0.1),
            nn.Linear(256, 512), nn.ReLU(inplace=True), nn.LayerNorm(512), nn.Dropout(0.1),
            nn.Linear(512, 1024), nn.ReLU(inplace=True),
        )

    def extract_feat(self, inputs, stage='neck'):
        """Extract features from both image and hamer features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H, W)
            stage (str): Which stage to output the feature.
        """
        # Image features (standard backbone processing)
        img_feats = self.backbone(inputs)
        if self.with_neck:
            img_feats = self.neck(img_feats)

        return img_feats

    def forward(self, inputs, data_samples=None, mode='tensor'):
        """Forward function.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H, W)
            data_samples (List[DataSample], optional): The data samples that
                include hamer_feats and other meta information.
            mode (str): Return mode, 'tensor', 'predict' or 'loss'.
        """
        # Extract image features
        img_feats = self.extract_feat(inputs)

        # Handle case where img_feats might be a tuple (from backbone stages)
        if isinstance(img_feats, tuple):
            img_feats = img_feats[-1]  # Use the last stage features

        if data_samples is not None:
            hamer_feats = torch.stack([sample.hamer_feats for sample in data_samples])
            hamer_feats = hamer_feats.to(img_feats.device)
        else:
            raise ValueError("data_samples is required for EgoClassifier")

        # Process hamer features
        hamer_feats = self.hamer_encoder(hamer_feats)

        # Concatenate features
        feats = torch.cat([img_feats, hamer_feats], dim=1)

        # Pass through head if available
        if self.with_head:
            # LinearClsHead expects a tuple of features
            feats_tuple = (feats,)
            if mode == 'loss':
                return self.head.loss(feats_tuple, data_samples)
            elif mode == 'predict':
                return self.head.predict(feats_tuple, data_samples)
            else:  # mode == 'tensor'
                return self.head(feats_tuple)

        return feats