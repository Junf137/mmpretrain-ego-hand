import torch
import torch.nn as nn
from mmpretrain.models import ImageClassifier
from mmpretrain.registry import MODELS
from mmpretrain.models.backbones import ResNet
from mmpretrain.models.heads import LinearClsHead
from mmpretrain.models.utils import FCneck  # Simple MLP neck

@MODELS.register_module()
class EgoClassifier(ImageClassifier):
    """Custom multimodal classifier for ego-hand detection.

    Takes image and hamer_feats, encodes separately, concatenates, classifies.
    """
    def __init__(self, backbone, neck=None, head=None, **kwargs):
        super().__init__(backbone=backbone, neck=neck, head=head, **kwargs)

        # Custom hamer encoder: MLP for 49-dim input to 256-dim
        self.hamer_encoder = nn.Sequential(
            nn.Linear(49, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

    def extract_feat(self, inputs, stage='neck'):
        img, hamer_feats = inputs  # inputs is tuple (img_tensor, hamer_tensor)

        # Image features
        img_feats = self.backbone(img)
        if self.with_neck:
            img_feats = self.neck(img_feats)

        # Hamer features
        hamer_feats = self.hamer_encoder(hamer_feats)  # (B, 256)

        # Concat: Assume img_feats is (B, C), e.g., 2048 for ResNet
        feats = torch.cat([img_feats, hamer_feats], dim=1)  # (B, 2048+256)

        return feats

    def forward(self, inputs, **kwargs):
        feats = self.extract_feat(inputs)
        if self.with_head:
            return self.head(feats, **kwargs)
        return feats