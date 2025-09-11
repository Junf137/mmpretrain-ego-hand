import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpretrain.registry import MODELS
from .image import ImageClassifier

@MODELS.register_module()
class EgoClassifier(ImageClassifier):
    """Custom multimodal classifier for ego-hand detection.

    Takes image and hamer_feats, encodes separately, concatenates, classifies.
    """
    def __init__(self, backbone, neck=None, head=None, use_tta_flip=False, **kwargs):
        super().__init__(backbone=backbone, neck=neck, head=head, **kwargs)
        self.use_tta_flip = use_tta_flip

        self.hamer_encoder = nn.Sequential(
            nn.Linear(57, 256), nn.ReLU(inplace=True), nn.LayerNorm(256), nn.Dropout(0.1),
            nn.Linear(256, 512), nn.ReLU(inplace=True), nn.LayerNorm(512), nn.Dropout(0.1),
            nn.Linear(512, 1024), nn.ReLU(inplace=True),
        )

    def extract_feat(self, inputs):
        """Extract features from both image and hamer features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H, W)
        """
        img_feats = self.backbone(inputs)
        if self.with_neck:
            img_feats = self.neck(img_feats)

        return img_feats

    @staticmethod
    def _flip_hamer_feats(feats_b57: torch.Tensor) -> torch.Tensor:
        # feats_b57: [B, 57]; applies same ops as your training-time flip
        feats = feats_b57.clone()
        joints = feats[:, :42].view(-1, 21, 2)
        joints[..., 0] = 1.0 - joints[..., 0]
        feats[:, :42] = joints.reshape(-1, 42)
        x1,y1,x2,y2 = feats[:,42],feats[:,43],feats[:,44],feats[:,45]
        feats[:,42] = 1.0 - x2
        feats[:,44] = 1.0 - x1
        feats[:,46] = 1.0 - feats[:,46]    # center_x
        feats[:,50] = -feats[:,50]         # dir x
        feats[:,52] = -feats[:,52]
        feats[:,54] = 1.0 - feats[:,54]    # hand_type
        return feats

    @staticmethod
    def _permute_flip_logits_4(logits4: torch.Tensor) -> torch.Tensor:
        # classes: [egoL, egoR, exoL, exoR] -> flip swaps left<->right
        idx = torch.tensor([1, 0, 3, 2], device=logits4.device)
        return logits4.index_select(1, idx)

    def _encode(self, inputs, hamer_feats):
        img_feats = self.extract_feat(inputs)
        if isinstance(img_feats, tuple):
            img_feats = img_feats[-1]
        hamer_enc = self.hamer_encoder(hamer_feats)
        feats = torch.cat([img_feats, hamer_enc], dim=1)
        return feats

    def forward(self, inputs, data_samples=None, mode='tensor'):
        """Forward function.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H, W)
            data_samples (List[DataSample], optional): The data samples that
                include hamer_feats and other meta information.
            mode (str): Return mode, 'tensor', 'predict' or 'loss'.
        """
        if data_samples is None:
            raise ValueError("data_samples is required for EgoClassifier")

        # Stack HAMER feats from samples
        hamer = torch.stack([s.hamer_feats for s in data_samples]).to(inputs.device)

        if mode == 'loss':
            feats = self._encode(inputs, hamer)
            return self.head.loss((feats,), data_samples)

        if mode == 'predict':
            if not self.use_tta_flip:
                feats = self._encode(inputs, hamer)
                return self.head.predict((feats,), data_samples)

            # --- TTA: original
            feats_orig = self._encode(inputs, hamer)
            logits_orig = self.head((feats_orig,))  # returns logits tensor

            # --- TTA: flipped
            inputs_f = torch.flip(inputs, dims=[-1])       # width flip
            hamer_f = self._flip_hamer_feats(hamer)
            feats_f = self._encode(inputs_f, hamer_f)
            logits_f = self.head((feats_f,))
            logits_f = self._permute_flip_logits_4(logits_f)

            # average logits, then reuse head's post-processing via predict-like path
            logits = 0.5 * (logits_orig + logits_f)
            scores = F.softmax(logits, dim=-1)

            # Build DataSamples the same way head.predict would
            # Minimal adapter:
            for i, sample in enumerate(data_samples):
                # attach score and label prediction
                sample.set_pred_score(scores[i])
                sample.set_pred_label(int(scores[i].argmax(dim=-1)))
            return data_samples  # list[DataSample]

        # mode == 'tensor'
        feats = self._encode(inputs, hamer)
        return self.head((feats,))
