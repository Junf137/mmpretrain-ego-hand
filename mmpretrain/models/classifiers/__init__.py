# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .ego_classifier import EgoClassifier
from .hugging_face import HuggingFaceClassifier
from .image import ImageClassifier
from .timm import TimmClassifier

__all__ = [
    'BaseClassifier', 'ImageClassifier', 'TimmClassifier',
    'HuggingFaceClassifier', 'EgoClassifier'
]
