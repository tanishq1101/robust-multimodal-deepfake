"""Robust Multimodal Deepfake Analysis."""

import os

# albumentations otherwise checks PyPI for updates on every import (and in every DataLoader worker)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

__version__ = "0.1.0"
