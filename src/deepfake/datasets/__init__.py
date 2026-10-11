"""PyTorch datasets over processed data."""

from deepfake.datasets.audio import AudioDataset
from deepfake.datasets.multimodal import MultimodalDataset
from deepfake.datasets.visual import VisualDataset

__all__ = ["AudioDataset", "MultimodalDataset", "VisualDataset"]
