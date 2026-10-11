"""Video-only view of the processed data (visual branch training)."""

from __future__ import annotations

from deepfake.datasets.multimodal import MultimodalDataset


class VisualDataset(MultimodalDataset):
    def __init__(self, index_csv, split, cfg, train=False, frame_transform=None):
        super().__init__(index_csv, split, cfg, train, use_video=True, use_audio=False,
                         frame_transform=frame_transform)
