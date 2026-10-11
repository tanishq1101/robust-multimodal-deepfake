"""Audio-only view of the processed data (audio branch training, incl. ASVspoof)."""

from __future__ import annotations

from deepfake.datasets.multimodal import MultimodalDataset


class AudioDataset(MultimodalDataset):
    def __init__(self, index_csv, split, cfg, train=False, mel_transform=None):
        super().__init__(index_csv, split, cfg, train, use_video=False, use_audio=True,
                         mel_transform=mel_transform)
