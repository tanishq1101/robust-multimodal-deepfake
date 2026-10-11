"""Audio branch: ResNet-18 over a log-mel spectrogram treated as a 1-channel image."""

from __future__ import annotations

import torch
import torchvision
from torch import nn


class AudioModel(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, backbone)(weights=weights)
        self.embed_dim = net.fc.in_features  # 512 for resnet18
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.embed_dim, 1))

    def forward(self, mel: torch.Tensor) -> dict:
        """mel [B, 1, n_mels, frames] -> logit [B], embedding [B, D]"""
        # ImageNet weights expect 3 channels: repeat the spectrogram.
        embedding = self.backbone(mel.expand(-1, 3, -1, -1))
        return {"logit": self.head(embedding).squeeze(-1), "embedding": embedding}
