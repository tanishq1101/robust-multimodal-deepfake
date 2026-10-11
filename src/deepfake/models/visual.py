"""Visual branch: per-frame CNN on face crops, averaged over the frames of a video."""

from __future__ import annotations

import timm
import torch
from torch import nn


class VisualModel(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features  # 1280 for efficientnet_b0
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.embed_dim, 1))

    def forward(self, frames: torch.Tensor, frame_mask: torch.Tensor | None = None) -> dict:
        """frames [B, T, 3, H, W], frame_mask [B, T] -> logit [B], embedding [B, D], frame_logits [B, T]"""
        B, T = frames.shape[:2]
        if frame_mask is None:
            frame_mask = torch.ones(B, T, dtype=torch.bool, device=frames.device)
        valid = frame_mask.flatten().bool()
        if valid.any():
            encoded = self.backbone(frames.flatten(0, 1)[valid])
            feats = encoded.new_zeros(B * T, self.embed_dim)
            feats[valid] = encoded
        else:
            feats = frames.new_zeros(B * T, self.embed_dim)
        feats = feats.view(B, T, -1)
        frame_logits = self.head(feats).squeeze(-1)
        w = frame_mask.float().unsqueeze(-1)
        embedding = (feats * w).sum(1) / w.sum(1).clamp(min=1)
        logit = (frame_logits * w.squeeze(-1)).sum(1) / w.squeeze(-1).sum(1).clamp(min=1)
        return {"logit": logit, "embedding": embedding, "frame_logits": frame_logits}
