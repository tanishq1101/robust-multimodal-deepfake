"""Fusion of the visual and audio branches.

LateFusion          weighted average of the two branch probabilities (no extra training).
IntermediateFusion  concatenates branch embeddings -> MLP. Missing modalities (no face /
                    no audio track) are zeroed and flagged, so the model learns to cope.
"""

from __future__ import annotations

import torch
from torch import nn

from deepfake.models.audio import AudioModel
from deepfake.models.visual import VisualModel


class MultimodalModel(nn.Module):
    def __init__(self, visual: VisualModel, audio: AudioModel, fusion: str = "intermediate",
                 dropout: float = 0.3, late_weight: float = 0.5):
        super().__init__()
        self.visual, self.audio, self.fusion = visual, audio, fusion
        self.late_weight = late_weight  # weight of the visual branch in late fusion
        d = visual.embed_dim + audio.embed_dim + 2  # +2 modality-present flags
        self.mlp = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1)
        )

    def forward(self, frames, frame_mask, mel, has_video, has_audio) -> dict:
        v = self.visual(frames, frame_mask)
        a = self.audio(mel)
        hv, ha = has_video.float(), has_audio.float()
        if self.fusion == "late":
            pv, pa = torch.sigmoid(v["logit"]), torch.sigmoid(a["logit"])
            wv = self.late_weight * hv
            wa = (1 - self.late_weight) * ha
            p = (wv * pv + wa * pa) / (wv + wa).clamp(min=1e-6)
            logit = torch.logit(p.clamp(1e-6, 1 - 1e-6))
        else:
            z = torch.cat([v["embedding"] * hv[:, None], a["embedding"] * ha[:, None],
                           hv[:, None], ha[:, None]], dim=1)
            logit = self.mlp(z).squeeze(-1)
        return {"logit": logit, "visual_logit": v["logit"], "audio_logit": a["logit"],
                "frame_logits": v["frame_logits"]}


def build_model(cfg: dict, pretrained: bool | None = None) -> MultimodalModel:
    m = cfg["model"]
    pre = m["pretrained"] if pretrained is None else pretrained
    return MultimodalModel(
        VisualModel(m["visual_backbone"], pre, m["dropout"]),
        AudioModel(m["audio_backbone"], pre, m["dropout"]),
        fusion=m["fusion"], dropout=m["dropout"],
    )
