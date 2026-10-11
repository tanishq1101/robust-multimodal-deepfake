"""Fusion of the visual and audio branches.

late          combine the two branch probabilities with a fixed rule (no extra training):
                weighted  w * p_video + (1 - w) * p_audio
                max       max(p_video, p_audio)
                noisy_or  1 - (1 - p_video)(1 - p_audio)
              The overall label is "fake if the face OR the voice is fake", so max / noisy-OR
              often beat a plain average; the rule and weight are tuned on validation.
intermediate  concatenates branch embeddings + modality-present flags -> MLP.

Missing modalities (no face / no audio track) are excluded from late fusion and zeroed +
flagged in intermediate fusion. With `modality_dropout` > 0 the model randomly hides one
modality during training so it learns to cope when one is missing.
"""

from __future__ import annotations

import torch
from torch import nn

from deepfake.models.audio import AudioModel
from deepfake.models.visual import VisualModel

LATE_RULES = ("weighted", "max", "noisy_or")


def late_combine(pv: torch.Tensor, pa: torch.Tensor, hv: torch.Tensor, ha: torch.Tensor,
                 rule: str = "weighted", weight: float = 0.5) -> torch.Tensor:
    """Combine branch probabilities; hv/ha (0/1 floats) mark which modalities are present."""
    pv0, pa0 = pv * hv, pa * ha  # an absent modality contributes "real" (0) to max / noisy-OR
    if rule == "max":
        p = torch.maximum(pv0, pa0)
    elif rule == "noisy_or":
        p = 1 - (1 - pv0) * (1 - pa0)
    elif rule == "weighted":
        if not 0 <= weight <= 1:
            raise ValueError("late fusion weight must be between 0 and 1")
        wv, wa = weight * hv, (1 - weight) * ha
        p = (wv * pv + wa * pa) / (wv + wa).clamp(min=1e-6)
        # At weights 0/1 the sole available branch must still count.
        p = torch.where((hv > 0) & (ha == 0), pv, p)
        p = torch.where((ha > 0) & (hv == 0), pa, p)
    else:
        raise ValueError(f"unknown late fusion rule: {rule}")
    # neither modality present -> undecided
    return torch.where((hv + ha) > 0, p, torch.full_like(p, 0.5))


class MultimodalModel(nn.Module):
    def __init__(self, visual: VisualModel, audio: AudioModel, fusion: str = "intermediate",
                 dropout: float = 0.3, late_weight: float = 0.5, late_rule: str = "weighted",
                 modality_dropout: float = 0.0):
        super().__init__()
        self.visual, self.audio, self.fusion = visual, audio, fusion
        self.late_weight = late_weight  # weight of the visual branch for the "weighted" rule
        self.late_rule = late_rule
        self.modality_dropout = modality_dropout
        d = visual.embed_dim + audio.embed_dim + 2  # +2 modality-present flags
        self.mlp = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1)
        )

    def _drop_modality(self, has_video: torch.Tensor, has_audio: torch.Tensor):
        """Training only: for a fraction of samples that have both, hide one at random."""
        both = has_video & has_audio
        drop = (torch.rand(both.shape, device=both.device) < self.modality_dropout) & both
        hide_video = torch.rand(both.shape, device=both.device) < 0.5
        return has_video & ~(drop & hide_video), has_audio & ~(drop & ~hide_video)

    def forward(self, frames, frame_mask, mel, has_video, has_audio) -> dict:
        has_video, has_audio = has_video.bool(), has_audio.bool()
        if self.training and self.modality_dropout > 0:
            has_video, has_audio = self._drop_modality(has_video, has_audio)
        # Missing inputs must not enter BatchNorm statistics or waste a CNN pass.
        v = {"logit": frames.new_zeros(len(frames)),
             "embedding": frames.new_zeros(len(frames), self.visual.embed_dim),
             "frame_logits": frames.new_zeros(frames.shape[:2])}
        a = {"logit": mel.new_zeros(len(mel)),
             "embedding": mel.new_zeros(len(mel), self.audio.embed_dim)}
        if has_video.any():
            encoded = self.visual(frames[has_video], frame_mask[has_video])
            for key, value in encoded.items():
                v[key] = value.new_zeros(v[key].shape).index_copy(
                    0, has_video.nonzero().flatten(), value)
        if has_audio.any():
            encoded = self.audio(mel[has_audio])
            for key, value in encoded.items():
                a[key] = value.new_zeros(a[key].shape).index_copy(
                    0, has_audio.nonzero().flatten(), value)
        hv, ha = has_video.float(), has_audio.float()
        if self.fusion == "late":
            p = late_combine(torch.sigmoid(v["logit"]), torch.sigmoid(a["logit"]), hv, ha,
                             self.late_rule, self.late_weight)
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
