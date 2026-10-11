"""Per-branch definitions shared by training and evaluation: model, dataset, forward pass.

  visual  (Phase 2)  EfficientNet on face crops        label: video_label
  audio   (Phase 3)  ResNet-18 on log-mel spectrograms label: audio_label
  fusion  (Phase 4)  visual + audio branches + fusion   label: label (fake if either is fake)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import albumentations as A
import numpy as np
import torch
from torch import nn

from deepfake.datasets import AudioDataset, MultimodalDataset, VisualDataset
from deepfake.models import AudioModel, MultimodalModel, VisualModel


# --------------------------------------------------------------------------- augmentation

class FrameAugment:
    """Light training augmentation for face crops (picklable for DataLoader workers).
    Robustness augmentations (compression, blur, noise) are added in Phase 5."""

    def __init__(self):
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        ])

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.aug(image=img)["image"]


class SpecAugment:
    """SpecAugment-style masking (Park et al., 2019) on a log-mel [n_mels, frames] array:
    a few random frequency bands and time spans are replaced by the spectrogram mean."""

    def __init__(self, freq_masks: int = 2, freq_width: int = 16, time_masks: int = 2,
                 time_width: int = 40):
        self.freq_masks, self.freq_width = freq_masks, freq_width
        self.time_masks, self.time_width = time_masks, time_width

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        mel = mel.copy()
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        fill = mel.mean()
        n_mels, n_frames = mel.shape
        for _ in range(self.freq_masks):
            w = int(rng.integers(0, self.freq_width + 1))
            f0 = int(rng.integers(0, max(1, n_mels - w)))
            mel[f0:f0 + w, :] = fill
        for _ in range(self.time_masks):
            w = int(rng.integers(0, self.time_width + 1))
            t0 = int(rng.integers(0, max(1, n_frames - w)))
            mel[:, t0:t0 + w] = fill
        return mel


# --------------------------------------------------------------------------- visual

def visual_forward(model: VisualModel, batch: dict) -> torch.Tensor:
    return model(batch["frames"], batch["frame_mask"])["logit"]


def build_visual_model(run_config: dict, pretrained: bool) -> nn.Module:
    return VisualModel(run_config["backbone"], pretrained=pretrained, dropout=run_config["dropout"])


def build_visual_dataset(index_csv, split: str, cfg: dict, train: bool, run_config: dict):
    ds = VisualDataset(index_csv, split, cfg, train=train,
                       frame_transform=FrameAugment() if train and run_config.get("augment", True) else None,
                       frames_per_item=run_config.get("frames_per_video"))
    # The visual branch learns "is the FACE fake": e.g. FakeAVCeleb RealVideo-FakeAudio is
    # a real face (video_label 0). Rows without a video label are dropped.
    ds.df = ds.df[ds.df["video_label"].astype(int) >= 0].reset_index(drop=True)
    return ds


# --------------------------------------------------------------------------- audio

def audio_forward(model: AudioModel, batch: dict) -> torch.Tensor:
    return model(batch["mel"])["logit"]


def build_audio_model(run_config: dict, pretrained: bool) -> nn.Module:
    return AudioModel(run_config["backbone"], pretrained=pretrained, dropout=run_config["dropout"])


def build_audio_dataset(index_csv, split: str, cfg: dict, train: bool, run_config: dict):
    ds = AudioDataset(index_csv, split, cfg, train=train,
                      mel_transform=SpecAugment() if train and run_config.get("augment", True) else None)
    # The audio branch learns "is the VOICE fake": FakeAVCeleb FakeVideo-RealAudio has
    # audio_label 0. DFDC has no audio labels (-1) and is dropped here.
    ds.df = ds.df[ds.df["audio_label"].astype(int) >= 0].reset_index(drop=True)
    return ds


# --------------------------------------------------------------------------- fusion

FUSION_INPUTS = ("frames", "frame_mask", "mel", "has_video", "has_audio")


def fusion_forward(model: MultimodalModel, batch: dict) -> torch.Tensor:
    return model(**{k: batch[k] for k in FUSION_INPUTS})["logit"]


def build_fusion_model(run_config: dict, pretrained: bool) -> nn.Module:
    d = run_config["dropout"]
    return MultimodalModel(
        VisualModel(run_config["visual_backbone"], pretrained=pretrained, dropout=d),
        AudioModel(run_config["audio_backbone"], pretrained=pretrained, dropout=d),
        fusion=run_config["fusion"], dropout=d,
        late_weight=run_config.get("late_weight", 0.5),
        late_rule=run_config.get("late_rule", "weighted"),
        modality_dropout=run_config.get("modality_dropout", 0.0),
    )


def build_fusion_dataset(index_csv, split: str, cfg: dict, train: bool, run_config: dict):
    aug = train and run_config.get("augment", True)
    ds = MultimodalDataset(index_csv, split, cfg, train=train,
                           frame_transform=FrameAugment() if aug else None,
                           mel_transform=SpecAugment() if aug else None,
                           frames_per_item=run_config.get("frames_per_video"))
    ds.df = ds.df[ds.df["label"].astype(int) >= 0].reset_index(drop=True)
    return ds


# --------------------------------------------------------------------------- registry

def _branch_head_groups(model: nn.Module, lr: float) -> list[dict]:
    """Single-branch models: backbone at lr, classifier head at 10x."""
    return [{"params": model.backbone.parameters(), "lr": lr},
            {"params": model.head.parameters(), "lr": lr * 10}]


def _fusion_groups(model: MultimodalModel, lr: float) -> list[dict]:
    """Fusion: both (pre-trained) branches at lr, the new fusion MLP at 10x."""
    return [{"params": list(model.visual.parameters()) + list(model.audio.parameters()), "lr": lr},
            {"params": model.mlp.parameters(), "lr": lr * 10}]


@dataclass
class Task:
    build_model: Callable[[dict, bool], nn.Module]
    build_dataset: Callable
    forward: Callable[[nn.Module, dict], torch.Tensor]
    label_key: str
    defaults: dict = field(default_factory=dict)  # task-specific CLI defaults
    # optimizer param groups (lr -> groups) and the parameters frozen during warm-up
    param_groups: Callable[[nn.Module, float], list] = _branch_head_groups
    frozen_params: Callable[[nn.Module], object] = lambda m: m.backbone.parameters()


TASKS = {
    "visual": Task(build_visual_model, build_visual_dataset, visual_forward, "video_label",
                   defaults={"dataset": "dfdc", "backbone_key": "visual_backbone", "batch_size": 8}),
    "audio": Task(build_audio_model, build_audio_dataset, audio_forward, "audio_label",
                  defaults={"dataset": "asvspoof", "backbone_key": "audio_backbone", "batch_size": 32}),
    "fusion": Task(build_fusion_model, build_fusion_dataset, fusion_forward, "label",
                   defaults={"dataset": "fakeavceleb", "batch_size": 8, "freeze_epochs": 2},
                   param_groups=_fusion_groups,
                   frozen_params=lambda m: list(m.visual.parameters()) + list(m.audio.parameters())),
}
