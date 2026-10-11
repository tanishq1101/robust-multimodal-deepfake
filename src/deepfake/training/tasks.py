"""Per-branch definitions shared by training and evaluation: model, dataset, forward pass.

Phase 2 registers "visual"; Phase 3 adds "audio"; Phase 4 adds "fusion".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import albumentations as A
import numpy as np
import torch
from torch import nn

from deepfake.datasets import VisualDataset
from deepfake.models import VisualModel


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


@dataclass
class Task:
    build_model: Callable[[dict, bool], nn.Module]
    build_dataset: Callable
    forward: Callable[[nn.Module, dict], torch.Tensor]
    label_key: str


TASKS = {
    "visual": Task(build_visual_model, build_visual_dataset, visual_forward, "video_label"),
}
