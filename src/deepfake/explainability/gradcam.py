"""Grad-CAM explanations.  (Phase 6)

Planned API (built on the `pytorch-grad-cam` package):
  visual_cam(model, frames)  -> heatmaps [T, H, W] over the face crops,
                                target layer: model.visual.backbone.conv_head
  audio_cam(model, mel)      -> heatmap [n_mels, frames] over the spectrogram,
                                target layer: model.audio.backbone.layer4
  overlay(img, heatmap)      -> RGB image for the report / demo app
"""

from __future__ import annotations


def visual_cam(model, frames):
    raise NotImplementedError("Phase 6")


def audio_cam(model, mel):
    raise NotImplementedError("Phase 6")


def overlay(img, heatmap):
    raise NotImplementedError("Phase 6")
