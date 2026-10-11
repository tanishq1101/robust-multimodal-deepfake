"""Degradations for robustness training and the stress-test suite.  (Phase 5)

Video (applied to RGB uint8 face crops / frames):
  jpeg(img, quality)            quality in {90, 50, 20}
  h264(video_path, crf)         re-encode the whole video, crf in {23, 30, 40}
  gaussian_blur(img, sigma)
  gaussian_noise(img, std)
  downscale(img, factor)        down- then up-sample

Audio (applied to float32 waveforms):
  add_noise(wav, snr_db)        snr in {20, 10, 5}
  mp3(wav, sr, bitrate)         bitrate in {64k, 32k}

Missing modality: handled in the dataset by passing use_video / use_audio = False.
"""

from __future__ import annotations

import numpy as np

SEVERITIES = {
    "jpeg": [90, 50, 20],
    "h264": [23, 30, 40],
    "gaussian_blur": [1.0, 2.0, 3.0],
    "gaussian_noise": [5.0, 15.0, 30.0],
    "downscale": [2, 3, 4],
    "audio_noise": [20, 10, 5],
    "mp3": ["64k", "32k"],
}


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    raise NotImplementedError("Phase 5")


def h264(video_path: str, crf: int, out_path: str) -> str:
    raise NotImplementedError("Phase 5")


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    raise NotImplementedError("Phase 5")


def gaussian_noise(img: np.ndarray, std: float) -> np.ndarray:
    raise NotImplementedError("Phase 5")


def downscale(img: np.ndarray, factor: int) -> np.ndarray:
    raise NotImplementedError("Phase 5")


def add_noise(wav: np.ndarray, snr_db: float) -> np.ndarray:
    raise NotImplementedError("Phase 5")


def mp3(wav: np.ndarray, sample_rate: int, bitrate: str) -> np.ndarray:
    raise NotImplementedError("Phase 5")
