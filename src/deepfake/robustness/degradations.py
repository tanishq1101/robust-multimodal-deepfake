"""Degradations for robustness training and the stress-test suite (Phase 5).

Video degradations act on a face-crop clip: uint8 RGB array [N, H, W, 3].
Audio degradations act on a mono float32 waveform at 16 kHz.

  name             modality  severity levels (mild -> severe)
  jpeg             video     quality 90, 50, 20
  h264             video     CRF 23, 30, 40          (clip re-encoded with libx264)
  gaussian_blur    video     sigma 1, 2, 3 px
  gaussian_noise   video     std 5, 15, 30 (0-255 scale)
  downscale        video     factor 2, 3, 4          (down- then up-sampled)
  audio_noise      audio     SNR 20, 10, 5 dB        (white noise)
  mp3              audio     64k, 32k, 16k bit/s     (libmp3lame round trip)

Note: H.264 / JPEG are applied to the stored 224x224 face crops, not the original full
frames. This approximates platform re-compression of the face region; it is cheaper and
needs no raw video, but is not identical to compressing the full frame before detection.

`Degradation(name, level)` is a picklable callable (safe for DataLoader workers);
`RandomDegradation` picks a random degradation + level for robust training.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

SEVERITIES: dict[str, list] = {
    "jpeg": [90, 50, 20],
    "h264": [23, 30, 40],
    "gaussian_blur": [1.0, 2.0, 3.0],
    "gaussian_noise": [5.0, 15.0, 30.0],
    "downscale": [2, 3, 4],
    "audio_noise": [20.0, 10.0, 5.0],
    "mp3": ["64k", "32k", "16k"],
}
VIDEO_DEGRADATIONS = ("jpeg", "h264", "gaussian_blur", "gaussian_noise", "downscale")
AUDIO_DEGRADATIONS = ("audio_noise", "mp3")


def _rng() -> np.random.Generator:
    # torch's RNG is seeded per DataLoader worker (see datasets.multimodal)
    return np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))


# --------------------------------------------------------------------------- video

def jpeg(clip: np.ndarray, quality: int) -> np.ndarray:
    out = []
    for f in clip:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        out.append(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return np.stack(out)


def h264(clip: np.ndarray, crf: int, fps: int = 8) -> np.ndarray:
    """Encode the clip as H.264 at the given CRF and decode it back (same shape)."""
    n, h, w, _ = clip.shape
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "clip.mp4"
        enc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", str(crf),
             "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)],
            input=np.ascontiguousarray(clip).tobytes(), capture_output=True)
        if enc.returncode != 0:
            raise RuntimeError(enc.stderr.decode()[-500:])
        dec = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=True)
    frames = np.frombuffer(dec.stdout, np.uint8).reshape(-1, h, w, 3)
    if len(frames) < n:  # pad if the decoder dropped trailing frames
        frames = np.concatenate([frames, np.repeat(frames[-1:], n - len(frames), 0)])
    return frames[:n].copy()


def gaussian_blur(clip: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack([cv2.GaussianBlur(f, (0, 0), sigmaX=float(sigma)) for f in clip])


def gaussian_noise(clip: np.ndarray, std: float) -> np.ndarray:
    noisy = clip.astype(np.float32) + _rng().normal(0, std, clip.shape).astype(np.float32)
    return noisy.clip(0, 255).astype(np.uint8)


def downscale(clip: np.ndarray, factor: int) -> np.ndarray:
    h, w = clip.shape[1:3]
    small = (max(1, w // int(factor)), max(1, h // int(factor)))
    return np.stack([cv2.resize(cv2.resize(f, small, interpolation=cv2.INTER_AREA), (w, h),
                                interpolation=cv2.INTER_LINEAR) for f in clip])


# --------------------------------------------------------------------------- audio

def audio_noise(wav: np.ndarray, snr_db: float) -> np.ndarray:
    power = float(np.mean(wav ** 2))
    if power == 0 or len(wav) == 0:
        return wav
    noise_power = power / (10 ** (snr_db / 10))
    return (wav + _rng().normal(0, np.sqrt(noise_power), wav.shape)).astype(np.float32)


def mp3(wav: np.ndarray, bitrate: str, sample_rate: int = 16000) -> np.ndarray:
    """MP3 encode + decode via ffmpeg/libmp3lame; output has the same length as the input."""
    if len(wav) == 0:
        return wav
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a.mp3"
        enc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(sample_rate), "-ac", "1",
             "-i", "-", "-c:a", "libmp3lame", "-b:a", str(bitrate), str(path)],
            input=wav.astype(np.float32).tobytes(), capture_output=True)
        if enc.returncode != 0:
            raise RuntimeError(enc.stderr.decode()[-500:])
        dec = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "f32le", "-ar", str(sample_rate),
             "-ac", "1", "-"], capture_output=True, check=True)
    out = np.frombuffer(dec.stdout, np.float32)
    return np.pad(out, (0, max(0, len(wav) - len(out))))[: len(wav)].copy()


# --------------------------------------------------------------------------- callables

FUNCTIONS = {
    "jpeg": jpeg, "h264": h264, "gaussian_blur": gaussian_blur, "gaussian_noise": gaussian_noise,
    "downscale": downscale, "audio_noise": audio_noise, "mp3": mp3,
}


def modality_of(name: str) -> str:
    return "video" if name in VIDEO_DEGRADATIONS else "audio"


class Degradation:
    """degrade = Degradation("jpeg", 2)  -> applies quality SEVERITIES["jpeg"][2] (= 20).
    `level` is the index into SEVERITIES (0 = mild, 2 = severe)."""

    def __init__(self, name: str, level: int):
        if name not in FUNCTIONS:
            raise ValueError(f"unknown degradation {name!r}; choose from {list(FUNCTIONS)}")
        self.name, self.level = name, int(level)
        self.param = SEVERITIES[name][self.level]
        self.modality = modality_of(name)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return FUNCTIONS[self.name](x, self.param)

    def __repr__(self) -> str:
        return f"Degradation({self.name}={self.param})"


class RandomDegradation:
    """Robust-training augmentation: with probability p apply one random degradation of the
    given modality at a random severity level."""

    def __init__(self, modality: str, p: float = 0.5, names: tuple[str, ...] | None = None):
        self.names = names or (VIDEO_DEGRADATIONS if modality == "video" else AUDIO_DEGRADATIONS)
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rng = _rng()
        if rng.random() >= self.p:
            return x
        name = self.names[int(rng.integers(len(self.names)))]
        return Degradation(name, int(rng.integers(len(SEVERITIES[name]))))(x)
