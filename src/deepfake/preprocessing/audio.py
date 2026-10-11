"""Audio extraction (ffmpeg) and log-mel spectrograms (librosa)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str | Path, sample_rate: int = 16000, max_seconds: float | None = None) -> np.ndarray:
    """Mono float32 waveform at `sample_rate`. Works for .mp4/.wav/.flac.

    Video files are decoded with ffmpeg straight to raw PCM (no temp files).
    Returns an empty array if the file has no audio track.
    """
    path = Path(path)
    if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    else:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found (Mac: brew install ffmpeg)")
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
               "-ar", str(sample_rate), "-f", "f32le"]
        if max_seconds:
            cmd += ["-t", str(max_seconds)]
        out = subprocess.run(cmd + ["-"], capture_output=True)
        if out.returncode != 0 and not out.stdout:
            return np.zeros(0, np.float32)
        wav = np.frombuffer(out.stdout, dtype=np.float32).copy()
    if max_seconds:
        wav = wav[: int(max_seconds * sample_rate)]
    return wav.astype(np.float32)


def log_mel(wav: np.ndarray, sample_rate: int = 16000, n_fft: int = 512, win_length: int = 400,
            hop_length: int = 160, n_mels: int = 128) -> np.ndarray:
    """Log-mel spectrogram in dB, shape [n_mels, frames], float16 to halve storage."""
    if len(wav) < n_fft:
        wav = np.pad(wav, (0, n_fft - len(wav)))
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        n_mels=n_mels, power=2.0,
    )
    return librosa.power_to_db(mel, ref=1.0, top_db=None).clip(-100, None).astype(np.float16)


def crop_or_pad(mel: np.ndarray, frames: int, random: bool = False,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Fixed-length time window: random crop for training, centre crop for evaluation."""
    t = mel.shape[1]
    if t < frames:
        return np.pad(mel, ((0, 0), (0, frames - t)), constant_values=mel.min() if t else -100)
    start = (rng or np.random.default_rng()).integers(0, t - frames + 1) if random else (t - frames) // 2
    return mel[:, start:start + frames]
