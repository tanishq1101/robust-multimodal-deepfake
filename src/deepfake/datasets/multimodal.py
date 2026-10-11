"""PyTorch dataset over a processed index (output of deepfake.preprocessing.run).

Each item is a dict:
  frames       [T, 3, H, W] float  ImageNet-normalised face crops (zeros if no video)
  frame_mask   [T] bool            which of the T slots hold a real frame
  mel          [1, n_mels, F] float standardised log-mel window (zeros if no audio)
  has_video, has_audio            bool flags (for missing-modality handling)
  label, video_label, audio_label  float (-1 = unknown)

Transforms (all optional, numpy in -> numpy out):
  clip_transform   whole face clip [N, H, W, 3] uint8   (e.g. H.264 / JPEG degradations)
  frame_transform  single face crop [H, W, 3] uint8     (e.g. flip, brightness)
  wav_transform    16 kHz waveform float32              (e.g. MP3, noise). When set, the
                   spectrogram is recomputed from the stored wav/*.flac instead of mel/*.npy
  mel_transform    log-mel [n_mels, F]                  (e.g. SpecAugment)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import soundfile as sf

from deepfake.preprocessing.audio import crop_or_pad, log_mel
from deepfake.preprocessing.faces import load_faces

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def face_to_tensor(face: np.ndarray) -> torch.Tensor:
    """RGB uint8 face crop [H, W, 3] -> ImageNet-normalised float tensor [3, H, W]."""
    x = (face.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def standardise_mel(mel: np.ndarray) -> np.ndarray:
    """Per-clip standardisation of a log-mel window (zero mean, unit variance)."""
    return (mel - mel.mean()) / (mel.std() + 1e-5)


class MultimodalDataset(Dataset):
    def __init__(self, index_csv: str | Path, split: str | None, cfg: dict, train: bool = False,
                 use_video: bool = True, use_audio: bool = True,
                 frame_transform: Callable[[np.ndarray], np.ndarray] | None = None,
                 mel_transform: Callable[[np.ndarray], np.ndarray] | None = None,
                 frames_per_item: int | None = None,
                 clip_transform: Callable[[np.ndarray], np.ndarray] | None = None,
                 wav_transform: Callable[[np.ndarray], np.ndarray] | None = None):
        """frames_per_item: in training, randomly pick this many of the stored frames per
        video (faster epochs + augmentation). Evaluation always uses every stored frame."""
        index_csv = Path(index_csv)
        self.root = index_csv.parent
        df = pd.read_csv(index_csv, keep_default_na=False)
        df = df[df["error"] == ""]
        if split:
            df = df[df["split"] == split]
        if not use_audio:
            df = df[df["faces_path"] != ""]
        if not use_video:
            df = df[df["mel_path"] != ""]
        self.df = df.reset_index(drop=True)
        self.train = train
        self.use_video, self.use_audio = use_video, use_audio
        self.stored_frames = cfg["video"]["num_frames"]
        self.num_frames = frames_per_item if (train and frames_per_item) else self.stored_frames
        self.face_size = cfg["video"]["face_size"]
        a = cfg["audio"]
        self.audio_cfg = a
        self.n_mels = a["n_mels"]
        self.mel_frames = int(a["window_seconds"] * a["sample_rate"] / a["hop_length"]) + 1
        self.frame_transform, self.mel_transform = frame_transform, mel_transform
        self.clip_transform, self.wav_transform = clip_transform, wav_transform

    @staticmethod
    def _rng() -> np.random.Generator:
        # Draw from torch's RNG, which DataLoader seeds differently in every worker
        # (a numpy generator created in __init__ would repeat across workers).
        return np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))

    def __len__(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> np.ndarray:
        return self.df["label"].to_numpy()

    def _video(self, rel: str, face_detection: bool = True) -> tuple[torch.Tensor, torch.Tensor, bool]:
        T, S = self.num_frames, self.face_size
        frames = torch.zeros(T, 3, S, S)
        mask = torch.zeros(T, dtype=torch.bool)
        if not (self.use_video and rel):
            return frames, mask, False
        faces, detected = load_faces(self.root / rel)
        # Match raw-file inference: fallback crops alone are not face evidence.
        if face_detection and not detected.any():
            return frames, mask, False
        if len(faces) > T:
            if self.num_frames < self.stored_frames:  # training subsample, keep temporal order
                faces = faces[np.sort(self._rng().choice(len(faces), T, replace=False))]
            else:
                faces = faces[:T]
        if self.clip_transform and len(faces):
            faces = self.clip_transform(faces)
        for i, f in enumerate(faces):
            if self.frame_transform:
                f = self.frame_transform(f)
            frames[i] = face_to_tensor(f)
            mask[i] = True
        return frames, mask, len(faces) > 0

    def _mel_from_wav(self, rel: str) -> np.ndarray:
        """Crop a window from the stored waveform, degrade it, compute the log-mel."""
        if not rel:
            raise RuntimeError("wav_transform needs wav/*.flac files: re-run deepfake.preprocessing.run "
                               "on this dataset (it only adds the missing files)")
        a = self.audio_cfg
        wav, _ = sf.read(str(self.root / rel), dtype="float32")
        n = int(a["window_seconds"] * a["sample_rate"])
        if len(wav) > n:
            start = int(self._rng().integers(0, len(wav) - n + 1)) if self.train else (len(wav) - n) // 2
            wav = wav[start:start + n]
        wav = self.wav_transform(wav)
        return log_mel(wav, a["sample_rate"], a["n_fft"], a["win_length"], a["hop_length"],
                       a["n_mels"]).astype(np.float32)

    def _audio(self, rel: str, wav_rel: str = "") -> tuple[torch.Tensor, bool]:
        if not (self.use_audio and rel):
            return torch.zeros(1, self.n_mels, self.mel_frames), False
        if self.wav_transform:
            mel = self._mel_from_wav(wav_rel)
        else:
            mel = np.load(self.root / rel).astype(np.float32)
        if mel.shape[1] == 0:
            return torch.zeros(1, self.n_mels, self.mel_frames), False
        mel = crop_or_pad(mel, self.mel_frames, random=self.train, rng=self._rng())
        if self.mel_transform:
            mel = self.mel_transform(mel)
        return torch.from_numpy(standardise_mel(mel))[None], True

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[i]
        frames, mask, has_v = self._video(r["faces_path"], bool(r.get("face_detection", True)))
        mel, has_a = self._audio(r["mel_path"], r.get("wav_path", ""))
        return {
            "frames": frames, "frame_mask": mask, "mel": mel,
            "has_video": torch.tensor(has_v), "has_audio": torch.tensor(has_a),
            "label": torch.tensor(float(r["label"])),
            "video_label": torch.tensor(float(r["video_label"])),
            "audio_label": torch.tensor(float(r["audio_label"])),
        }
