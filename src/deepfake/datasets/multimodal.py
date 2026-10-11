"""PyTorch dataset over a processed index (output of deepfake.preprocessing.run).

Each item is a dict:
  frames       [T, 3, H, W] float  ImageNet-normalised face crops (zeros if no video)
  frame_mask   [T] bool            which of the T slots hold a real frame
  mel          [1, n_mels, F] float standardised log-mel window (zeros if no audio)
  has_video, has_audio            bool flags (for missing-modality handling)
  label, video_label, audio_label  float (-1 = unknown)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from deepfake.preprocessing.audio import crop_or_pad
from deepfake.preprocessing.faces import load_faces

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class MultimodalDataset(Dataset):
    def __init__(self, index_csv: str | Path, split: str | None, cfg: dict, train: bool = False,
                 use_video: bool = True, use_audio: bool = True,
                 frame_transform: Callable[[np.ndarray], np.ndarray] | None = None,
                 mel_transform: Callable[[np.ndarray], np.ndarray] | None = None):
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
        self.num_frames = cfg["video"]["num_frames"]
        self.face_size = cfg["video"]["face_size"]
        a = cfg["audio"]
        self.n_mels = a["n_mels"]
        self.mel_frames = int(a["window_seconds"] * a["sample_rate"] / a["hop_length"]) + 1
        self.frame_transform, self.mel_transform = frame_transform, mel_transform
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> np.ndarray:
        return self.df["label"].to_numpy()

    def _video(self, rel: str) -> tuple[torch.Tensor, torch.Tensor, bool]:
        T, S = self.num_frames, self.face_size
        frames = torch.zeros(T, 3, S, S)
        mask = torch.zeros(T, dtype=torch.bool)
        if not (self.use_video and rel):
            return frames, mask, False
        faces, _ = load_faces(self.root / rel)
        faces = faces[:T]
        for i, f in enumerate(faces):
            if self.frame_transform:
                f = self.frame_transform(f)
            x = (f.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            frames[i] = torch.from_numpy(x.transpose(2, 0, 1))
            mask[i] = True
        return frames, mask, len(faces) > 0

    def _audio(self, rel: str) -> tuple[torch.Tensor, bool]:
        if not (self.use_audio and rel):
            return torch.zeros(1, self.n_mels, self.mel_frames), False
        mel = np.load(self.root / rel).astype(np.float32)
        if mel.shape[1] == 0:
            return torch.zeros(1, self.n_mels, self.mel_frames), False
        mel = crop_or_pad(mel, self.mel_frames, random=self.train, rng=self.rng)
        if self.mel_transform:
            mel = self.mel_transform(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)
        return torch.from_numpy(mel)[None], True

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[i]
        frames, mask, has_v = self._video(r["faces_path"])
        mel, has_a = self._audio(r["mel_path"])
        return {
            "frames": frames, "frame_mask": mask, "mel": mel,
            "has_video": torch.tensor(has_v), "has_audio": torch.tensor(has_a),
            "label": torch.tensor(float(r["label"])),
            "video_label": torch.tensor(float(r["video_label"])),
            "audio_label": torch.tensor(float(r["audio_label"])),
        }
