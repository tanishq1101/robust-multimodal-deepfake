"""Uniform frame sampling from a video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sample_frames(video_path: str | Path, num_frames: int) -> list[np.ndarray]:
    """Return up to `num_frames` RGB frames spread evenly across the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    if total > 0:
        idx = np.linspace(0, total - 1, num=min(num_frames, total)).round().astype(int)
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not frames:  # frame count unreliable for some codecs: decode sequentially instead
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        all_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            all_frames.append(frame)
        if all_frames:
            idx = np.linspace(0, len(all_frames) - 1, num=min(num_frames, len(all_frames)))
            frames = [cv2.cvtColor(all_frames[int(round(i))], cv2.COLOR_BGR2RGB) for i in idx]
    cap.release()
    return frames
