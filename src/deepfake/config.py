"""Configuration loading and data-path resolution (local Mac vs. Google Colab)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/deepfake_project")


class Config(dict):
    """A dict with attribute access for nested sections (cfg.video.num_frames)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(value) if isinstance(value, dict) else value


def data_root() -> Path:
    """Where datasets, processed data and checkpoints live."""
    env = os.environ.get("DEEPFAKE_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    if COLAB_DRIVE_ROOT.parent.exists():
        return COLAB_DRIVE_ROOT
    return REPO_ROOT / "data"


def raw_root() -> Path:
    """Where downloaded (raw) datasets live.

    On Colab this defaults to the VM's local disk (/content/raw, ~100 GB but wiped when the
    session ends) because raw datasets don't fit in a free 15 GB Google Drive. Only the much
    smaller processed data, manifests and checkpoints are kept on Drive.
    """
    env = os.environ.get("DEEPFAKE_RAW_ROOT")
    if env:
        return Path(env).expanduser()
    if not os.environ.get("DEEPFAKE_DATA_ROOT") and COLAB_DRIVE_ROOT.parent.exists():
        return Path("/content/raw")
    return data_root() / "raw"


def load_config(path: str | Path | None = None) -> Config:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = Config(yaml.safe_load(f))
    root = data_root()
    cfg["data_root"] = str(root)
    cfg["paths"] = {k: str(root / v) for k, v in cfg["paths"].items()}
    cfg["paths"]["raw"] = str(raw_root())
    return cfg


def get_path(cfg: Config, name: str, *parts: str) -> Path:
    """get_path(cfg, "raw", "dfdc") -> <data_root>/raw/dfdc"""
    return Path(cfg["paths"][name], *parts)
