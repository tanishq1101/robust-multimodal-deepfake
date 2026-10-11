"""Shared plots for training runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_history(run_dir: str | Path) -> Path | None:
    """Loss and validation AUC/accuracy per epoch -> <run_dir>/curves.png"""
    run_dir = Path(run_dir)
    if not (run_dir / "history.csv").exists():
        return None
    h = pd.read_csv(run_dir / "history.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(h.epoch, h.train_loss, "o-", label="train")
    ax[0].plot(h.epoch, h.val_loss, "o-", label="val")
    ax[0].set(title="loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(h.epoch, h.val_auc, "o-", label="val AUC")
    ax[1].plot(h.epoch, h.val_acc, "o-", label="val accuracy")
    ax[1].set(title="validation", xlabel="epoch", ylim=(0, 1.02))
    ax[1].legend()
    fig.tight_layout()
    out = run_dir / "curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
