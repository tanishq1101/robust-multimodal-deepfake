"""Everything known about a trained model, read from its run folder (for the demo's model page).

A run folder (written by training/runner.py and the Colab notebooks) contains:
  best.pt  config.json  history.csv  curves.png  data.json
  test/metrics.json  test/per_category.csv  test/predictions.csv  test/roc.png
Every file except best.pt is optional; missing parts are simply not shown.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BRANCHES = {
    "audio": "Voice model: ResNet-18 on 128-bin log-mel spectrograms",
    "visual": "Face model: EfficientNet-B0 on 16 face crops per video",
    "fusion": "Face + voice fusion model",
}
DATASETS = {
    "asvspoof": "ASVspoof 2019 Logical Access",
    "fakeavceleb": "FakeAVCeleb v1.2",
    "dfdc": "DFDC train sample",
    "celebdf": "Celeb-DF v2",
}


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, keep_default_na=False) if path.is_file() else None


def _file(path: Path) -> str | None:
    return str(path) if path.is_file() else None


def dataset_name(run_config: dict, data: dict | None) -> str:
    """Dataset key from data.json, or the processed folder name in the index path."""
    key = (data or {}).get("dataset") or Path(run_config.get("index", "")).parent.name
    return DATASETS.get(key, key or "unknown")


def load_run(checkpoint: str | Path, run_config: dict | None = None) -> dict:
    ckpt = Path(checkpoint)
    run_dir = ckpt.parent
    config = run_config or _json(run_dir / "config.json") or {}
    history = _csv(run_dir / "history.csv")
    if history is None:  # older runs: the history is also stored inside the checkpoint
        import torch

        state = torch.load(ckpt, map_location="cpu", weights_only=False).get("state", {})
        history = pd.DataFrame(state.get("history", [])) if state.get("history") else None
    metrics = _json(run_dir / "test" / "metrics.json")
    data = _json(run_dir / "data.json")
    return {
        "name": run_dir.name,
        "checkpoint": str(ckpt),
        "task": config.get("task", "unknown"),
        "branch": BRANCHES.get(config.get("task"), config.get("task", "unknown")),
        "dataset": dataset_name(config, data),
        "config": config,
        "data": data,
        "history": history,
        "metrics": metrics,
        "per_category": _csv(run_dir / "test" / "per_category.csv"),
        "curves": _file(run_dir / "curves.png"),
        "roc": _file(run_dir / "test" / "roc.png"),
        "best_epoch": (metrics or {}).get("best_epoch"),
    }


def split_table(run: dict) -> pd.DataFrame:
    """Train / validation / test clip counts (test counts come from the confusion matrix)."""
    rows = []
    for name, s in ((run["data"] or {}).get("splits") or {}).items():
        rows.append({"Split": name, "Source": s.get("partition", ""), "Clips": s["clips"],
                     "Fake": s["fake"], "Real": s["real"]})
    m = run["metrics"]
    if m and "confusion_matrix" in m and not any(r["Split"] == "test" for r in rows):
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        rows.append({"Split": "test", "Source": m.get("split", ""), "Clips": tn + fp + fn + tp,
                     "Fake": fn + tp, "Real": tn + fp})
    return pd.DataFrame(rows, columns=["Split", "Source", "Clips", "Fake", "Real"])


def category_table(run: dict) -> pd.DataFrame | None:
    """Per-category results (ASVspoof attack / FakeAVCeleb category) in readable units."""
    df = run["per_category"]
    if df is None or df.empty:
        return None
    out = pd.DataFrame({"Category": df["category"], "Clips": df["n"].astype(int)})
    fake = df["fake_fraction"].astype(float) > 0.5
    acc = df["accuracy"].astype(float)
    out["Correct at threshold"] = [f"{100 * a:.1f}% {'detected' if f else 'accepted'}" for a, f in zip(acc, fake)]
    out["Mean P(fake)"] = df["mean_score"].astype(float).round(4)
    auc = pd.to_numeric(df["auc_vs_real"], errors="coerce")
    eer = pd.to_numeric(df["eer_vs_real"], errors="coerce")
    out["AUC vs. real"] = [f"{a:.4f}" if pd.notna(a) else "–" for a in auc]
    out["EER vs. real"] = [f"{100 * e:.2f}%" if pd.notna(e) else "–" for e in eer]
    return out


def config_table(run: dict) -> pd.DataFrame:
    c = run["config"]
    rows = [("Backbone", c.get("backbone", "")),
            ("Training dataset", run["dataset"]), ("Initialised from", c.get("init_from") or "ImageNet weights"),
            ("Max epochs", c.get("epochs")), ("Best epoch", run["best_epoch"]),
            ("Batch size", c.get("batch_size")), ("Learning rate", c.get("lr")),
            ("Weight decay", c.get("weight_decay")), ("Frozen backbone epochs", c.get("freeze_epochs")),
            ("Early-stopping patience", c.get("patience")), ("Dropout", c.get("dropout")),
            ("Augmentation", "on" if c.get("augment") else "off"),
            ("Robust (degradation) training", "on" if c.get("robust_augment") else "off"), ("Seed", c.get("seed"))]
    a = (c.get("preprocessing") or {}).get("audio")
    if a and run["task"] in ("audio", "fusion"):
        rows.append(("Audio input", f"{a['sample_rate'] // 1000} kHz mono, {a['n_mels']}-bin log-mel, "
                                    f"{a['window_seconds']:g} s window (first {a['max_seconds']:g} s)"))
    v = (c.get("preprocessing") or {}).get("video")
    if v and run["task"] in ("visual", "fusion"):
        rows.append(("Video input", f"{v['num_frames']} frames, {v['face_size']}² face crops"))
    return pd.DataFrame([(k, "" if val is None else str(val)) for k, val in rows], columns=["Setting", "Value"])


def history_table(run: dict) -> pd.DataFrame | None:
    h = run["history"]
    if h is None or h.empty:
        return None
    cols = {"epoch": "Epoch", "train_loss": "Train loss", "val_loss": "Val loss", "val_auc": "Val AUC",
            "val_eer": "Val EER", "val_acc": "Val accuracy"}
    out = h[[c for c in cols if c in h]].rename(columns=cols).copy()
    out["Epoch"] = out["Epoch"].astype(int)
    for c in out.columns[1:]:
        values = out[c].astype(float)
        out[c] = [f"{100 * v:.2f}%" for v in values] if c in ("Val EER", "Val accuracy") else [f"{v:.4f}" for v in values]
    return out
