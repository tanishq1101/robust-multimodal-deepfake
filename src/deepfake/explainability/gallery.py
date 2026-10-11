"""Grad-CAM gallery: for every category, the most confident correct predictions and the most
confident mistakes, each rendered as face-frame + spectrogram heatmaps.

Usage:
  python -m deepfake.explainability.gallery --checkpoint <ckpts>/fusion_intermediate_fakeavceleb/best.pt
  python -m deepfake.explainability.gallery --checkpoint <ckpts>/visual_dfdc/best.pt --dataset dfdc --per-category 3

Writes to --out (default <data_root>/outputs/gallery_<run-name>/):
  <category>__<correct|wrong>__<sample>.png   one figure per example
  gallery.csv                                  sample, category, label, P(fake), outcome, file
  index.md                                     all figures, grouped by category
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from deepfake.config import get_path, load_config
from deepfake.explainability.gradcam import explain, render_explanation


def load_model(checkpoint, device):
    from deepfake.inference import load_model as _load  # single implementation, avoids an import cycle
    return _load(checkpoint, device)


def pick_examples(scored: pd.DataFrame, per_category: int) -> pd.DataFrame:
    """Most confident correct + most confident wrong predictions per category."""
    scored = scored.assign(pred=(scored.score >= 0.5).astype(int))
    scored["correct"] = scored.pred == scored.label
    scored["confidence"] = (scored.score - 0.5).abs()
    picks = []
    for _, g in scored.groupby("category", sort=True):
        for ok in (True, False):
            sel = g[g.correct == ok].sort_values("confidence", ascending=False).head(per_category)
            picks.append(sel.assign(outcome="correct" if ok else "wrong"))
    return pd.concat(picks) if picks else scored.head(0)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:80]


def build_gallery(checkpoint: Path, index: Path, split: str, cfg: dict, out: Path, per_category: int = 2,
                  limit: int | None = 200, target: str = "predicted", device=None) -> pd.DataFrame:
    from deepfake.training.engine import get_device, predict
    from deepfake.training.tasks import TASKS

    device = torch.device(device) if isinstance(device, str) else (device or get_device())
    model, rc = load_model(checkpoint, device)
    task = TASKS[rc["task"]]
    ds = task.build_dataset(index, split, cfg, False, rc)
    if limit and len(ds) > limit:
        ds.df = ds.df.sample(limit, random_state=cfg["seed"]).reset_index(drop=True)
    if len(ds) == 0:
        raise SystemExit(f"No samples in split '{split}' of {index}")

    pred = predict(model, DataLoader(ds, batch_size=8), task.forward, device, label_key=task.label_key,
                   desc="scoring")
    cat = ds.df["category"] if "category" in ds.df else pd.Series(["all"] * len(ds))
    scored = pd.DataFrame({"pos": np.arange(len(ds)), "sample_id": ds.df["sample_id"],
                           "category": cat.replace("", "all"), "label": pred["labels"].astype(int),
                           "score": pred["scores"]})
    picks = pick_examples(scored, per_category)

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in picks.itertuples():
        item = ds[int(r.pos)]
        batch = {k: v[None].to(device) for k, v in item.items()}
        exp = explain(model, rc["task"], batch, target=target)
        truth = "FAKE" if r.label == 1 else "REAL"
        fig = render_explanation(batch, exp, title=f"{r.category} · {r.sample_id} · truth {truth} · {r.outcome}")
        fname = f"{_safe(r.category)}__{r.outcome}__{_safe(r.sample_id)}.png"
        fig.savefig(out / fname, dpi=110)
        plt.close(fig)
        rows.append({"sample_id": r.sample_id, "category": r.category, "label": r.label,
                     "p_fake": exp["prob"], "outcome": r.outcome, "explained": exp["target"], "file": fname})
    table = pd.DataFrame(rows)
    table.to_csv(out / "gallery.csv", index=False)

    md = [f"# Grad-CAM gallery: {rc['task']} model", "",
          f"Checkpoint `{checkpoint}` · split `{split}` · {len(ds)} scored samples · "
          f"heatmaps show evidence for the {'predicted class' if target == 'predicted' else target}.", ""]
    for c, g in table.groupby("category", sort=True) if len(table) else []:
        md += [f"## {c}", ""]
        for r in g.itertuples():
            md += [f"**{r.outcome}** · `{r.sample_id}` · P(fake) {r.p_fake:.3f}", "", f"![]({r.file})", ""]
    (out / "index.md").write_text("\n".join(md))
    print(f"{len(table)} explanations -> {out}")
    return table


def main(argv=None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="fakeavceleb")
    p.add_argument("--index")
    p.add_argument("--split", default="test")
    p.add_argument("--per-category", type=int, default=2, help="examples per category and outcome")
    p.add_argument("--limit", type=int, default=200, help="samples scored to choose examples from")
    p.add_argument("--target", choices=["predicted", "fake", "real"], default="predicted")
    p.add_argument("--out")
    p.add_argument("--device")
    args = p.parse_args(argv)
    cfg = load_config()
    index = Path(args.index) if args.index else get_path(cfg, "processed", args.dataset, "index.csv")
    ckpt = Path(args.checkpoint)
    out = Path(args.out) if args.out else get_path(cfg, "outputs", f"gallery_{ckpt.parent.name}")
    return build_gallery(ckpt, index, args.split, cfg, out, args.per_category, args.limit, args.target,
                         args.device)


if __name__ == "__main__":
    main()
