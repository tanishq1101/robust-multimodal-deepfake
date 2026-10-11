"""Evaluate a trained checkpoint on a split of a processed dataset.

Usage:
  python -m deepfake.evaluation.evaluate --checkpoint <run_dir>/best.pt --dataset dfdc --split test
  python -m deepfake.evaluation.evaluate --checkpoint best.pt --index other/index.csv --out results/

Writes to --out (default: <run_dir>/<split>_<dataset>/):
  metrics.json       AUC, EER, ACC, P/R/F1, confusion matrix (+ metrics at the EER threshold)
  predictions.csv    sample_id, category, label, score
  per_category.csv   per category: n, mean score, accuracy; for fake categories also
                     AUC/EER against all real samples (e.g. EER per ASVspoof attack)
  roc.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from deepfake.config import get_path, load_config  # noqa: E402
from deepfake.evaluation.metrics import compute_metrics  # noqa: E402


def plot_roc(labels, scores, auc: float, path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_category(pred: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Break results down by the manifest `category` column.

    Every category gets n / mean score / accuracy. Categories containing fakes additionally
    get AUC and EER computed against *all* real samples, which shows which kinds of fakes
    (attack systems, manipulation types) the model struggles with.
    """
    if "category" not in pred or (pred["category"] == "").all():
        return pd.DataFrame()
    real = pred[pred.label == 0]
    rows = []
    for cat, g in pred.groupby("category"):
        row = {"category": cat, "n": len(g), "fake_fraction": g.label.mean(),
               "mean_score": g.score.mean(),
               "accuracy": ((g.score >= threshold).astype(int) == g.label).mean()}
        fakes = g[g.label == 1]
        if len(fakes) and len(real):
            both = pd.concat([real, fakes])
            m = compute_metrics(both.label, both.score, threshold)
            row.update(auc_vs_real=m["auc"], eer_vs_real=m["eer"])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_checkpoint(checkpoint: str | Path, index: str | Path, split: str, cfg: dict,
                        out_dir: str | Path, device: torch.device | str | None = None,
                        workers: int = 0, batch_size: int = 8) -> dict | None:
    from deepfake.training.engine import get_device, predict
    from deepfake.training.tasks import TASKS

    checkpoint, out_dir = Path(checkpoint), Path(out_dir)
    device = torch.device(device) if isinstance(device, str) else (device or get_device())
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    run_config = ckpt["run_config"]
    task = TASKS[run_config["task"]]

    ds = task.build_dataset(index, split, cfg, False, run_config)
    if len(ds) == 0:
        print(f"No samples in split '{split}' of {index}; skipping evaluation.")
        return None
    model = task.build_model(run_config, False)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    loader = DataLoader(ds, batch_size, shuffle=False, num_workers=workers)
    pred = predict(model, loader, task.forward, device, label_key=task.label_key,
                   use_amp=device.type == "cuda", desc=f"eval {split}")
    metrics = compute_metrics(pred["labels"], pred["scores"])
    metrics.update(split=split, index=str(index), checkpoint=str(checkpoint),
                   best_epoch=ckpt["state"]["best_epoch"])
    if metrics["eer_threshold"] == metrics["eer_threshold"]:  # also report at the EER threshold
        at_eer = compute_metrics(pred["labels"], pred["scores"], threshold=metrics["eer_threshold"])
        metrics["at_eer_threshold"] = {k: at_eer[k] for k in ("accuracy", "precision", "recall", "f1")}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    preds = pd.DataFrame({
        "sample_id": ds.df["sample_id"],
        "category": ds.df["category"] if "category" in ds.df else "",
        "label": pred["labels"].astype(int), "score": pred["scores"],
    })
    preds.to_csv(out_dir / "predictions.csv", index=False)
    cats = per_category(preds)
    if len(cats):
        cats.to_csv(out_dir / "per_category.csv", index=False)
        print(cats.round(4).to_string(index=False))
    if metrics["auc"] == metrics["auc"]:
        plot_roc(pred["labels"], pred["scores"], metrics["auc"], out_dir / "roc.png",
                 f"{run_config['task']} · {split}")
    print(f"[{split}] n={metrics['n']} AUC={metrics['auc']:.4f} EER={metrics['eer']:.4f} "
          f"ACC={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} -> {out_dir}")
    return metrics


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", help="processed dataset name")
    p.add_argument("--index", help="explicit processed index.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--out")
    p.add_argument("--device")
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args(argv)
    if not (args.dataset or args.index):
        p.error("give --dataset or --index")
    cfg = load_config()
    index = Path(args.index) if args.index else get_path(cfg, "processed", args.dataset, "index.csv")
    name = args.dataset or Path(args.index).parent.name
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / f"{args.split}_{name}"
    evaluate_checkpoint(args.checkpoint, index, args.split, cfg, out, args.device, args.workers)


if __name__ == "__main__":
    main()
