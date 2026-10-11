"""Phase 4 comparison table: score several models on the SAME test samples against the
SAME overall label (fake if face OR voice is fake).

Every model sees the full multimodal test set, so a visual-only model is (correctly)
penalised for missing fake voices and an audio-only model for missing fake faces.

Usage:
  python -m deepfake.evaluation.compare --dataset fakeavceleb \\
      --model visual_only=<ckpts>/visual_fakeavceleb/best.pt \\
      --model audio_only=<ckpts>/audio_fakeavceleb/best.pt \\
      --model late=<ckpts>/fusion_late_fakeavceleb/best.pt \\
      --model intermediate=<ckpts>/fusion_intermediate_fakeavceleb/best.pt

Writes to --out (default <data_root>/outputs/compare_<dataset>_<split>/):
  comparison.csv        model, n, AUC, EER, accuracy, F1
  per_category.csv      accuracy of every model on every category (RV-RA, RV-FA, FV-RA, FV-FA)
  comparison.md         both tables in Markdown, ready for reports/results.md
  comparison.png        bar chart of AUC and accuracy
  <model>/              full evaluation output for each model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from deepfake.config import get_path, load_config  # noqa: E402
from deepfake.evaluation.evaluate import evaluate_checkpoint  # noqa: E402


def compare(models: dict[str, Path], index: Path, split: str, cfg: dict, out: Path,
            device=None, workers: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, cats = [], []
    for name, ckpt in models.items():
        m = evaluate_checkpoint(ckpt, index, split, cfg, out / name, device=device, workers=workers,
                                label_key="label", dataset_task="fusion")
        if m is None:
            continue
        rows.append({"model": name, "n": m["n"], "auc": m["auc"], "eer": m["eer"],
                     "accuracy": m["accuracy"], "f1": m["f1"]})
        pc = out / name / "per_category.csv"
        if pc.exists():
            c = pd.read_csv(pc, keep_default_na=False)[["category", "accuracy"]]
            cats.append(c.assign(model=name))
    table = pd.DataFrame(rows)
    per_cat = (pd.concat(cats).pivot(index="category", columns="model", values="accuracy")
               [list(table.model)] if cats else pd.DataFrame())
    return table, per_cat


def save_outputs(table: pd.DataFrame, per_cat: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison.csv", index=False)
    md = ["## Overall (label = fake if face or voice is fake)", "",
          table.round(4).to_markdown(index=False)]
    if len(per_cat):
        per_cat.to_csv(out / "per_category.csv")
        md += ["", "## Accuracy per category", "", per_cat.round(4).to_markdown()]
    (out / "comparison.md").write_text("\n".join(md) + "\n")

    fig, ax = plt.subplots(figsize=(1.6 * len(table) + 2, 3.5))
    x = range(len(table))
    ax.bar([i - 0.2 for i in x], table.auc, 0.4, label="AUC")
    ax.bar([i + 0.2 for i in x], table.accuracy, 0.4, label="accuracy")
    ax.set_xticks(list(x), table.model)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Model comparison (same test set, overall label)")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    p.add_argument("--dataset", default="fakeavceleb")
    p.add_argument("--index", help="explicit processed index.csv")
    p.add_argument("--split", default="test")
    p.add_argument("--out")
    p.add_argument("--device")
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args(argv)
    cfg = load_config()
    models = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            p.error(f"--model must be NAME=CHECKPOINT, got {spec!r}")
        models[name] = Path(path)
    index = Path(args.index) if args.index else get_path(cfg, "processed", args.dataset, "index.csv")
    out = Path(args.out) if args.out else get_path(cfg, "outputs", f"compare_{args.dataset}_{args.split}")
    table, per_cat = compare(models, index, args.split, cfg, out, args.device, args.workers)
    save_outputs(table, per_cat, out)
    print("\n" + (out / "comparison.md").read_text())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
