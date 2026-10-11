"""Phase 5 stress test: how much do models degrade under compression, blur, noise, MP3 and
missing modalities?

For every model: clean test set, then every applicable degradation at every severity level
(video degradations for visual/fusion models, audio degradations for audio/fusion models),
plus video-only / audio-only conditions for fusion models.

Usage:
  python -m deepfake.evaluation.robustness_eval --dataset fakeavceleb \\
      --model visual=<ckpts>/visual_fakeavceleb/best.pt \\
      --model visual_robust=<ckpts>/visual_fakeavceleb_robust/best.pt
  # compare fusion with single branches against the overall label:
  python -m deepfake.evaluation.robustness_eval --label overall --model ... --model ...
  # quick run on a subset:
  python -m deepfake.evaluation.robustness_eval --limit 200 --degradations jpeg mp3 --model ...

Writes to --out (default <data_root>/outputs/robustness_<dataset>_<split>/):
  robustness.csv   one row per (model, condition): AUC, EER, accuracy, F1, drop vs clean
  robustness.md    AUC table (conditions x models) + mean AUC drop per model
  robustness.png   AUC vs severity, one panel per degradation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from deepfake.config import get_path, load_config  # noqa: E402
from deepfake.evaluation.metrics import compute_metrics  # noqa: E402
from deepfake.robustness.degradations import (AUDIO_DEGRADATIONS, SEVERITIES,  # noqa: E402
                                              VIDEO_DEGRADATIONS, Degradation)

TASK_MODALITIES = {"visual": {"video"}, "audio": {"audio"}, "fusion": {"video", "audio"}}


def conditions_for(task: str, degradations: list[str], levels: list[int], missing: bool) -> list[dict]:
    """All test conditions applicable to a model of the given task."""
    mods = TASK_MODALITIES[task]
    conds = [{"condition": "clean", "degradation": "clean", "level": -1, "param": ""}]
    for name in degradations:
        kind = "video" if name in VIDEO_DEGRADATIONS else "audio"
        if kind not in mods:
            continue
        for lv in levels:
            if lv < len(SEVERITIES[name]):
                param = SEVERITIES[name][lv]
                conds.append({"condition": f"{name}@{param}", "degradation": name, "level": lv, "param": param})
    if missing and task == "fusion":
        conds += [{"condition": "video_only", "degradation": "missing_modality", "level": 0, "param": "no audio"},
                  {"condition": "audio_only", "degradation": "missing_modality", "level": 1, "param": "no video"}]
    return conds


def evaluate_model(name: str, checkpoint: Path, index: Path, split: str, cfg: dict, label: str,
                   degradations: list[str], levels: list[int], device, workers: int, batch_size: int,
                   limit: int | None, missing: bool) -> list[dict]:
    from deepfake.training.engine import get_device, predict
    from deepfake.training.tasks import TASKS

    device = torch.device(device) if isinstance(device, str) else (device or get_device())
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rc = ckpt["run_config"]
    cfg = {**cfg, **rc.get("preprocessing", {})}
    task = TASKS[rc["task"]]
    model = task.build_model(rc, False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    data_task = "fusion" if label == "overall" else rc["task"]
    label_key = "label" if label == "overall" else task.label_key

    rows = []
    for c in conditions_for(rc["task"], degradations, levels, missing):
        ds = TASKS[data_task].build_dataset(index, split, cfg, False, rc)
        if limit and len(ds) > limit:
            ds.df = ds.df.sample(limit, random_state=cfg["seed"]).reset_index(drop=True)
        if c["degradation"] in VIDEO_DEGRADATIONS:
            ds.clip_transform = Degradation(c["degradation"], c["level"])
        elif c["degradation"] in AUDIO_DEGRADATIONS:
            if "wav_path" not in ds.df or (ds.df["wav_path"] == "").all():
                print(f"[{name}] skipping {c['condition']}: no wav/ files (re-run preprocessing)")
                continue
            ds.wav_transform = Degradation(c["degradation"], c["level"])
        elif c["condition"] == "video_only":
            ds.use_audio = False
        elif c["condition"] == "audio_only":
            ds.use_video = False
        torch.manual_seed(cfg["seed"])  # identical random noise for every model
        pred = predict(model, DataLoader(ds, batch_size, num_workers=workers), task.forward, device,
                       label_key=label_key, use_amp=device.type == "cuda", desc=f"{name} {c['condition']}")
        m = compute_metrics(pred["labels"], pred["scores"])
        rows.append({"model": name, **c, "n": m["n"], "auc": m["auc"], "eer": m["eer"],
                     "accuracy": m["accuracy"], "f1": m["f1"]})
        print(f"[{name}] {c['condition']:22s} AUC {m['auc']:.4f}  EER {m['eer']:.4f}  ACC {m['accuracy']:.4f}")
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[df.condition == "clean"].set_index("model")["auc"]
    df["auc_drop"] = df["model"].map(clean) - df["auc"]
    return df


def save_outputs(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "robustness.csv", index=False)
    order = list(dict.fromkeys(df.condition))
    table = df.pivot(index="condition", columns="model", values="auc").loc[order, list(dict.fromkeys(df.model))]
    degraded = df[~df.condition.isin(["clean", "video_only", "audio_only"])]
    drop = degraded.groupby("model")["auc_drop"].agg(mean_auc_drop="mean", worst_auc_drop="max")
    md = ["## AUC per condition", "", table.round(4).to_markdown(), "",
          "## AUC drop vs clean (degradations only)", "", drop.round(4).to_markdown()]
    (out / "robustness.md").write_text("\n".join(md) + "\n")

    degs = [d for d in dict.fromkeys(df.degradation) if d not in ("clean", "missing_modality")]
    if not degs:
        return
    cols = min(4, len(degs))
    rows = (len(degs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for ax, deg in zip(axes.flat, degs):
        for model, g in df.groupby("model", sort=False):
            clean = g[g.condition == "clean"]
            d = g[g.degradation == deg].sort_values("level")
            if d.empty:
                continue
            x = [0] + [lv + 1 for lv in d.level]
            ax.plot(x, list(clean.auc) + list(d.auc), "o-", label=model)
            ax.set_xticks(x, ["clean"] + [str(p) for p in d.param], fontsize=8)
        ax.set(title=deg, ylim=(0, 1.02), ylabel="AUC")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(degs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    p.add_argument("--dataset", default="fakeavceleb")
    p.add_argument("--index")
    p.add_argument("--split", default="test")
    p.add_argument("--label", choices=["native", "overall"], default="native",
                   help="native: each model's own label (video_label / audio_label / label); "
                        "overall: everyone scored on 'fake if face or voice is fake'")
    p.add_argument("--degradations", nargs="+", default=list(SEVERITIES), choices=list(SEVERITIES))
    p.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2], help="severity indices (0=mild)")
    p.add_argument("--no-missing", action="store_true", help="skip video-only / audio-only conditions")
    p.add_argument("--limit", type=int, help="random subset of the split (H.264 / MP3 are slow)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out")
    p.add_argument("--device")
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args(argv)
    cfg = load_config()
    index = Path(args.index) if args.index else get_path(cfg, "processed", args.dataset, "index.csv")
    out = Path(args.out) if args.out else get_path(cfg, "outputs", f"robustness_{args.dataset}_{args.split}")

    rows = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            p.error(f"--model must be NAME=CHECKPOINT, got {spec!r}")
        rows += evaluate_model(name, Path(path), index, args.split, cfg, args.label, args.degradations,
                               args.levels, args.device, args.workers, args.batch_size, args.limit,
                               not args.no_missing)
    df = summarize(pd.DataFrame(rows))
    save_outputs(df, out)
    print("\n" + (out / "robustness.md").read_text())
    print(f"Saved to {out}")
    return df


if __name__ == "__main__":
    main()
