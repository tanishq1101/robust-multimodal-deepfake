"""Generic training CLI used by train_visual.py, train_audio.py and train_fusion.py.

Outputs in <data_root>/checkpoints/<run-name>/:
  config.json  last.pt  best.pt  history.csv  curves.png
  test/metrics.json  test/predictions.csv  test/roc.png  test/per_category.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

from deepfake.config import get_path, load_config
from deepfake.evaluation.evaluate import evaluate_checkpoint
from deepfake.evaluation.plots import plot_history
from deepfake.training.engine import fit, get_device, seed_everything
from deepfake.training.tasks import TASKS


def build_parser(task_name: str, cfg: dict, doc: str) -> argparse.ArgumentParser:
    task, t, m = TASKS[task_name], cfg["train"], cfg["model"]
    d = task.defaults
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=d["dataset"], help="processed dataset name")
    p.add_argument("--index", help="explicit processed index.csv (overrides --dataset)")
    p.add_argument("--run-name", help=f"default: {task_name}_<dataset>")
    if "backbone_key" in d:
        p.add_argument("--backbone", default=m[d["backbone_key"]])
    p.add_argument("--init-from", help="start from this checkpoint's weights (e.g. ASVspoof-pretrained "
                                       "best.pt) — optimizer state is not loaded")
    p.add_argument("--epochs", type=int, default=t["epochs"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--lr", type=float, default=t["lr"], help="backbone LR; the head uses 10x")
    p.add_argument("--weight-decay", type=float, default=t["weight_decay"])
    p.add_argument("--dropout", type=float, default=m["dropout"])
    p.add_argument("--freeze-epochs", type=int, default=d.get("freeze_epochs", 1),
                   help="train only the classifier head / fusion layer for the first N epochs")
    p.add_argument("--patience", type=int, default=t["early_stopping_patience"])
    p.add_argument("--workers", type=int, default=t["num_workers"])
    p.add_argument("--device", help="cuda | mps | cpu (default: auto)")
    p.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--robust-augment", action="store_true",
                   help="Phase 5: randomly apply compression/blur/noise (video) and MP3/noise (audio)")
    p.add_argument("--robust-p", type=float, default=0.5, help="probability per sample for --robust-augment")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing last.pt")
    p.add_argument("--seed", type=int, default=cfg["seed"])
    return p


def class_balance_weight(labels) -> torch.Tensor:
    """pos_weight for BCE = n_real / n_fake, so both classes contribute equally."""
    n_fake = float((labels == 1).sum())
    n_real = float((labels == 0).sum())
    return torch.tensor(n_real / n_fake if n_fake and n_real else 1.0)


def validate_index(index: Path) -> None:
    """Reject split leakage before any model is built or training output is written."""
    df = pd.read_csv(index, keep_default_na=False)
    if not df["split"].isin(["train", "val", "test", "excluded"]).all():
        raise ValueError("Index has missing or invalid splits; assign train/val/test first")
    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in processed index")
    df = df[df["split"] != "excluded"]
    for key in ("group", "path"):
        if key in df:
            if key == "group" and (df[key] == "").any():
                raise ValueError("Every training/evaluation row needs a non-empty split group")
            known = df[df[key] != ""]
            if (known.groupby(key)["split"].nunique() > 1).any():
                raise ValueError(f"Split leakage: {key} appears in more than one split")


def validate_labels(ds, key: str, split: str) -> None:
    labels = ds.df[key].to_numpy()
    if len(ds) == 0 or not set(labels) <= {0, 1}:
        raise ValueError(f"{split} requires non-empty binary {key} labels")
    if set(labels) != {0, 1}:
        raise ValueError(f"{split} must contain both real and fake {key} labels")


def run(task_name: str, args: argparse.Namespace, cfg: dict, extra_config: dict | None = None,
        init_fn=None) -> Path:
    """Train one task. init_fn(model) can load pre-trained weights (skipped when resuming)."""
    seed_everything(args.seed)
    task = TASKS[task_name]
    index = Path(args.index) if args.index else get_path(cfg, "processed", args.dataset, "index.csv")
    validate_index(index)
    run_name = args.run_name or f"{task_name}_{args.dataset}"
    run_dir = get_path(cfg, "checkpoints", run_name)
    run_config = {
        "task": task_name, "index": str(index), "backbone": getattr(args, "backbone", None),
        "dropout": args.dropout,
        "augment": not args.no_augment, "robust_augment": args.robust_augment, "robust_p": args.robust_p,
        "init_from": args.init_from, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "freeze_epochs": args.freeze_epochs, "patience": args.patience, "seed": args.seed,
        "preprocessing": {k: dict(cfg[k]) for k in ("video", "audio")},
        **(extra_config or {}),
    }

    train_ds = task.build_dataset(index, "train", cfg, True, run_config)
    val_ds = task.build_dataset(index, "val", cfg, False, run_config)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f"Empty train ({len(train_ds)}) or val ({len(val_ds)}) split for "
                         f"task '{task_name}' in {index}")
    labels = train_ds.df[task.label_key].to_numpy().astype(int)
    validate_labels(train_ds, task.label_key, "train")
    validate_labels(val_ds, task.label_key, "val")
    print(f"[{task_name}] train: {len(train_ds)} ({int((labels == 1).sum())} fake) | val: {len(val_ds)}")

    device = get_device(args.device)
    print(f"device: {device}")
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              drop_last=False, **loader_kw)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, **loader_kw)

    resuming = (run_dir / "last.pt").exists() and not args.no_resume
    branches_loaded = run_config.get("visual_ckpt") and run_config.get("audio_ckpt")
    model = task.build_model(run_config, bool(not args.no_pretrained and not resuming
                                            and not args.init_from and not branches_loaded))
    if not resuming:
        if args.init_from:
            ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            print(f"Initialised weights from {args.init_from}")
        if init_fn:
            init_fn(model)
    model.to(device)

    optimizer = torch.optim.AdamW(task.param_groups(model, args.lr), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_balance_weight(labels).to(device))

    def freeze_schedule(m, epoch):
        trainable = epoch > args.freeze_epochs
        for p in task.frozen_params(m):
            p.requires_grad = trainable
        if epoch == 1 and args.freeze_epochs:
            print(f"Pre-trained part frozen for the first {args.freeze_epochs} epoch(s)")

    fit(model, train_loader, val_loader, task.forward, criterion, optimizer, scheduler, device,
        args.epochs, run_dir, run_config, patience=args.patience, label_key=task.label_key,
        resume=not args.no_resume, on_epoch_start=freeze_schedule)
    plot_history(run_dir)
    evaluate_checkpoint(run_dir / "best.pt", index, "test", cfg, run_dir / "test",
                        device=device, workers=args.workers)
    return run_dir


def main_for(task_name: str, doc: str, argv=None, extra_args=None) -> Path:
    """Entry point helper: extra_args(parser) adds task-specific options, returns the
    names of those options to store in the run config."""
    cfg = load_config()
    p = build_parser(task_name, cfg, doc)
    extra_keys = extra_args(p) if extra_args else []
    args = p.parse_args(argv)
    return run(task_name, args, cfg, {k: getattr(args, k) for k in extra_keys})
