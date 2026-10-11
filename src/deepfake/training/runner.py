"""Generic single-branch training CLI used by train_visual.py and train_audio.py.

Outputs in <data_root>/checkpoints/<run-name>/:
  config.json  last.pt  best.pt  history.csv  curves.png
  test/metrics.json  test/predictions.csv  test/roc.png  test/per_category.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
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
    p.add_argument("--backbone", default=m[d["backbone_key"]])
    p.add_argument("--init-from", help="start from this checkpoint's weights (e.g. ASVspoof-pretrained "
                                       "best.pt) — optimizer state is not loaded")
    p.add_argument("--epochs", type=int, default=t["epochs"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--lr", type=float, default=t["lr"], help="backbone LR; the head uses 10x")
    p.add_argument("--weight-decay", type=float, default=t["weight_decay"])
    p.add_argument("--dropout", type=float, default=m["dropout"])
    p.add_argument("--freeze-epochs", type=int, default=1,
                   help="train only the classifier head for the first N epochs")
    p.add_argument("--patience", type=int, default=t["early_stopping_patience"])
    p.add_argument("--workers", type=int, default=t["num_workers"])
    p.add_argument("--device", help="cuda | mps | cpu (default: auto)")
    p.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing last.pt")
    p.add_argument("--seed", type=int, default=cfg["seed"])
    return p


def class_balance_weight(labels) -> torch.Tensor:
    """pos_weight for BCE = n_real / n_fake, so both classes contribute equally."""
    n_fake = float((labels == 1).sum())
    n_real = float((labels == 0).sum())
    return torch.tensor(n_real / n_fake if n_fake and n_real else 1.0)


def run(task_name: str, args: argparse.Namespace, cfg: dict, extra_config: dict | None = None) -> Path:
    seed_everything(args.seed)
    task = TASKS[task_name]
    index = Path(args.index) if args.index else get_path(cfg, "processed", args.dataset, "index.csv")
    run_name = args.run_name or f"{task_name}_{args.dataset}"
    run_dir = get_path(cfg, "checkpoints", run_name)
    run_config = {
        "task": task_name, "index": str(index), "backbone": args.backbone, "dropout": args.dropout,
        "augment": not args.no_augment, "init_from": args.init_from, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "freeze_epochs": args.freeze_epochs, "patience": args.patience, "seed": args.seed,
        **(extra_config or {}),
    }

    train_ds = task.build_dataset(index, "train", cfg, True, run_config)
    val_ds = task.build_dataset(index, "val", cfg, False, run_config)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f"Empty train ({len(train_ds)}) or val ({len(val_ds)}) split for "
                         f"task '{task_name}' in {index}")
    labels = train_ds.df[task.label_key].to_numpy().astype(int)
    print(f"[{task_name}] train: {len(train_ds)} ({int((labels == 1).sum())} fake) | val: {len(val_ds)}")

    device = get_device(args.device)
    print(f"device: {device}")
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              drop_last=len(train_ds) > args.batch_size, **loader_kw)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, **loader_kw)

    model = task.build_model(run_config, not args.no_pretrained)
    if args.init_from and not (run_dir / "last.pt").exists():
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Initialised weights from {args.init_from}")
    model.to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.lr * 10},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_balance_weight(labels).to(device))

    def freeze_schedule(m, epoch):
        trainable = epoch > args.freeze_epochs
        for p in m.backbone.parameters():
            p.requires_grad = trainable
        if epoch == 1 and args.freeze_epochs:
            print(f"Backbone frozen for the first {args.freeze_epochs} epoch(s)")

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
