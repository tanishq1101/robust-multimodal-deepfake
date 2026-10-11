"""Phase 2: train the visual branch (EfficientNet-B0 on face crops).

Reads the processed index from notebook 02 / deepfake.preprocessing.run, trains with
early stopping on validation AUC, checkpoints every epoch (resumable), then evaluates the
best checkpoint on the test split.

Usage:
  python -m deepfake.training.train_visual --dataset dfdc
  python -m deepfake.training.train_visual --dataset fakeavceleb --epochs 15 --batch-size 16
  python -m deepfake.training.train_visual --index path/to/index.csv --run-name test --epochs 1

Outputs in <data_root>/checkpoints/<run-name>/:
  config.json  last.pt  best.pt  history.csv  test/metrics.json  test/predictions.csv  test/roc.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from deepfake.config import get_path, load_config
from deepfake.evaluation.evaluate import evaluate_checkpoint
from deepfake.training.engine import fit, get_device, seed_everything
from deepfake.training.tasks import TASKS


def parse_args(argv=None, cfg=None):
    t, m = cfg["train"], cfg["model"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dfdc", help="processed dataset name")
    p.add_argument("--index", help="explicit processed index.csv (overrides --dataset)")
    p.add_argument("--run-name", help="default: visual_<dataset>")
    p.add_argument("--backbone", default=m["visual_backbone"])
    p.add_argument("--epochs", type=int, default=t["epochs"])
    p.add_argument("--batch-size", type=int, default=t["batch_size"], help="videos per batch")
    p.add_argument("--frames-per-video", type=int, default=8,
                   help="random frames per video while training (eval uses all 16)")
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
    return p.parse_args(argv)


def class_balance_weight(labels) -> torch.Tensor:
    """pos_weight for BCE = n_real / n_fake, so both classes contribute equally."""
    n_fake = float((labels == 1).sum())
    n_real = float((labels == 0).sum())
    return torch.tensor(n_real / n_fake if n_fake and n_real else 1.0)


def main(argv=None) -> Path:
    cfg = load_config()
    args = parse_args(argv, cfg)
    seed_everything(args.seed)
    task = TASKS["visual"]

    index = Path(args.index) if args.index else get_path(cfg, "processed", args.dataset, "index.csv")
    run_name = args.run_name or f"visual_{args.dataset}"
    run_dir = get_path(cfg, "checkpoints", run_name)
    run_config = {
        "task": "visual", "index": str(index), "backbone": args.backbone, "dropout": args.dropout,
        "frames_per_video": args.frames_per_video, "augment": not args.no_augment,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "freeze_epochs": args.freeze_epochs,
        "patience": args.patience, "seed": args.seed,
    }

    train_ds = task.build_dataset(index, "train", cfg, True, run_config)
    val_ds = task.build_dataset(index, "val", cfg, False, run_config)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f"Empty train ({len(train_ds)}) or val ({len(val_ds)}) split in {index}")
    labels = train_ds.df[task.label_key].to_numpy()
    print(f"train: {len(train_ds)} videos ({int((labels == 1).sum())} fake) | val: {len(val_ds)} videos")

    device = get_device(args.device)
    print(f"device: {device}")
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size,
                              **loader_kw)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, **loader_kw)

    model = task.build_model(run_config, not args.no_pretrained).to(device)
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

    evaluate_checkpoint(run_dir / "best.pt", index, "test", cfg, run_dir / "test",
                        device=device, workers=args.workers)
    return run_dir


if __name__ == "__main__":
    main()
