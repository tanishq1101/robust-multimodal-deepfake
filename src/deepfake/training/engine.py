"""Shared training machinery for all branches (visual / audio / fusion).

- device selection (CUDA > Apple MPS > CPU), mixed precision on CUDA
- train / evaluate loops with a pluggable `forward_fn(model, batch) -> logits`
- checkpoints every epoch (last.pt) + best model (best.pt), resumable after Colab disconnects
- early stopping on validation AUC (falls back to -loss if AUC is undefined)
- history.csv with per-epoch metrics
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake.evaluation.metrics import compute_metrics

ForwardFn = Callable[[nn.Module, dict], torch.Tensor]


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@dataclass
class TrainState:
    epoch: int = 0                      # number of completed epochs
    best_score: float = -math.inf
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    history: list[dict] = field(default_factory=list)


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, state: TrainState, run_config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "state": asdict(state),
        "run_config": run_config,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }, tmp)
    tmp.replace(path)  # atomic: a disconnect mid-save never corrupts the previous checkpoint


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None,
                    map_location="cpu") -> tuple[TrainState, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return TrainState(**ckpt["state"]), ckpt.get("run_config", {})


def train_one_epoch(model, loader: DataLoader, forward_fn: ForwardFn, criterion, optimizer,
                    device: torch.device, scaler=None, label_key: str = "label",
                    grad_clip: float = 1.0, desc: str = "train") -> float:
    model.train()
    # requires_grad=False alone does not freeze BatchNorm buffers or dropout.
    for module in model.modules():
        params = list(module.parameters())
        if params and not any(p.requires_grad for p in params):
            module.eval()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = to_device(batch, device)
        y = batch[label_key]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss = criterion(forward_fn(model, batch).float(), y)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss; check input tensors and labels")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader: DataLoader, forward_fn: ForwardFn, device: torch.device,
            criterion=None, label_key: str = "label", use_amp: bool = False,
            desc: str = "eval") -> dict:
    """Returns {"labels", "scores" (P(fake)), "loss"} over the whole loader."""
    model.eval()
    labels, scores, total = [], [], 0.0
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = to_device(batch, device)
        y = batch[label_key]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = forward_fn(model, batch).float()
        if criterion is not None:
            total += criterion(logits, y).item() * len(y)
        labels.append(y.cpu().numpy())
        scores.append(torch.sigmoid(logits).cpu().numpy())
    labels = np.concatenate(labels) if labels else np.zeros(0)
    return {"labels": labels, "scores": np.concatenate(scores) if scores else np.zeros(0),
            "loss": total / max(len(labels), 1)}


def monitor_score(metrics: dict, loss: float) -> float:
    auc = metrics.get("auc", float("nan"))
    return auc if auc == auc else -loss  # NaN check: single-class val set


def fit(model, train_loader, val_loader, forward_fn: ForwardFn, criterion, optimizer, scheduler,
        device: torch.device, epochs: int, run_dir: Path, run_config: dict, patience: int = 3,
        label_key: str = "label", resume: bool = True,
        on_epoch_start: Callable[[nn.Module, int], None] | None = None) -> TrainState:
    """Full training loop with checkpointing, resume and early stopping."""
    run_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    last = run_dir / "last.pt"
    state = TrainState()
    if resume and last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        old = saved.get("run_config", {})
        # Epoch limits and patience may change; model/data/optimizer settings may not.
        changed = [k for k in old if k not in {"epochs", "patience"} and old[k] != run_config.get(k)]
        if changed:
            raise ValueError(f"Resume configuration changed: {', '.join(changed)}. Use a new --run-name.")
        state, _ = load_checkpoint(last, model, optimizer, scheduler, scaler, map_location=device)
        rng = saved.get("rng")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        if scheduler and hasattr(scheduler, "T_max"):
            extended = scheduler.T_max != max(epochs, 1)
            scheduler.T_max = max(epochs, 1)
            if extended:
                # An old schedule may already be at LR=0; an extended run must learn.
                for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                    group["lr"] = scheduler.eta_min + (base_lr - scheduler.eta_min) * (
                        1 + math.cos(math.pi * state.epoch / scheduler.T_max)) / 2
        print(f"Resumed from {last} (completed epochs: {state.epoch}, best AUC: {state.best_score:.4f})")
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str))

    while state.epoch < epochs and state.epochs_without_improvement < patience:
        ep = state.epoch + 1
        if on_epoch_start:
            on_epoch_start(model, ep)
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, forward_fn, criterion, optimizer, device,
                                     scaler, label_key, desc=f"epoch {ep}/{epochs}")
        pred = predict(model, val_loader, forward_fn, device, criterion, label_key,
                       use_amp=scaler is not None, desc="val")
        if scheduler:
            scheduler.step()
        m = compute_metrics(pred["labels"], pred["scores"])
        score = monitor_score(m, pred["loss"])
        row = {"epoch": ep, "train_loss": train_loss, "val_loss": pred["loss"],
               "val_auc": m["auc"], "val_eer": m["eer"], "val_acc": m["accuracy"],
               "val_f1": m["f1"], "lr": optimizer.param_groups[0]["lr"],
               "seconds": round(time.time() - t0, 1)}
        state.history.append(row)
        state.epoch = ep
        improved = score > state.best_score
        if improved:
            state.best_score, state.best_epoch = score, ep
            state.epochs_without_improvement = 0
        else:
            state.epochs_without_improvement += 1
        # Save best first, then last: if interrupted in between, resume still sees a consistent state.
        if improved:
            save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, scaler, state, run_config)
        save_checkpoint(last, model, optimizer, scheduler, scaler, state, run_config)
        pd.DataFrame(state.history).to_csv(run_dir / "history.csv", index=False)
        print(f"epoch {ep}: train_loss {train_loss:.4f} | val_loss {pred['loss']:.4f} "
              f"| val_auc {m['auc']:.4f} | val_acc {m['accuracy']:.4f}{'  *best*' if improved else ''}")

    if state.epochs_without_improvement >= patience:
        print(f"Early stopping: no improvement for {patience} epochs.")
    print(f"Best epoch {state.best_epoch} (score {state.best_score:.4f}) -> {run_dir / 'best.pt'}")
    return state
