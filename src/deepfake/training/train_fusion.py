"""Phase 4: multimodal fusion of the visual and audio branches.

Trains on the overall `label` (fake if the face OR the voice is fake), typically on
FakeAVCeleb. Both branches start from their best Phase 2 / Phase 3 checkpoints.

  intermediate  [visual embedding | audio embedding | presence flags] -> MLP.
                Branches frozen for --freeze-epochs (only the MLP learns), then fine-tuned
                jointly at --lr (MLP at 10x). --modality-dropout randomly hides one modality.
  late          no training: picks the combination rule (weighted / max / noisy_or) and
                visual weight that maximise validation AUC; saves calibration.csv.

Usage:
  python -m deepfake.training.train_fusion --fusion intermediate \\
      --visual-ckpt <data_root>/checkpoints/visual_fakeavceleb/best.pt \\
      --audio-ckpt  <data_root>/checkpoints/audio_fakeavceleb/best.pt
  python -m deepfake.training.train_fusion --fusion late --visual-ckpt ... --audio-ckpt ...

Outputs in <data_root>/checkpoints/<run-name>/ (default run name: fusion_<type>_<dataset>).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake.config import get_path, load_config
from deepfake.evaluation.evaluate import evaluate_checkpoint
from deepfake.evaluation.metrics import compute_metrics
from deepfake.models.fusion import LATE_RULES, late_combine
from deepfake.training.engine import TrainState, get_device, save_checkpoint, seed_everything, to_device
from deepfake.training.runner import build_parser, run
from deepfake.training.tasks import FUSION_INPUTS, TASKS


def _extra(p):
    p.add_argument("--fusion", choices=["intermediate", "late"], default="intermediate")
    p.add_argument("--visual-ckpt", help="best.pt of the visual branch (Phase 2)")
    p.add_argument("--audio-ckpt", help="best.pt of the audio branch (Phase 3)")
    p.add_argument("--modality-dropout", type=float, default=0.2,
                   help="probability of hiding one modality per training sample (intermediate)")
    p.add_argument("--frames-per-video", type=int, default=8)
    return ["fusion", "visual_ckpt", "audio_ckpt", "modality_dropout", "frames_per_video"]


def _branch_config(path: str | None, default_backbone: str) -> tuple[str, dict | None]:
    """Backbone name + state dict from a branch checkpoint (or the config default)."""
    if not path:
        return default_backbone, None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["run_config"]["backbone"], ckpt["model"]


def load_branches(model, visual_state, audio_state) -> None:
    if visual_state is not None:
        model.visual.load_state_dict(visual_state)
        print("Visual branch initialised from checkpoint")
    if audio_state is not None:
        model.audio.load_state_dict(audio_state)
        print("Audio branch initialised from checkpoint")


@torch.no_grad()
def branch_probabilities(model, loader, device) -> dict:
    model.eval()
    out = {k: [] for k in ("pv", "pa", "hv", "ha", "label")}
    for batch in tqdm(loader, desc="branch scores", leave=False):
        b = to_device(batch, device)
        r = model(**{k: b[k] for k in FUSION_INPUTS})
        out["pv"].append(torch.sigmoid(r["visual_logit"]).cpu())
        out["pa"].append(torch.sigmoid(r["audio_logit"]).cpu())
        out["hv"].append(b["has_video"].float().cpu())
        out["ha"].append(b["has_audio"].float().cpu())
        out["label"].append(b["label"].cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def calibrate_late(probs: dict) -> pd.DataFrame:
    """Validation AUC for every late-fusion rule (and weight, for 'weighted')."""
    rows = []
    for rule in LATE_RULES:
        weights = np.round(np.linspace(0, 1, 21), 2) if rule == "weighted" else [0.5]
        for w in weights:
            p = late_combine(probs["pv"], probs["pa"], probs["hv"], probs["ha"], rule, float(w))
            m = compute_metrics(probs["label"].numpy(), p.numpy())
            rows.append({"rule": rule, "weight": float(w), "val_auc": m["auc"], "val_eer": m["eer"],
                         "val_acc": m["accuracy"]})
    return pd.DataFrame(rows).sort_values("val_auc", ascending=False, kind="stable")


def run_late(args, cfg, run_config_extra, visual_state, audio_state) -> Path:
    if visual_state is None or audio_state is None:
        raise SystemExit("Late fusion needs both --visual-ckpt and --audio-ckpt")
    seed_everything(args.seed)
    task = TASKS["fusion"]
    index = Path(args.index) if args.index else get_path(cfg, "processed", args.dataset, "index.csv")
    run_dir = get_path(cfg, "checkpoints", args.run_name or f"fusion_late_{args.dataset}")
    run_config = {"task": "fusion", "index": str(index), "dropout": args.dropout, "augment": False,
                  "seed": args.seed, **run_config_extra}

    device = get_device(args.device)
    model = task.build_model(run_config, False)
    load_branches(model, visual_state, audio_state)
    model.to(device)
    val_ds = task.build_dataset(index, "val", cfg, False, run_config)
    if len(val_ds) == 0:
        raise SystemExit(f"Empty val split in {index}")
    probs = branch_probabilities(model, DataLoader(val_ds, args.batch_size, num_workers=args.workers), device)

    table = calibrate_late(probs)
    best = table.iloc[0]
    run_config.update(late_rule=best.rule, late_weight=float(best.weight))
    model.late_rule, model.late_weight = best.rule, float(best.weight)
    print(table.head(8).round(4).to_string(index=False))
    print(f"Chosen late fusion: rule={best.rule} weight={best.weight} (val AUC {best.val_auc:.4f})")

    run_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(run_dir / "calibration.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str))
    state = TrainState(epoch=0, best_score=float(best.val_auc), best_epoch=0)
    save_checkpoint(run_dir / "best.pt", model, None, None, None, state, run_config)
    evaluate_checkpoint(run_dir / "best.pt", index, "test", cfg, run_dir / "test",
                        device=device, workers=args.workers)
    return run_dir


def main(argv=None) -> Path:
    cfg = load_config()
    p = build_parser("fusion", cfg, __doc__)
    keys = _extra(p)
    args = p.parse_args(argv)
    extra = {k: getattr(args, k) for k in keys}

    v_backbone, v_state = _branch_config(args.visual_ckpt, cfg["model"]["visual_backbone"])
    a_backbone, a_state = _branch_config(args.audio_ckpt, cfg["model"]["audio_backbone"])
    extra.update(visual_backbone=v_backbone, audio_backbone=a_backbone)

    if args.fusion == "late":
        return run_late(args, cfg, extra, v_state, a_state)
    if v_state is None or a_state is None:
        print("WARNING: branch checkpoint(s) missing; those branches start from ImageNet weights")
    if not args.run_name:
        args.run_name = f"fusion_intermediate_{args.dataset}"
    return run("fusion", args, cfg, extra, init_fn=lambda m: load_branches(m, v_state, a_state))


if __name__ == "__main__":
    main()
