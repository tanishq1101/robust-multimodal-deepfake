"""Phase 2: train the visual branch (EfficientNet-B0 on face crops).

Trains on `video_label` with early stopping on validation AUC, checkpoints every epoch
(resumable), then evaluates the best checkpoint on the test split.

Usage:
  python -m deepfake.training.train_visual --dataset dfdc
  python -m deepfake.training.train_visual --dataset fakeavceleb --epochs 15 --batch-size 16 \\
      --init-from <data_root>/checkpoints/visual_dfdc/best.pt
  python -m deepfake.training.train_visual --index path/to/index.csv --run-name test --epochs 1

Outputs in <data_root>/checkpoints/<run-name>/ (see deepfake.training.runner).
"""

from __future__ import annotations

from deepfake.training.runner import main_for


def _extra(p):
    p.add_argument("--frames-per-video", type=int, default=8,
                   help="random frames per video while training (eval uses all 16)")
    return ["frames_per_video"]


def main(argv=None):
    return main_for("visual", __doc__, argv, _extra)


if __name__ == "__main__":
    main()
