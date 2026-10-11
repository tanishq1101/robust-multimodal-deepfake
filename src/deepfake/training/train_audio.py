"""Phase 3: train the audio branch (ResNet-18 on log-mel spectrograms).

Trains on `audio_label` (bona fide vs. synthetic/converted speech). Training uses a random
4 s window + SpecAugment; evaluation uses the centre 4 s window.

Typical two-step recipe:
  # 1. pre-train on ASVspoof 2019 LA (large, speech-only)
  python -m deepfake.training.train_audio --dataset asvspoof --epochs 15
  # 2. fine-tune on FakeAVCeleb audio
  python -m deepfake.training.train_audio --dataset fakeavceleb --lr 3e-5 --freeze-epochs 0 \\
      --init-from <data_root>/checkpoints/audio_asvspoof/best.pt

Outputs in <data_root>/checkpoints/<run-name>/ (see deepfake.training.runner); for ASVspoof
test/per_category.csv holds the EER for every attack system (A07–A19).
"""

from __future__ import annotations

from deepfake.training.runner import main_for


def main(argv=None):
    return main_for("audio", __doc__, argv)


if __name__ == "__main__":
    main()
