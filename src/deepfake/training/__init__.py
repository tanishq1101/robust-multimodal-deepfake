"""Training.

  engine.py        shared loop: device, AMP, checkpoint/resume, early stopping, history
  tasks.py         per-branch registry (model, dataset, forward fn, label key, CLI defaults)
  runner.py        generic training CLI: data, optimizer, freeze schedule, fit, test eval
  train_visual.py  Phase 2: EfficientNet-B0 on face crops          (done)
  train_audio.py   Phase 3: ResNet-18 on log-mel spectrograms      (done)
  train_fusion.py  Phase 4: late / intermediate fusion             (done)

Checkpoints go to <data_root>/checkpoints/<run-name>/ every epoch so Colab disconnects can resume.
"""
