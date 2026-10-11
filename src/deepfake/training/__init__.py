"""Training.

  engine.py        shared loop: device, AMP, checkpoint/resume, early stopping, history
  tasks.py         per-branch registry (model, dataset, forward fn, label key)
  train_visual.py  Phase 2: EfficientNet-B0 on face crops          (done)
  train_audio.py   Phase 3: ResNet-18 on log-mel spectrograms      (todo)
  train_fusion.py  Phase 4: multimodal fusion                      (todo)

Checkpoints go to <data_root>/checkpoints/<run-name>/ every epoch so Colab disconnects can resume.
"""
