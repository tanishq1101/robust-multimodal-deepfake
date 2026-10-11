"""Training loops.  (Phases 2-4)

Planned modules:
  train_visual.py   fine-tune VisualModel on VisualDataset (BCE, AdamW, early stopping on val AUC)
  train_audio.py    AudioModel: pre-train on ASVspoof 2019 LA, fine-tune on FakeAVCeleb
  train_fusion.py   MultimodalModel on FakeAVCeleb, branches initialised from the above

Checkpoints go to <data_root>/checkpoints/ every epoch so Colab disconnects can resume.
"""
