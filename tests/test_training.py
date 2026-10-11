"""End-to-end smoke tests: preprocess -> train (visual / audio) -> resume -> evaluate."""

import json
import os
import shutil

import numpy as np
import pandas as pd
import torch

from deepfake.evaluation.evaluate import per_category
from deepfake.training import train_audio, train_visual
from deepfake.training.engine import load_checkpoint
from deepfake.training.tasks import TASKS, SpecAugment

DEVICE = os.environ.get("DEEPFAKE_TEST_DEVICE", "cpu")


COMMON = ["--batch-size", "2", "--no-pretrained", "--device", DEVICE, "--workers", "0",
          "--freeze-epochs", "1", "--patience", "10"]


def _train(index, epochs, extra=()):
    return train_visual.main(["--index", str(index), "--run-name", "smoke", "--epochs", str(epochs),
                              "--frames-per-video", "2", *COMMON, *extra])


def _train_audio(index, run_name, epochs, extra=()):
    return train_audio.main(["--index", str(index), "--run-name", run_name, "--epochs", str(epochs),
                             *COMMON, *extra])


def test_train_resume_evaluate(tiny_index, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFAKE_DATA_ROOT", str(tmp_path))
    run_dir = _train(tiny_index, epochs=2)
    for f in ["config.json", "last.pt", "best.pt", "history.csv", "test/metrics.json",
              "test/predictions.csv"]:
        assert (run_dir / f).exists(), f
    assert len(pd.read_csv(run_dir / "history.csv")) == 2

    metrics = json.loads((run_dir / "test" / "metrics.json").read_text())
    assert metrics["n"] == 4 and 0.0 <= metrics["auc"] <= 1.0
    assert len(pd.read_csv(run_dir / "test" / "predictions.csv")) == 4

    # Resume: asking for 3 epochs continues from epoch 2 instead of starting over.
    _train(tiny_index, epochs=3)
    hist = pd.read_csv(run_dir / "history.csv")
    assert list(hist.epoch) == [1, 2, 3]

    model = TASKS["visual"].build_model(json.loads((run_dir / "config.json").read_text()), False)
    state, cfg = load_checkpoint(run_dir / "last.pt", model)
    assert state.epoch == 3 and cfg["task"] == "visual"


def test_frame_subsampling(tiny_index, cfg):
    rc = {"frames_per_video": 4, "augment": True}
    train = TASKS["visual"].build_dataset(tiny_index, "train", cfg, True, rc)
    val = TASKS["visual"].build_dataset(tiny_index, "val", cfg, False, rc)
    assert train[0]["frames"].shape[0] == 4
    assert val[0]["frames"].shape[0] == cfg["video"]["num_frames"]
    assert not torch.equal(train[0]["frames"], torch.zeros_like(train[0]["frames"]))


def test_audio_train_and_init_from(tiny_index, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFAKE_DATA_ROOT", str(tmp_path))
    run_dir = _train_audio(tiny_index, "audio_smoke", epochs=2)
    for f in ["best.pt", "history.csv", "curves.png", "test/metrics.json", "test/per_category.csv"]:
        assert (run_dir / f).exists(), f
    assert json.loads((run_dir / "config.json").read_text())["task"] == "audio"
    cats = pd.read_csv(run_dir / "test" / "per_category.csv")
    assert len(cats) == 4  # all FakeAVCeleb-style categories

    # Fine-tune from the pre-trained checkpoint; with lr=0 the head must keep its weights.
    ft_dir = _train_audio(tiny_index, "audio_ft", epochs=1,
                          extra=["--init-from", str(run_dir / "best.pt"), "--lr", "0"])
    init = torch.load(run_dir / "best.pt", weights_only=False)["model"]
    tuned = torch.load(ft_dir / "last.pt", weights_only=False)["model"]
    assert torch.equal(init["head.1.weight"], tuned["head.1.weight"])


def test_audio_dataset_drops_unlabelled(tiny_index, tmp_path, cfg):
    idx = pd.read_csv(tiny_index, keep_default_na=False)
    idx.loc[idx.split == "train", "audio_label"] = -1   # e.g. DFDC has no audio labels
    path = tmp_path / "index.csv"
    idx.to_csv(path, index=False)
    for d in ("faces", "mel"):
        shutil.copytree(tiny_index.parent / d, tmp_path / d)
    assert len(TASKS["audio"].build_dataset(path, "train", cfg, True, {})) == 0
    assert len(TASKS["audio"].build_dataset(path, "val", cfg, False, {})) == 4


def test_spec_augment_masks():
    mel = np.random.default_rng(0).normal(size=(128, 401)).astype(np.float32)
    out = SpecAugment(freq_masks=2, freq_width=16, time_masks=2, time_width=40)(mel)
    assert out.shape == mel.shape
    assert not np.array_equal(out, mel)
    assert np.array_equal(SpecAugment(0, 0, 0, 0)(mel), mel)


def test_per_category_breakdown():
    pred = pd.DataFrame({
        "category": ["bonafide"] * 4 + ["A07"] * 2 + ["A17"] * 2,
        "label": [0, 0, 0, 0, 1, 1, 1, 1],
        "score": [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.35, 0.05],
    })
    cats = per_category(pred).set_index("category")
    assert cats.loc["A07", "eer_vs_real"] == 0.0      # easy attack
    assert cats.loc["A17", "eer_vs_real"] > 0.3       # hard attack (AUC 3/8)
    assert cats.loc["A17", "auc_vs_real"] == 0.375
    assert cats.loc["bonafide", "accuracy"] == 1.0
    assert "eer_vs_real" not in cats.columns or np.isnan(cats.loc["bonafide", "eer_vs_real"])
