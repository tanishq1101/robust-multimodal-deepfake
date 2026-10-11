"""End-to-end smoke test of Phase 2: preprocess -> train visual branch -> resume -> evaluate."""

import json
import os
import shutil
import subprocess

import pandas as pd
import pytest
import torch

from deepfake.preprocessing import run as prep
from deepfake.training import train_visual
from deepfake.training.engine import load_checkpoint
from deepfake.training.tasks import TASKS

DEVICE = os.environ.get("DEEPFAKE_TEST_DEVICE", "cpu")


@pytest.fixture(scope="module")
def tiny_index(tmp_path_factory):
    """8 synthetic clips: 'real' = test pattern, 'fake' = colour bars; split 4/2/2."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    vids = tmp_path_factory.mktemp("tiny_videos")
    for i in range(8):
        src = "testsrc" if i % 2 == 0 else "smptebars"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"{src}=duration=1:size=160x120:rate=10",
                        "-pix_fmt", "yuv420p", str(vids / f"v{i}.mp4")], check=True)
    out = tmp_path_factory.mktemp("tiny_processed")
    prep.main(["--input", str(vids), "--out", str(out), "--workers", "1", "--no-detector"])
    idx = pd.read_csv(out / "index.csv", keep_default_na=False)
    n = idx.sample_id.str[1:].astype(int)
    idx["label"] = idx["video_label"] = n % 2
    idx["split"] = ["train" if k < 4 else "val" if k < 6 else "test" for k in n]
    idx.to_csv(out / "index.csv", index=False)
    return out / "index.csv"


def _train(index, epochs, extra=()):
    return train_visual.main([
        "--index", str(index), "--run-name", "smoke", "--epochs", str(epochs), "--batch-size", "2",
        "--frames-per-video", "2", "--no-pretrained", "--device", DEVICE, "--workers", "0",
        "--freeze-epochs", "1", "--patience", "10", *extra,
    ])


def test_train_resume_evaluate(tiny_index, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFAKE_DATA_ROOT", str(tmp_path))
    run_dir = _train(tiny_index, epochs=2)
    for f in ["config.json", "last.pt", "best.pt", "history.csv", "test/metrics.json",
              "test/predictions.csv"]:
        assert (run_dir / f).exists(), f
    assert len(pd.read_csv(run_dir / "history.csv")) == 2

    metrics = json.loads((run_dir / "test" / "metrics.json").read_text())
    assert metrics["n"] == 2 and 0.0 <= metrics["auc"] <= 1.0
    assert len(pd.read_csv(run_dir / "test" / "predictions.csv")) == 2

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
