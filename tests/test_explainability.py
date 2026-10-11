"""Phase 6: Grad-CAM correctness, shapes for every task, rendering and the gallery."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from deepfake.explainability import explain, overlay, render_explanation
from deepfake.explainability.gallery import build_gallery, load_model, pick_examples
from deepfake.training import train_fusion
from deepfake.training.tasks import TASKS

from conftest import COMMON, DEVICE


class ToyBackbone(nn.Module):
    """'layer4' copies the input; the score is the mean of the map -> evidence = where input is high."""

    def __init__(self):
        super().__init__()
        self.layer4 = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        nn.init.ones_(self.layer4.weight)

    def forward(self, x):
        return self.layer4(x).mean(dim=(1, 2, 3))


class ToyAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ToyBackbone()

    def forward(self, mel):
        return {"logit": self.backbone(mel) * 10 - 1}


def _toy_batch():
    mel = torch.zeros(1, 1, 128, 401)
    mel[..., :32, :100] = 1.0  # the only region that can raise the score
    return {"mel": mel, "frames": torch.zeros(1, 1, 3, 8, 8)}


def test_gradcam_highlights_the_evidence():
    exp = explain(ToyAudio(), "audio", _toy_batch(), target="fake")
    cam = exp["audio_cam"]
    assert cam.shape == (128, 401) and cam.max() == pytest.approx(1.0)
    assert cam[:32, :100].mean() > 0.99      # evidence region lit up
    assert cam[64:, 200:].max() == 0.0        # nothing elsewhere
    # explaining the opposite class: the region only argues for "fake" -> no evidence for "real"
    assert explain(ToyAudio(), "audio", _toy_batch(), target="real")["audio_cam"].max() == 0.0


def _batch(ds, i, device):
    return {k: v[None].to(device) for k, v in ds[i].items()}


@pytest.mark.parametrize("task,ckpt_idx", [("visual", 1), ("audio", 2)])
def test_explain_single_branch(branches, tiny_index, cfg, task, ckpt_idx):
    model, rc = load_model(branches[ckpt_idx], DEVICE)
    ds = TASKS[task].build_dataset(tiny_index, "test", cfg, False, rc)
    exp = explain(model, task, _batch(ds, 0, DEVICE))
    assert 0 <= exp["prob"] <= 1 and exp["target"] in ("fake", "real")
    if task == "visual":
        cam = exp["visual_cam"]
        assert cam.shape == (cfg["video"]["num_frames"], 224, 224) and exp["audio_cam"] is None
        valid = ds[0]["frame_mask"].numpy()
        assert cam[~valid].max(initial=0) == 0      # padded frames have no heatmap
    else:
        assert exp["audio_cam"].shape == (128, 401) and exp["visual_cam"] is None
    for cam in (exp["visual_cam"], exp["audio_cam"]):
        if cam is not None:
            assert cam.min() >= 0 and cam.max() <= 1 + 1e-6
    # no gradients left behind on the model
    assert all(p.grad is None for p in model.parameters())


def test_explain_fusion_and_missing_modality(branches, tiny_index, cfg, monkeypatch):
    root, v, a = branches
    monkeypatch.setenv("DEEPFAKE_DATA_ROOT", str(root))
    run = train_fusion.main(["--index", str(tiny_index), "--run-name", "fx", "--epochs", "1",
                             "--visual-ckpt", str(v), "--audio-ckpt", str(a), "--frames-per-video", "2",
                             "--freeze-epochs", "0", *COMMON])
    model, rc = load_model(run / "best.pt", DEVICE)
    ds = TASKS["fusion"].build_dataset(tiny_index, "test", cfg, False, rc)
    exp = explain(model, "fusion", _batch(ds, 0, DEVICE))
    assert exp["visual_cam"].shape[1:] == (224, 224) and exp["audio_cam"].shape == (128, 401)

    ds.use_audio = False  # missing voice -> no spectrogram explanation
    exp = explain(model, "fusion", _batch(ds, 0, DEVICE))
    assert exp["visual_cam"] is not None and exp["audio_cam"] is None

    fig = render_explanation(_batch(ds, 0, "cpu"), exp, title="test")
    assert len(fig.axes) >= 2


def test_overlay():
    img = np.zeros((32, 32, 3), np.uint8)
    cam = np.zeros((32, 32)); cam[:16] = 1
    out = overlay(img, cam)
    assert out.shape == img.shape and out.dtype == np.uint8
    # JET: hot = red, cold = blue
    assert out[:16, :, 0].mean() > out[:16, :, 2].mean()
    assert out[16:, :, 2].mean() > out[16:, :, 0].mean()


def test_pick_examples():
    scored = pd.DataFrame({"pos": range(6), "sample_id": list("abcdef"), "category": ["x"] * 3 + ["y"] * 3,
                           "label": [1, 1, 0, 0, 0, 1], "score": [0.9, 0.2, 0.1, 0.45, 0.8, 0.6]})
    picks = pick_examples(scored, 1).set_index("sample_id")
    assert picks.loc["a", "outcome"] == "correct" and picks.loc["b", "outcome"] == "wrong"
    assert picks.loc["e", "outcome"] == "wrong"  # most confident mistake in y


def test_gallery(branches, tiny_index, cfg, tmp_path):
    _, v, _ = branches
    table = build_gallery(v, tiny_index, "test", cfg, tmp_path / "gal", per_category=1, device=DEVICE)
    assert len(table) >= 4  # 4 categories, at least one example each
    assert all((tmp_path / "gal" / f).exists() for f in table.file)
    assert (tmp_path / "gal" / "index.md").exists() and (tmp_path / "gal" / "gallery.csv").exists()
