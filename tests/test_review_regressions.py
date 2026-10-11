"""Regression checks for training integrity and unsupported inference inputs."""

import json
import random
import copy
import tarfile

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from deepfake.data.download import _safe_target, extract
from deepfake.data.splits import assign_splits
from deepfake.datasets import MultimodalDataset
from deepfake.inference import DeepfakeDetector
from deepfake.models.fusion import late_combine
from deepfake.models import AudioModel, MultimodalModel, VisualModel
from deepfake.training.engine import fit, train_one_epoch
from deepfake.training.runner import validate_index, validate_labels


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_weighted_fusion_uses_only_available_branch(weight):
    p = late_combine(torch.tensor([0.8, 0.2]), torch.tensor([0.4, 0.9]),
                     torch.tensor([1., 0.]), torch.tensor([0., 1.]), weight=weight)
    assert torch.allclose(p, torch.tensor([0.8, 0.9]))


def test_partial_official_splits_keep_groups_together():
    df = pd.DataFrame({"group": ["a", "a", "b", "c", "d"],
                       "split": ["test", "", "", "", ""], "label": [0, 1, 0, 1, 0]})
    out = assign_splits(df, 0.2, 0.2, 0)
    assert out.loc[0, "split"] == "test" and out.loc[1, "split"] == "excluded"
    active = out[out.split != "excluded"]
    assert active.groupby("group").split.nunique().max() == 1
    pd.testing.assert_frame_equal(assign_splits(out, 0.2, 0.2, 0), out)


def test_zero_validation_ratio():
    df = pd.DataFrame({"group": list("abcd"), "split": ""})
    assert "val" not in set(assign_splits(df, 0, 0.25, 0).split)


@pytest.mark.parametrize("key", ["group", "path"])
def test_training_rejects_split_leakage(tmp_path, key):
    df = pd.DataFrame({"sample_id": ["a", "b"], "split": ["train", "test"],
                       "group": ["a", "b"], "path": ["a.mp4", "b.mp4"]})
    df[key] = "same"
    path = tmp_path / "index.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Split leakage"):
        validate_index(path)


def test_validation_requires_two_classes():
    class Dataset:
        df = pd.DataFrame({"label": [1, 1]})
        def __len__(self):
            return len(self.df)
    with pytest.raises(ValueError, match="both real and fake"):
        validate_labels(Dataset(), "label", "val")


def test_archive_rejects_sibling_prefix_and_links(tmp_path):
    with pytest.raises(RuntimeError, match="Unsafe path"):
        _safe_target(tmp_path / "data", "../data_escape/file")
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tf:
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../escape"
        tf.addfile(link)
    with pytest.raises(RuntimeError, match="link or special file"):
        extract(archive, tmp_path / "data")


def test_no_face_matches_inference(tiny_index, tmp_path, cfg):
    import shutil
    df = pd.read_csv(tiny_index, keep_default_na=False)
    df["face_detection"] = True  # stored synthetic frames have no face detections
    df.to_csv(tmp_path / "index.csv", index=False)
    shutil.copytree(tiny_index.parent / "faces", tmp_path / "faces")
    ds = MultimodalDataset(tmp_path / "index.csv", "train", cfg, use_audio=False)
    item = ds[0]
    assert not item["has_video"] and not item["frame_mask"].any()


@pytest.mark.parametrize("task", ["visual", "audio", "fusion"])
def test_missing_evidence_abstains(task, monkeypatch):
    det = DeepfakeDetector.__new__(DeepfakeDetector)
    det.task, det.threshold, det.checkpoint = task, 0.5, "test.pt"
    monkeypatch.setattr(det, "preprocess", lambda _: ({}, {"has_face": False, "has_audio": False}))
    monkeypatch.setattr(det, "_scores", lambda _: pytest.fail("Unsupported input must not be scored"))
    res = det.analyse("ignored")
    assert res["verdict"] == "INCONCLUSIVE" and res["p_fake"] is None
    assert all(v is None for v in res["scores"].values())


def test_frozen_batchnorm_and_dropout_do_not_change():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Dropout(0.9))
            self.head = nn.Linear(2, 1)
        def forward(self, x):
            return self.head(self.backbone(x)).squeeze(-1)
    model = Model()
    for p in model.backbone.parameters():
        p.requires_grad = False
    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    loader = DataLoader([{"x": torch.randn(2), "label": torch.tensor(float(i % 2))}
                         for i in range(4)], batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters())
    train_one_epoch(model, loader, lambda m, b: m(b["x"]), nn.BCEWithLogitsLoss(),
                    optimizer, torch.device("cpu"))
    assert not model.backbone.training and model.head.training
    assert all(torch.equal(v, before[k]) for k, v in model.backbone.state_dict().items())


def test_padding_cannot_change_visual_training_statistics():
    model = VisualModel(pretrained=False, dropout=0).train()
    other = copy.deepcopy(model)
    valid = torch.randn(2, 1, 3, 64, 64)
    padded = torch.cat([valid, torch.full_like(valid, 1000)], dim=1)
    a = model(valid)["logit"]
    b = other(padded, torch.tensor([[True, False], [True, False]]))["logit"]
    assert torch.allclose(a, b)
    for key, value in model.backbone.state_dict().items():
        assert torch.equal(value, other.backbone.state_dict()[key]), key


def test_missing_branch_does_not_update_batchnorm():
    model = MultimodalModel(VisualModel(pretrained=False), AudioModel(pretrained=False)).train()
    before = {k: v.clone() for k, v in model.audio.state_dict().items()}
    out = model(torch.randn(2, 1, 3, 64, 64), torch.ones(2, 1, dtype=torch.bool),
                torch.randn(2, 1, 128, 401), torch.ones(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool))
    out["logit"].sum().backward()
    assert all(torch.equal(v, before[k]) for k, v in model.audio.state_dict().items())


def test_resume_preserves_config_and_rng(tmp_path):
    loader = DataLoader([{"x": torch.tensor([float(i)]), "label": torch.tensor(float(i % 2))}
                         for i in range(4)], batch_size=2)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    rc = {"task": "toy", "lr": 0.01, "epochs": 1}
    args = (model, loader, loader, lambda m, b: m(b["x"]).squeeze(-1),
            nn.BCEWithLogitsLoss(), optimizer, scheduler, torch.device("cpu"))
    fit(*args, 1, tmp_path, rc)
    saved = torch.load(tmp_path / "last.pt", weights_only=False)
    assert "rng" in saved
    random.setstate(saved["rng"]["python"])
    expected = random.random()
    random.seed(99)
    fit(*args, 1, tmp_path, rc)
    assert random.random() == expected
    with pytest.raises(ValueError, match="Resume configuration changed"):
        fit(*args, 2, tmp_path, {**rc, "lr": 0.02, "epochs": 2})
    assert json.loads((tmp_path / "config.json").read_text()) == rc
    before = model.weight.detach().clone()
    fit(*args, 2, tmp_path, {**rc, "epochs": 2})
    assert scheduler.T_max == 2
    assert not torch.equal(model.weight, before)
