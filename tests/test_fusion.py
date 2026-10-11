"""Phase 4: fusion models, late-fusion calibration and the comparison table."""

import json

import pandas as pd
import torch

from deepfake.evaluation import compare
from deepfake.models import AudioModel, MultimodalModel, VisualModel
from deepfake.models.fusion import LATE_RULES, late_combine
from deepfake.training import train_fusion

from conftest import COMMON, DEVICE


def test_late_combine_rules():
    pv, pa = torch.tensor([0.9, 0.1, 0.8]), torch.tensor([0.1, 0.9, 0.7])
    one = torch.ones(3)
    assert torch.allclose(late_combine(pv, pa, one, one, "max"), torch.tensor([0.9, 0.9, 0.8]))
    assert torch.allclose(late_combine(pv, pa, one, one, "noisy_or"), torch.tensor([0.91, 0.91, 0.94]))
    assert torch.allclose(late_combine(pv, pa, one, one, "weighted", 1.0), pv)
    # missing audio -> only the visual probability counts; nothing present -> 0.5
    assert torch.allclose(late_combine(pv, pa, one, torch.zeros(3), "weighted", 0.3), pv)
    assert torch.allclose(late_combine(pv, pa, torch.zeros(3), torch.zeros(3), "max"), torch.full((3,), 0.5))


def test_modality_dropout_hides_exactly_one():
    m = MultimodalModel(VisualModel(pretrained=False), AudioModel(pretrained=False), modality_dropout=1.0)
    hv = torch.tensor([True] * 200 + [True, False])
    ha = torch.tensor([True] * 200 + [False, True])
    v, a = m._drop_modality(hv, ha)
    assert ((v ^ a)[:200]).all()            # both present -> exactly one hidden
    assert v[200] and not a[200]            # single modality is never removed
    assert a[201] and not v[201]
    assert 50 < int(v[:200].sum()) < 150    # roughly half hide the video


def test_intermediate_fusion(branches, tiny_index, monkeypatch):
    root, v_ckpt, a_ckpt = branches
    monkeypatch.setenv("DEEPFAKE_DATA_ROOT", str(root))
    run_dir = train_fusion.main(["--index", str(tiny_index), "--run-name", "fi", "--epochs", "1",
                                 "--freeze-epochs", "1", "--visual-ckpt", str(v_ckpt),
                                 "--audio-ckpt", str(a_ckpt), "--frames-per-video", "2", *COMMON])
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["task"] == "fusion" and cfg["fusion"] == "intermediate"
    assert (run_dir / "test" / "metrics.json").exists()
    # epoch 1 is frozen: branch weights must equal the loaded branch checkpoints
    fused = torch.load(run_dir / "last.pt", weights_only=False)["model"]
    vis = torch.load(v_ckpt, weights_only=False)["model"]
    aud = torch.load(a_ckpt, weights_only=False)["model"]
    assert torch.equal(fused["visual.head.1.weight"], vis["head.1.weight"])
    assert torch.equal(fused["audio.head.1.weight"], aud["head.1.weight"])
    for key, value in vis.items():
        assert torch.equal(fused[f"visual.{key}"], value), key
    for key, value in aud.items():
        assert torch.equal(fused[f"audio.{key}"], value), key


def test_late_fusion_and_comparison(branches, tiny_index, monkeypatch):
    root, v_ckpt, a_ckpt = branches
    monkeypatch.setenv("DEEPFAKE_DATA_ROOT", str(root))
    late_dir = train_fusion.main(["--index", str(tiny_index), "--run-name", "fl", "--fusion", "late",
                                  "--visual-ckpt", str(v_ckpt), "--audio-ckpt", str(a_ckpt), *COMMON])
    calib = pd.read_csv(late_dir / "calibration.csv")
    assert len(calib) == 21 + len(LATE_RULES) - 1
    cfg = json.loads((late_dir / "config.json").read_text())
    assert cfg["late_rule"] in LATE_RULES and 0 <= cfg["late_weight"] <= 1
    assert (late_dir / "test" / "metrics.json").exists()

    out = root / "compare"
    compare.main(["--index", str(tiny_index), "--out", str(out), "--device", DEVICE,
                  "--model", f"visual_only={v_ckpt}", "--model", f"audio_only={a_ckpt}",
                  "--model", f"late={late_dir / 'best.pt'}"])
    table = pd.read_csv(out / "comparison.csv")
    assert list(table.model) == ["visual_only", "audio_only", "late"]
    assert (table.n == 4).all()  # every model scored on the same 4 test clips
    per_cat = pd.read_csv(out / "per_category.csv", index_col=0)
    assert len(per_cat) == 4 and list(per_cat.columns) == ["visual_only", "audio_only", "late"]
    assert (out / "comparison.md").exists() and (out / "comparison.png").exists()
    # the comparison scores against the overall label, not the branch's own label
    m = json.loads((out / "visual_only" / "metrics.json").read_text())
    assert m["label_key"] == "label"
