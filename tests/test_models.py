import pytest
import torch

from deepfake.models import AudioModel, VisualModel, build_model


@pytest.fixture(scope="module")
def batch():
    B, T = 2, 4
    return {
        "frames": torch.randn(B, T, 3, 224, 224),
        "frame_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
        "mel": torch.randn(B, 1, 128, 401),
        "has_video": torch.tensor([True, True]),
        "has_audio": torch.tensor([True, False]),
    }


def test_visual_model(batch):
    m = VisualModel(pretrained=False).eval()
    out = m(batch["frames"], batch["frame_mask"])
    assert out["logit"].shape == (2,)
    assert out["embedding"].shape == (2, 1280)
    assert out["frame_logits"].shape == (2, 4)


def test_audio_model(batch):
    m = AudioModel(pretrained=False).eval()
    out = m(batch["mel"])
    assert out["logit"].shape == (2,) and out["embedding"].shape == (2, 512)


@pytest.mark.parametrize("fusion", ["late", "intermediate"])
def test_fusion(cfg, batch, fusion):
    cfg = {**cfg, "model": {**cfg["model"], "fusion": fusion}}
    m = build_model(cfg, pretrained=False).eval()
    with torch.no_grad():
        out = m(**batch)
    assert out["logit"].shape == (2,)
    assert torch.isfinite(out["logit"]).all()
    if fusion == "late":
        # sample 2 has no audio -> fused prediction equals the visual prediction
        assert torch.allclose(torch.sigmoid(out["logit"][1]), torch.sigmoid(out["visual_logit"][1]), atol=1e-4)


def test_fusion_backward(cfg, batch):
    m = build_model(cfg, pretrained=False).train()
    out = m(**batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logit"], torch.tensor([0.0, 1.0]))
    loss.backward()
    assert m.mlp[0].weight.grad is not None
