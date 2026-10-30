"""Phase 7: single-file inference with project checkpoints, its CLI and checkpoint discovery."""

import json
import subprocess

import numpy as np
import pytest
import soundfile as sf

from deepfake import inference
from deepfake.inference import DeepfakeDetector

from conftest import DEVICE


@pytest.fixture(scope="module")
def silent_video(video_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("noaudio") / "silent.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video_dir / "clip.mp4"), "-an", "-c:v", "copy", str(out)],
                   check=True)
    return out


def test_fusion_inference(late_fusion_ckpt, video_dir):
    det = DeepfakeDetector(late_fusion_ckpt, DEVICE, use_face_detector=False)
    res = det.analyse(video_dir / "clip.mp4")
    assert res["task"] == "fusion" and res["verdict"] in ("FAKE", "REAL")
    assert 0 <= res["p_fake"] <= 1
    assert set(res["scores"]) == {"overall", "visual", "audio"}
    assert res["info"]["frames_sampled"] == 16 and res["info"]["has_audio"]
    assert len(res["face_overlays"]) == 6 and res["face_overlays"][0].shape == (224, 224, 3)
    assert res["spectrogram"].ndim == 3 and res["figure"].ndim == 3
    assert res["warnings"] == []


def test_no_face_and_no_audio_warnings(late_fusion_ckpt, video_dir, silent_video):
    det = DeepfakeDetector(late_fusion_ckpt, DEVICE)            # real detector: test pattern has no face
    res = det.analyse(video_dir / "clip.mp4", with_explanation=False)
    assert not res["info"]["has_face"] and res["scores"]["visual"] is None
    assert any("No face" in w for w in res["warnings"])

    det = DeepfakeDetector(late_fusion_ckpt, DEVICE, use_face_detector=False)
    res = det.analyse(silent_video)
    assert not res["info"]["has_audio"] and res["scores"]["audio"] is None
    assert "spectrogram" not in res and "face_overlays" in res
    assert any("No audio" in w for w in res["warnings"])


def test_audio_file_with_audio_model(branches, tmp_path):
    t = np.arange(16000 * 2) / 16000
    sf.write(tmp_path / "voice.wav", 0.3 * np.sin(2 * np.pi * 440 * t), 16000)
    res = DeepfakeDetector(branches[2], DEVICE).analyse(tmp_path / "voice.wav")
    assert res["task"] == "audio" and set(res["scores"]) == {"audio"}
    assert res["spectrogram"].ndim == 3 and "face_overlays" not in res


def test_cli(late_fusion_ckpt, video_dir, tmp_path):
    inference.main([str(video_dir / "clip.mp4"), "--checkpoint", str(late_fusion_ckpt), "--out", str(tmp_path),
                    "--device", DEVICE])
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["verdict"] in ("FAKE", "REAL") and (tmp_path / "explanation.png").exists()


def test_find_best_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFAKE_DATA_ROOT", str(tmp_path))
    data_ckpts, published = tmp_path / "checkpoints", tmp_path / "models"
    assert inference.find_best_checkpoint(roots=[data_ckpts, published]) is None
    (published / "audio_asvspoof_x").mkdir(parents=True)
    (published / "audio_asvspoof_x" / "best.pt").touch()
    # the published models/ folder is the fallback ...
    assert inference.find_best_checkpoint(roots=[data_ckpts, published]).parent.name == "audio_asvspoof_x"
    for run in ["visual_dfdc", "fusion_late_fakeavceleb", "fusion_intermediate_fakeavceleb"]:
        (data_ckpts / run).mkdir(parents=True)
        (data_ckpts / run / "best.pt").touch()
    # ... and a better model type always wins
    assert inference.find_best_checkpoint(roots=[data_ckpts, published]).parent.name == "fusion_intermediate_fakeavceleb"


def test_score_image_with_face_model(branches, late_fusion_ckpt):
    rgb = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    for ckpt in (branches[1], late_fusion_ckpt):          # visual and fusion checkpoints
        p = DeepfakeDetector(ckpt, DEVICE, use_face_detector=False).score_image(rgb)
        assert 0 <= p <= 1
    with pytest.raises(ValueError):
        DeepfakeDetector(branches[2], DEVICE).score_image(rgb)
