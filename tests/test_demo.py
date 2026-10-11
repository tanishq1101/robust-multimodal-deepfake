"""Phase 7: single-file inference, CLI, checkpoint discovery and the Gradio UI."""

import json
import subprocess

import gradio as gr
import numpy as np
import pytest
import soundfile as sf

from deepfake import inference
from deepfake.demo import create_demo, format_details, make_handler
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
    assert inference.find_best_checkpoint() is None
    for run in ["visual_dfdc", "fusion_late_fakeavceleb", "fusion_intermediate_fakeavceleb"]:
        (tmp_path / "checkpoints" / run).mkdir(parents=True)
        (tmp_path / "checkpoints" / run / "best.pt").touch()
    assert inference.find_best_checkpoint().parent.name == "fusion_intermediate_fakeavceleb"


def test_gradio_ui(late_fusion_ckpt, video_dir):
    det = DeepfakeDetector(late_fusion_ckpt, DEVICE, use_face_detector=False)
    assert isinstance(create_demo(det), gr.Blocks)
    handler = make_handler(det)
    label, details, faces, spec = handler(str(video_dir / "clip.mp4"), "predicted")
    assert set(label) == {"FAKE", "REAL"} and abs(sum(label.values()) - 1) < 1e-6
    assert "Verdict" in details and "Face model" in details and len(faces) == 6 and spec is not None
    with pytest.raises(gr.Error):
        handler(None, "predicted")


def test_format_details_single_branch():
    res = {"verdict": "REAL", "p_fake": 0.1, "threshold": 0.5, "task": "visual", "checkpoint": "/x/visual_dfdc/best.pt",
           "scores": {"visual": 0.1}, "warnings": ["careful"],
           "info": {"faces_detected": 3, "frames_sampled": 16, "has_audio": False, "audio_seconds": 0.0}}
    md = format_details(res)
    assert "Overall" not in md and "Face model" in md and "visual_dfdc" in md and "careful" in md
