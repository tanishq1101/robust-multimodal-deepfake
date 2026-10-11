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


def test_gradio_ui(late_fusion_ckpt, video_dir, tmp_path):
    det = DeepfakeDetector(late_fusion_ckpt, DEVICE, use_face_detector=False)
    assert isinstance(create_demo(det), gr.Blocks)
    handler = make_handler(det, "fusion model")
    verdict, evidence, spec, faces, empty, raw, summary = handler("video", str(video_dir / "clip.mp4"), None,
                                                                  0.5, "predicted")
    assert raw["verdict"] in ("FAKE", "REAL") and raw["verdict"] in verdict and "fusion model" in verdict
    assert "Face model" in evidence and "Voice model" in evidence and "not part of" not in evidence
    assert spec["visible"] and faces["visible"] and len(faces["value"]) == 6 and not empty["visible"]
    assert "Verdict" in summary and "face_overlays" not in raw
    # the threshold slider decides the verdict without re-running the model
    _, _, _, _, _, low, _ = handler("video", str(video_dir / "clip.mp4"), None, 0.01, "predicted")
    _, _, _, _, _, high, _ = handler("video", str(video_dir / "clip.mp4"), None, 0.99, "predicted")
    assert (low["verdict"], high["verdict"]) == ("FAKE", "REAL") and low["p_fake"] == high["p_fake"]
    with pytest.raises(gr.Error):
        handler("video", None, None, 0.5, "predicted")
    with pytest.raises(gr.Error):
        handler("audio", str(video_dir / "clip.mp4"), None, 0.5, "predicted")


def test_audio_tab_and_inconclusive(branches, late_fusion_ckpt, video_dir, silent_video, tmp_path):
    t = np.arange(16000 * 2) / 16000
    sf.write(tmp_path / "voice.wav", 0.3 * np.sin(2 * np.pi * 440 * t), 16000)
    audio_det = DeepfakeDetector(branches[2], DEVICE)
    verdict, evidence, spec, faces, *_ , raw, _ = make_handler(audio_det, "voice model")(
        "audio", None, str(tmp_path / "voice.wav"), 0.5, "predicted")
    assert raw["task"] == "audio" and "voice model" in verdict
    assert evidence.count("not part of the loaded model") == 2 and spec["visible"] and not faces["visible"]
    # an audio-only model cannot judge a silent video: INCONCLUSIVE, no probability
    verdict, *_ , raw, _ = make_handler(audio_det)("video", str(silent_video), None, 0.5, "predicted")
    assert raw["verdict"] == "INCONCLUSIVE" and raw["p_fake"] is None and "INCONCLUSIVE" in verdict


def test_verdict_html_empty_state():
    from deepfake.demo import verdict_html
    assert "No analysis yet" in verdict_html(None)


def test_format_details_single_branch():
    res = {"verdict": "REAL", "p_fake": 0.1, "threshold": 0.5, "task": "visual", "checkpoint": "/x/visual_dfdc/best.pt",
           "scores": {"visual": 0.1}, "warnings": ["careful"],
           "info": {"faces_detected": 3, "frames_sampled": 16, "has_audio": False, "audio_seconds": 0.0}}
    md = format_details(res)
    assert "Overall" not in md and "Face model" in md and "visual_dfdc" in md and "careful" in md
