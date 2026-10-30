"""Analyzer (video / audio / image), the pretrained face-model helpers, the CLI and the Gradio UI.

The DFDC face model is 267 MB, so these tests use a stub with the same interface; loading the real
checkpoint format is tested on a small file written the same way.
"""

import json
import subprocess

import cv2
import gradio as gr
import numpy as np
import pytest
import soundfile as sf
import torch

from deepfake import analyze, demo
from deepfake.analyzer import VOICE_CAVEAT, Analyzer, media_kind
from deepfake.face_model import confident_mean, crop_face, load_state_dict, to_canvas
from deepfake.inference import DeepfakeDetector

from conftest import DEVICE


class StubFace:
    """Same interface as FaceSwapDetector: every frame has a face scored `p`."""
    name = "stub face model"

    def __init__(self, p=0.9, face=True):
        self.p, self.face = p, face

    def faces(self, frames):
        if not self.face:
            return []
        return [{"frame": i, "box": (10, 10, 40, 40), "canvas": to_canvas(f[:60, :60], 64)}
                for i, f in enumerate(frames)]

    def score_canvases(self, canvases):
        return np.full(len(canvases), self.p)

    def gradcam(self, canvas, target="fake"):
        return np.ones(canvas.shape[:2], np.float32) * 0.5

    def score_image(self, rgb, explain=True, threshold=0.5):
        found = self.faces([rgb])
        if not found:
            return None
        out = {"p_fake": self.p, "box": found[0]["box"], "canvas": found[0]["canvas"]}
        if explain:
            out.update(overlay=found[0]["canvas"], explained_class="fake" if self.p >= threshold else "real")
        return out


@pytest.fixture(scope="module")
def media(video_dir, tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    t = np.arange(16000 * 9) / 16000
    sf.write(d / "speech.wav", (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000)
    cv2.imwrite(str(d / "photo.jpg"), np.full((120, 160, 3), 128, np.uint8))
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video_dir / "clip.mp4"), "-an", "-c:v", "copy",
                    str(d / "silent.mp4")], check=True)
    return {"video": video_dir / "clip.mp4", "silent": d / "silent.mp4", "audio": d / "speech.wav",
            "image": d / "photo.jpg"}


@pytest.fixture(scope="module")
def voice(branches):
    return DeepfakeDetector(branches[2], DEVICE)


def test_media_kind():
    assert [media_kind(f) for f in ("a.MP4", "b.wav", "c.jpeg", "d.webm", "e.png")] == \
        ["video", "audio", "image", "video", "image"]


def test_face_helpers():
    canvas = to_canvas(np.full((50, 100, 3), 255, np.uint8), 380)
    assert canvas.shape == (380, 380, 3) and canvas[0].max() == 0 and canvas[190].max() == 255
    crop = crop_face(np.zeros((300, 300, 3), np.uint8), (90, 90, 60, 90))
    assert crop.shape == (90 + 2 * 30, 60 + 2 * 20, 3)
    assert confident_mean([0.1, 0.2, 0.9]) == pytest.approx(0.4)
    assert confident_mean([0.95] * 16) == pytest.approx(0.95)         # mostly confident fakes
    assert confident_mean([0.05] * 15 + [0.6]) == pytest.approx(0.05)  # nearly all confident reals


def test_load_state_dict_is_weights_only(tmp_path):
    """The DFDC checkpoint stores numpy scalars next to the weights; they load without full unpickling."""
    path = tmp_path / "ckpt"
    torch.save({"state_dict": {"module.fc.weight": torch.ones(1, 2)}, "epoch": 3,
                "bce_best": np.float64(0.2)}, path)
    assert list(load_state_dict(path)) == ["module.fc.weight"]

    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))
    torch.save({"state_dict": {}, "x": Evil()}, tmp_path / "evil")
    with pytest.raises(Exception):
        load_state_dict(tmp_path / "evil")


def test_video_face_and_voice(voice, media):
    a = Analyzer(voice, StubFace(0.9))
    res = a.analyse(media["video"])
    assert res["kind"] == "video" and res["decided_by"] == "face + voice (max rule)"
    assert res["face"]["faces_found"] == res["face"]["frames_sampled"] > 0
    assert res["p_fake"] == pytest.approx(max(res["face"]["p_fake"], res["voice"]["p_fake"]))
    assert len(res["face"]["overlays"]) == 4 and res["timeline"].ndim == 3
    assert res["voice"]["spectrogram"].ndim == 3 and res["info"]["resolution"] == "320×240"
    # the video rule picks the branch; the threshold moves only the verdict
    assert a.analyse(media["video"], video_rule="face", with_explanation=False)["p_fake"] == pytest.approx(0.9)
    low = a.analyse(media["video"], threshold=0.05, with_explanation=False)
    high = a.analyse(media["video"], threshold=0.95, with_explanation=False)
    assert (low["verdict"], high["verdict"]) == ("FAKE", "REAL") and low["p_fake"] == high["p_fake"]
    with pytest.raises(ValueError):
        a.analyse(media["video"], video_rule="loudest")


def test_voice_caveat_only_when_voice_alone_says_fake(voice, media):
    res = Analyzer(voice, StubFace(0.0)).analyse(media["video"], threshold=0.01, with_explanation=False)
    assert res["verdict"] == "FAKE" and VOICE_CAVEAT in res["warnings"]
    res = Analyzer(voice, StubFace(0.99)).analyse(media["video"], threshold=0.01, with_explanation=False)
    assert VOICE_CAVEAT not in res["warnings"]


def test_audio_windows(voice, media):
    res = Analyzer(voice, None).analyse(media["audio"])
    w = res["voice"]["windows"]
    assert res["kind"] == "audio" and res["face"] is None and res["decided_by"] == "voice"
    assert len(w) == 4 and w[0]["start"] == 0 and w[-1]["end"] == pytest.approx(9.0, abs=0.05)
    assert res["p_fake"] == pytest.approx(np.mean([x["p_fake"] for x in w]))


def test_inconclusive_cases(voice, media):
    res = Analyzer(voice, StubFace(face=False)).analyse(media["silent"])
    assert res["verdict"] == "INCONCLUSIVE" and res["p_fake"] is None and len(res["warnings"]) == 2
    res = Analyzer(voice, StubFace(face=False)).analyse(media["image"])
    assert res["verdict"] == "INCONCLUSIVE" and "No face found" in res["warnings"][0]
    res = Analyzer(voice, None).analyse(media["image"])
    assert res["verdict"] == "INCONCLUSIVE" and "No face model" in res["warnings"][0]
    res = Analyzer(None, StubFace(0.2)).analyse(media["audio"])
    assert res["verdict"] == "INCONCLUSIVE" and "No voice model" in res["warnings"][0]
    with pytest.raises(ValueError):
        Analyzer(None, None)


def test_image(media):
    res = Analyzer(None, StubFace(0.8)).analyse(media["image"])
    assert res["verdict"] == "FAKE" and res["face"]["annotated"].shape == (120, 160, 3)
    assert "timeline" not in res and res["voice"] is None


def test_project_fusion_checkpoint_decides(late_fusion_ckpt, media):
    a = Analyzer(DeepfakeDetector(late_fusion_ckpt, DEVICE, use_face_detector=False), None)
    assert a.face_source == a.voice_source == "project"
    res = a.analyse(media["video"])
    assert res["decided_by"] == "fusion model" and res["face"]["p_fake"] is not None


def test_cli_and_summary(voice, media, tmp_path, monkeypatch):
    monkeypatch.setattr(analyze.Analyzer, "from_defaults", classmethod(lambda cls, *a, **k: Analyzer(voice, StubFace())))
    res = analyze.main([str(media["video"]), "--out", str(tmp_path)])
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["verdict"] == res["verdict"] and "overlays" not in saved["face"]
    assert (tmp_path / "timeline.png").exists() and (tmp_path / "face_gradcam_1.png").exists()


def test_gradio_ui(voice, media, monkeypatch):
    a = Analyzer(voice, StubFace(0.9))
    assert isinstance(demo.create_demo(a), gr.Blocks)
    no_progress = lambda *args, **kw: None                       # noqa: E731
    out = demo.make_handler(a, "video", [])(str(media["video"]), 0.5, "max", progress=no_progress)
    verdict, evidence, timeline, faces, spec, raw, details = out
    assert raw["verdict"] in verdict and "Face" in evidence and "Voice" in evidence
    assert timeline["visible"] and faces["visible"] and spec["visible"] and "overlays" not in raw["face"]
    out = demo.make_handler(a, "image", [])(str(media["image"]), 0.5, progress=no_progress)
    assert len(out) == 6 and out[3]["visible"]                  # verdict, evidence, faces, annotated, raw, md
    out = demo.make_handler(a, "audio", [])(str(media["audio"]), 0.5, progress=no_progress)
    assert len(out) == 6 and out[3]["visible"]                  # verdict, evidence, timeline, spec, raw, md
    with pytest.raises(gr.Error):
        demo.make_handler(a, "video", [])(None, 0.5, progress=no_progress)
    assert "No analysis yet" in demo.verdict_html(None, "image")


def test_sample_truth_and_manifest():
    samples = demo.load_samples()
    kinds = {s["kind"] for s in samples}
    assert kinds == {"video", "audio", "image"}
    for kind in kinds:
        labels = [s["label"] for s in samples if s["kind"] == kind]
        assert labels.count("real") == 1 and labels.count("fake") >= 2
    s = samples[0]
    assert demo.sample_truth(s["abs"], samples)["path"] == s["path"]
