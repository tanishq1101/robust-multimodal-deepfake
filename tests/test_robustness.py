"""Phase 5: degradations, waveform storage, robust training and the stress-test suite."""

import cv2
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from deepfake.datasets import AudioDataset, VisualDataset
from deepfake.evaluation import robustness_eval
from deepfake.preprocessing.frames import sample_frames
from deepfake.robustness.degradations import (AUDIO_DEGRADATIONS, SEVERITIES, VIDEO_DEGRADATIONS,
                                              Degradation, RandomDegradation)
from deepfake.training import train_audio, train_fusion, train_visual

from conftest import COMMON, DEVICE


@pytest.fixture(scope="module")
def clip(video_dir):
    frames = sample_frames(video_dir / "clip.mp4", 8)
    return np.stack([cv2.resize(f, (224, 224)) for f in frames])


@pytest.fixture(scope="module")
def wav():
    t = np.arange(16000 * 2) / 16000
    chirp = np.sin(2 * np.pi * (300 + 1500 * t) * t)
    return (0.3 * chirp + 0.1 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)


@pytest.mark.parametrize("name", VIDEO_DEGRADATIONS)
def test_video_degradations_get_worse_with_severity(clip, name):
    errors = []
    for level in range(len(SEVERITIES[name])):
        torch.manual_seed(0)
        out = Degradation(name, level)(clip)
        assert out.shape == clip.shape and out.dtype == np.uint8
        errors.append(np.abs(out.astype(float) - clip.astype(float)).mean())
    assert errors[0] > 0
    assert errors == sorted(errors), f"{name}: error should grow with severity, got {errors}"


def test_audio_noise_hits_target_snr(wav):
    for level, snr in enumerate(SEVERITIES["audio_noise"]):
        torch.manual_seed(0)
        noisy = Degradation("audio_noise", level)(wav)
        measured = 10 * np.log10(np.mean(wav ** 2) / np.mean((noisy - wav) ** 2))
        assert abs(measured - snr) < 0.5


def test_mp3_round_trip(wav):
    errors = []
    for level in range(len(SEVERITIES["mp3"])):
        out = Degradation("mp3", level)(wav)
        assert out.shape == wav.shape and out.dtype == np.float32
        errors.append(np.abs(out - wav).mean())
    assert errors[0] > 0 and errors[-1] > errors[0]  # lower bitrate -> more distortion


def test_random_degradation(clip, wav):
    assert RandomDegradation("video", p=0.0)(clip) is clip
    assert not np.array_equal(RandomDegradation("video", p=1.0, names=("gaussian_noise",))(clip), clip)
    assert not np.array_equal(RandomDegradation("audio", p=1.0, names=("audio_noise",))(wav), wav)
    with pytest.raises(ValueError):
        Degradation("nope", 0)


def test_preprocessing_saves_waveform(tiny_index):
    idx = pd.read_csv(tiny_index, keep_default_na=False)
    assert (idx.wav_path != "").all()
    audio, sr = sf.read(tiny_index.parent / idx.wav_path.iloc[0])
    assert sr == 16000 and abs(len(audio) - 16000) < 1600


def test_dataset_degradation_hooks(tiny_index, cfg):
    clean = AudioDataset(tiny_index, "test", cfg)[0]["mel"]
    ds = AudioDataset(tiny_index, "test", cfg)
    ds.wav_transform = Degradation("audio_noise", 2)
    noisy = ds[0]["mel"]
    assert noisy.shape == clean.shape == (1, 128, 401)
    assert not torch.allclose(noisy, clean)

    vds = VisualDataset(tiny_index, "test", cfg)
    base = vds[0]["frames"]
    vds.clip_transform = Degradation("h264", 2)
    assert vds[0]["frames"].shape == base.shape
    assert not torch.equal(vds[0]["frames"], base)


def test_robust_training(tiny_index, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFAKE_DATA_ROOT", str(tmp_path))
    v = train_visual.main(["--index", str(tiny_index), "--run-name", "vr", "--epochs", "1",
                           "--frames-per-video", "2", "--robust-augment", "--robust-p", "1", *COMMON])
    a = train_audio.main(["--index", str(tiny_index), "--run-name", "ar", "--epochs", "1",
                          "--robust-augment", "--robust-p", "1", *COMMON])
    for run_dir in (v, a):
        cfg = pd.read_json(run_dir / "config.json", typ="series")
        assert cfg["robust_augment"] and (run_dir / "test" / "metrics.json").exists()


def test_conditions_for():
    conds = robustness_eval.conditions_for("visual", list(SEVERITIES), [0, 1, 2], missing=True)
    assert conds[0]["condition"] == "clean"
    assert len(conds) == 1 + 3 * len(VIDEO_DEGRADATIONS)  # no audio conditions, no missing for single branch
    fusion = robustness_eval.conditions_for("fusion", list(SEVERITIES), [2], missing=True)
    assert len(fusion) == 1 + len(SEVERITIES) + 2
    assert {"video_only", "audio_only"} <= {c["condition"] for c in fusion}


def test_stress_suite(branches, tiny_index, monkeypatch):
    root, v_ckpt, a_ckpt = branches
    monkeypatch.setenv("DEEPFAKE_DATA_ROOT", str(root))
    late = train_fusion.main(["--index", str(tiny_index), "--run-name", "late_r", "--fusion", "late",
                              "--visual-ckpt", str(v_ckpt), "--audio-ckpt", str(a_ckpt), *COMMON])
    out = root / "robust_out"
    df = robustness_eval.main([
        "--index", str(tiny_index), "--out", str(out), "--device", DEVICE, "--label", "overall",
        "--degradations", "jpeg", "h264", "mp3", "audio_noise", "--levels", "0", "2",
        "--model", f"visual={v_ckpt}", "--model", f"audio={a_ckpt}", "--model", f"late={late / 'best.pt'}",
    ])
    counts = df.groupby("model").size().to_dict()
    assert counts == {"visual": 1 + 2 * 2, "audio": 1 + 2 * 2, "late": 1 + 4 * 2 + 2}
    assert (df.n == 4).all()
    assert (df[df.condition == "clean"].auc_drop == 0).all()
    for f in ("robustness.csv", "robustness.md", "robustness.png"):
        assert (out / f).exists()
    assert set(AUDIO_DEGRADATIONS) <= set(df.degradation)
