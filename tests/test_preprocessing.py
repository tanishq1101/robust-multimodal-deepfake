import json

import numpy as np
import pandas as pd
import pytest

from deepfake.data.manifest import build_asvspoof, build_dfdc, build_fakeavceleb
from deepfake.data.splits import assign_splits
from deepfake.datasets import AudioDataset, MultimodalDataset, VisualDataset
from deepfake.preprocessing import run as prep
from deepfake.preprocessing.audio import crop_or_pad, load_audio, log_mel
from deepfake.preprocessing.faces import FaceDetector, extract_faces, load_faces, save_faces
from deepfake.preprocessing.frames import sample_frames


def test_sample_frames(video_dir):
    frames = sample_frames(video_dir / "clip.mp4", 16)
    assert len(frames) == 16
    assert frames[0].shape == (240, 320, 3) and frames[0].dtype == np.uint8


def test_audio_and_mel(video_dir):
    wav = load_audio(video_dir / "clip.mp4", 16000)
    assert abs(len(wav) - 3 * 16000) < 1600
    mel = log_mel(wav)
    assert mel.shape[0] == 128 and abs(mel.shape[1] - 301) <= 10
    assert crop_or_pad(mel, 401).shape == (128, 401)
    assert crop_or_pad(mel, 100, random=True).shape == (128, 100)


def test_faces_roundtrip(tmp_path, video_dir):
    frames = sample_frames(video_dir / "clip.mp4", 4)
    faces, detected = extract_faces(frames, detector=None, size=224)
    assert faces.shape == (4, 224, 224, 3) and not detected.any()
    save_faces(tmp_path / "f.npz", faces, detected)
    back, det = load_faces(tmp_path / "f.npz")
    assert back.shape == faces.shape and (det == detected).all()
    assert np.abs(back.astype(int) - faces.astype(int)).mean() < 5  # JPEG q95


def test_face_detector_runs(video_dir):
    try:
        det = FaceDetector()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"detector unavailable: {e}")
    frame = sample_frames(video_dir / "clip.mp4", 1)[0]
    box = det.detect(frame)
    assert box is None or len(box) == 4


def test_end_to_end_folder(tmp_path, video_dir, cfg):
    out = tmp_path / "processed"
    prep.main(["--input", str(video_dir), "--out", str(out), "--workers", "1", "--no-detector"])
    index = pd.read_csv(out / "index.csv", keep_default_na=False)
    assert len(index) == 1 and index.error.iloc[0] == ""
    assert (out / index.faces_path.iloc[0]).exists() and (out / index.mel_path.iloc[0]).exists()

    item = MultimodalDataset(out / "index.csv", "test", cfg)[0]
    assert item["frames"].shape == (16, 3, 224, 224)
    assert item["frame_mask"].all()
    assert item["mel"].shape == (1, 128, 401)
    assert item["has_video"] and item["has_audio"]
    assert VisualDataset(out / "index.csv", "test", cfg)[0]["has_audio"].item() is False
    assert AudioDataset(out / "index.csv", "test", cfg)[0]["has_video"].item() is False


def test_manifest_fakeavceleb(tmp_path):
    raw = tmp_path / "raw" / "fakeavceleb" / "FakeAVCeleb_v1.2"
    for cat in ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]:
        for ident in ["id00001", "id00002"]:
            f = raw / cat / "African" / "men" / ident / "00001.mp4"
            f.parent.mkdir(parents=True)
            f.touch()
    df = build_fakeavceleb(tmp_path / "raw" / "fakeavceleb", tmp_path / "raw")
    assert len(df) == 8
    rvfa = df[df.sample_id.str.startswith("RealVideo-FakeAudio")].iloc[0]
    assert (rvfa.label, rvfa.video_label, rvfa.audio_label) == (1, 0, 1)
    assert set(df.group) == {"id00001", "id00002"}
    assert not df.path.str.startswith("/").any()


def test_manifest_dfdc(tmp_path):
    d = tmp_path / "raw" / "dfdc" / "train_sample_videos"
    d.mkdir(parents=True)
    meta = {"a.mp4": {"label": "REAL"}, "b.mp4": {"label": "FAKE", "original": "a.mp4"}}
    (d / "metadata.json").write_text(json.dumps(meta))
    for n in meta:
        (d / n).touch()
    df = build_dfdc(tmp_path / "raw" / "dfdc", tmp_path / "raw").set_index("sample_id")
    assert df.loc["b", "label"] == 1 and df.loc["a", "label"] == 0
    assert df.loc["a", "group"] == df.loc["b", "group"] == "a"


def test_manifest_asvspoof(tmp_path):
    d = tmp_path / "raw" / "asvspoof" / "LA"
    (d / "ASVspoof2019_LA_cm_protocols").mkdir(parents=True)
    (d / "ASVspoof2019_LA_train").mkdir()
    lines = {"train.trn": "LA_0079 LA_T_1 - - bonafide\nLA_0079 LA_T_2 - A01 spoof",
             "dev.trl": "LA_0070 LA_D_1 - - bonafide", "eval.trl": "LA_0001 LA_E_1 - A10 spoof"}
    for k, v in lines.items():
        (d / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{k}.txt").write_text(v)
    df = build_asvspoof(tmp_path / "raw" / "asvspoof", tmp_path / "raw")
    assert list(df.split) == ["train", "train", "val", "test"]
    assert list(df.label) == [0, 1, 0, 1]
    assert df.path.iloc[0] == "asvspoof/LA/ASVspoof2019_LA_train/flac/LA_T_1.flac"


def test_splits_keep_identities_together():
    df = pd.DataFrame({"group": [f"id{i // 4}" for i in range(400)], "label": [i % 2 for i in range(400)],
                       "split": ""})
    out = assign_splits(df, 0.15, 0.15, seed=0)
    assert out.groupby("group").split.nunique().max() == 1
    frac = out.split.value_counts(normalize=True)
    assert 0.6 < frac["train"] < 0.8 and frac["val"] > 0.1 and frac["test"] > 0.1


def test_splits_respect_official_test():
    df = pd.DataFrame({"group": ["a", "b", "c", "d", "e"], "label": [0, 1, 0, 1, 0],
                       "split": ["test", "", "", "", ""]})
    out = assign_splits(df, 0.25, 0.15, seed=0)
    assert (out.split == "test").sum() == 1
    assert set(out.split) <= {"train", "val", "test"}
