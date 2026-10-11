import os
import shutil
import subprocess

import pandas as pd
import pytest

from deepfake.config import load_config

# category -> (video fake, audio fake), mirroring FakeAVCeleb
CATEGORIES = [
    ("RealVideo-RealAudio", 0, 0),
    ("FakeVideo-FakeAudio", 1, 1),
    ("RealVideo-FakeAudio", 0, 1),
    ("FakeVideo-RealAudio", 1, 0),
]


DEVICE = os.environ.get("DEEPFAKE_TEST_DEVICE", "cpu")
COMMON = ["--batch-size", "2", "--no-pretrained", "--device", DEVICE, "--workers", "0", "--patience", "10"]


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def _need_ffmpeg():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")


@pytest.fixture(scope="session")
def video_dir(tmp_path_factory):
    """Folder with one 3-second synthetic clip: test pattern video + 440 Hz tone."""
    _need_ffmpeg()
    d = tmp_path_factory.mktemp("videos")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "clip.mp4")],
        check=True,
    )
    return d


@pytest.fixture(scope="session")
def tiny_index(tmp_path_factory):
    """12 processed synthetic clips covering all 4 FakeAVCeleb-style categories, split 4/4/4
    (every split contains every category).
    real face = test pattern, fake face = colour bars; real voice = 440 Hz tone, fake = pink noise."""
    from deepfake.preprocessing import run as prep

    _need_ffmpeg()
    vids = tmp_path_factory.mktemp("tiny_videos")
    for i in range(12):
        _, vf, af = CATEGORIES[i % 4]
        video = "smptebars" if vf else "testsrc"
        audio = "anoisesrc=d=1:c=pink" if af else "sine=frequency=440:duration=1"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"{video}=duration=1:size=160x120:rate=10",
                        "-f", "lavfi", "-i", audio, "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                        str(vids / f"v{i}.mp4")], check=True)
    out = tmp_path_factory.mktemp("tiny_processed")
    prep.main(["--input", str(vids), "--out", str(out), "--workers", "1", "--no-detector"])
    idx = pd.read_csv(out / "index.csv", keep_default_na=False)
    n = idx.sample_id.str[1:].astype(int)
    idx["category"] = [CATEGORIES[k % 4][0] for k in n]
    idx["video_label"] = [CATEGORIES[k % 4][1] for k in n]
    idx["audio_label"] = [CATEGORIES[k % 4][2] for k in n]
    idx["label"] = idx.video_label | idx.audio_label
    idx["split"] = ["train" if k < 4 else "val" if k < 8 else "test" for k in n]
    idx.to_csv(out / "index.csv", index=False)
    return out / "index.csv"


@pytest.fixture(scope="session")
def branches(tiny_index, tmp_path_factory):
    """Data root with one-epoch visual and audio branch checkpoints: (root, visual.pt, audio.pt)."""
    from deepfake.training import train_audio, train_visual

    root = tmp_path_factory.mktemp("branch_root")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DEEPFAKE_DATA_ROOT", str(root))
        v = train_visual.main(["--index", str(tiny_index), "--run-name", "v", "--epochs", "1",
                               "--frames-per-video", "2", "--freeze-epochs", "0", *COMMON])
        a = train_audio.main(["--index", str(tiny_index), "--run-name", "a", "--epochs", "1",
                              "--freeze-epochs", "0", *COMMON])
    return root, v / "best.pt", a / "best.pt"
