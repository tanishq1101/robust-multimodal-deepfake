import shutil
import subprocess

import pytest

from deepfake.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def video_dir(tmp_path_factory):
    """Folder with one 3-second synthetic clip: test pattern video + 440 Hz tone."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("videos")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "clip.mp4")],
        check=True,
    )
    return d
