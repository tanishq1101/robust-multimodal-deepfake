"""Build the labelled demo clips in demo_samples/ from their public sources.

The output is committed to the repo, so this script only needs re-running to rebuild or audit it.
Sources (details and licences in demo_samples/README.md):
  DFDC preview clips      lynaeydofd.mp4 (real) and mqzvfufzoq.mp4 (face swap), consenting paid
                          actors, as shipped in github.com/polimi-ispl/icpr2020dfdc
  VITS paper samples      the same LJ Speech sentences as the original recording and as
                          Tacotron 2 / Glow-TTS / VITS speech, github.com/jaywalnut310/vits-demo

Two of the fake videos are made here: the real or face-swapped DFDC clip with its soundtrack
replaced by synthetic speech. They stand in for FakeAVCeleb's RealVideo-FakeAudio and
FakeVideo-FakeAudio categories (lips and voice no longer match, which is the point).

Usage:  python scripts/make_demo_samples.py [--out demo_samples]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".cache" / "deepfake" / "demo_sources"
DFDC = "https://raw.githubusercontent.com/polimi-ispl/icpr2020dfdc/master/notebook/samples/"
VITS = "https://raw.githubusercontent.com/jaywalnut310/vits-demo/master/wavs/"
DFDC_CREDIT = "DFDC preview set (Dolhansky et al., 2019), via polimi-ispl/icpr2020dfdc"
VITS_CREDIT = "VITS paper samples (Kim et al., 2021), github.com/jaywalnut310/vits-demo"
LJ_TEXT = ("that not more than one bottle of wine or one quart of beer could be issued at one time. "
           "No account was taken of the amount of liquors admitted in one day,")

# name -> how to build it, its ground truth and where it comes from
SAMPLES = {
    "video": [
        {"file": "real_actor.mp4", "label": "real", "category": "RealVideo-RealAudio",
         "build": ("video", "lynaeydofd.mp4", None),
         "description": "Unedited DFDC recording of a consenting actor.", "source": DFDC_CREDIT},
        {"file": "fake_faceswap.mp4", "label": "fake", "category": "FakeVideo-RealAudio",
         "build": ("video", "mqzvfufzoq.mp4", None),
         "description": "DFDC face swap: another actor's face on this body, original voice.", "source": DFDC_CREDIT},
        {"file": "fake_cloned_voice.mp4", "label": "fake", "category": "RealVideo-FakeAudio",
         "build": ("video", "lynaeydofd.mp4", "ss_01_vits.wav"),
         "description": "The real clip above with its soundtrack replaced by VITS text-to-speech.",
         "source": f"{DFDC_CREDIT}; voice: {VITS_CREDIT}"},
        {"file": "fake_faceswap_and_voice.mp4", "label": "fake", "category": "FakeVideo-FakeAudio",
         "build": ("video", "mqzvfufzoq.mp4", "ss_03_gh.wav"),
         "description": "The face swap above with its soundtrack replaced by Glow-TTS + HiFi-GAN speech.",
         "source": f"{DFDC_CREDIT}; voice: {VITS_CREDIT}"},
    ],
    "audio": [
        {"file": "real_ljspeech.wav", "label": "real", "category": "bona fide",
         "build": ("audio", "ss_01_gt.wav"),
         "description": "Original LJ Speech recording (public domain audiobook reading).", "source": VITS_CREDIT,
         "text": LJ_TEXT},
        {"file": "fake_vits.wav", "label": "fake", "category": "VITS",
         "build": ("audio", "ss_01_vits.wav"),
         "description": "The same sentence spoken by VITS, an end-to-end TTS model trained on this voice.",
         "source": VITS_CREDIT, "text": LJ_TEXT},
        {"file": "fake_tacotron2.wav", "label": "fake", "category": "Tacotron 2 + HiFi-GAN",
         "build": ("audio", "ss_01_th.wav"),
         "description": "The same sentence from Tacotron 2 with a HiFi-GAN vocoder.", "source": VITS_CREDIT,
         "text": LJ_TEXT},
        {"file": "fake_glowtts.wav", "label": "fake", "category": "Glow-TTS + HiFi-GAN",
         "build": ("audio", "ss_01_gh.wav"),
         "description": "The same sentence from Glow-TTS with a HiFi-GAN vocoder.", "source": VITS_CREDIT,
         "text": LJ_TEXT},
    ],
    "image": [
        {"file": "real_face.jpg", "label": "real", "category": "real photo",
         "build": ("image", "lynaeydofd.mp4", 3.0),
         "description": "Frame of the unedited DFDC recording.", "source": DFDC_CREDIT},
        {"file": "fake_faceswap_front.jpg", "label": "fake", "category": "face swap",
         "build": ("image", "mqzvfufzoq.mp4", 2.0),
         "description": "Frame of the DFDC face swap, looking at the camera.", "source": DFDC_CREDIT},
        {"file": "fake_faceswap_gesture.jpg", "label": "fake", "category": "face swap",
         "build": ("image", "mqzvfufzoq.mp4", 7.0),
         "description": "Another frame of the same face swap, mid-gesture.", "source": DFDC_CREDIT},
    ],
}


def fetch(base: str, name: str) -> Path:
    dest = CACHE / name
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(base + name, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


def ffmpeg(*args) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def build(spec: dict, out: Path) -> None:
    kind, src, *rest = spec["build"]
    if kind == "video":
        video, voice = fetch(DFDC, src), rest[0]
        # 720p keeps faces large enough for the detector at a fraction of the 1080p size
        common = ["-vf", "scale=-2:720", "-c:v", "libx264", "-crf", "24", "-preset", "slow",
                  "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if voice is None:
            ffmpeg("-i", video, *common, "-c:a", "aac", "-b:a", "96k", out)
        else:  # replace the soundtrack, padded with silence to the video length
            ffmpeg("-i", video, "-i", fetch(VITS, voice), "-map", "0:v", "-map", "1:a", *common,
                   "-af", "apad", "-shortest", "-c:a", "aac", "-b:a", "96k", out)
    elif kind == "audio":
        ffmpeg("-i", fetch(VITS, src), "-ac", "1", "-c:a", "pcm_s16le", out)
    else:
        ffmpeg("-ss", rest[0], "-i", fetch(DFDC, src), "-frames:v", "1", "-vf", "scale=-2:720",
               "-q:v", "3", out)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "demo_samples"))
    args = p.parse_args(argv)
    out = Path(args.out)
    manifest = []
    for kind, specs in SAMPLES.items():
        (out / kind).mkdir(parents=True, exist_ok=True)
        for spec in specs:
            build(spec, out / kind / spec["file"])
            manifest.append({"kind": kind, "path": f"{kind}/{spec['file']}",
                             **{k: v for k, v in spec.items() if k not in ("build", "file")}})
            print("built", f"{kind}/{spec['file']}")
    (out / "samples.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", out / "samples.json")


if __name__ == "__main__":
    main()
