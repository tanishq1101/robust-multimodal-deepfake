"""Download the published trained weights (GitHub Release) into models/<run>/best.pt.

The run folders in models/ are in git (config, history, test metrics, figures); the weights are
too large for git (>100 MB), so they are attached to a GitHub Release and verified by SHA-256.

Usage:
  python scripts/download_model.py                 # every published model
  python scripts/download_model.py audio_asvspoof_3a53cf51
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

REPO = "tanishq1101/robust-multimodal-deepfake"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PUBLISHED = {
    "audio_asvspoof_3a53cf51": {
        "tag": "model-audio-asvspoof-v1",
        "asset": "audio_asvspoof_3a53cf51_best.pt",
        "sha256": "bdd5f5ec458936669161459fb9c9b084a069caaa1c226d8e5d31b688c7f4acc8",
        "size": 134_258_271,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str) -> Path:
    spec = PUBLISHED[name]
    dest = MODELS_DIR / name / "best.pt"
    if dest.is_file() and sha256(dest) == spec["sha256"]:
        print(f"Already present and verified: {dest}")
        return dest
    url = f"https://github.com/{REPO}/releases/download/{spec['tag']}/{spec['asset']}"
    tmp = dest.with_suffix(".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f, tqdm(total=spec["size"], unit="B", unit_scale=True, desc=name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    if sha256(tmp) != spec["sha256"]:
        tmp.unlink()
        raise RuntimeError(f"Checksum mismatch for {url}; the file was discarded.")
    tmp.replace(dest)
    print(f"Saved and verified: {dest}")
    return dest


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", help=f"default: all published models ({', '.join(PUBLISHED)})")
    args = p.parse_args(argv)
    unknown = sorted(set(args.models) - set(PUBLISHED))
    if unknown:
        p.error(f"unknown model(s): {', '.join(unknown)}; published: {', '.join(PUBLISHED)}")
    for name in args.models or PUBLISHED:
        download(name)


if __name__ == "__main__":
    sys.exit(main())
