"""Download, extract and verify the project datasets.

Datasets (saved under <data_root>/raw/<name>/):
  dfdc         DFDC train sample from Kaggle (~400 videos with audio).  Needs a Kaggle
               account, accepted competition rules, and an API token (kaggle.json).
  asvspoof     ASVspoof 2019 Logical Access (LA.zip, 7.6 GB). Free, no account.
  fakeavceleb  FakeAVCeleb v1.2. Access by request form; pass the link you receive
               with --url, or an already downloaded archive with --archive.
  celebdf      Celeb-DF v2. Access by request form; same --url / --archive options.

Usage:
  python -m deepfake.data.download status
  python -m deepfake.data.download asvspoof
  python -m deepfake.data.download asvspoof --check        # only test the link
  python -m deepfake.data.download dfdc
  python -m deepfake.data.download fakeavceleb --url "<link from the email>"
  python -m deepfake.data.download celebdf --archive /path/to/Celeb-DF-v2.zip
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from deepfake.config import get_path, load_config

ASVSPOOF_LA_URL = (
    "https://datashare.ed.ac.uk/server/api/core/bitstreams/"
    "a9f87c35-f055-4015-80e2-2fdff0d46269/content"
)
ASVSPOOF_LA_SIZE = 7_640_952_520
ASVSPOOF_LA_MIRROR = "https://zenodo.org/api/records/6906306/files/LA.zip/content"
ASVSPOOF_LA_MD5 = "30c98f11d8b2bc21f2c257bfd78bb5c5"
USER_AGENT = "Mozilla/5.0 (deepfake-research-download)"

FAKEAVCELEB_FORM = (
    "https://docs.google.com/forms/d/e/1FAIpQLSfPDd3oV0auqmmWEgCSaTEQ6CGpFeB-ozQJ35x-B_0Xjd93bw/viewform"
)
CELEBDF_REPO = "https://github.com/yuezunli/celeb-deepfakeforensics"
FAKEAVCELEB_CATEGORIES = (
    "RealVideo-RealAudio",
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
)
CELEBDF_FOLDERS = ("Celeb-real", "Celeb-synthesis", "YouTube-real")


# --------------------------------------------------------------------------- helpers

def http_download(url: str, dest: Path, expected_size: int | None = None) -> Path:
    """Stream a file to disk with resume support (safe to re-run after a Colab disconnect)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.stat().st_size if dest.exists() else 0
    if expected_size and done == expected_size:
        print(f"Already downloaded: {dest}")
        return dest
    headers = {"User-Agent": USER_AGENT}
    if done:
        headers["Range"] = f"bytes={done}-"
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        if done and r.status_code != 206:  # server ignored Range: start over
            done = 0
        total = done + int(r.headers.get("Content-Length", 0)) or expected_size
        with open(dest, "ab" if done else "wb") as f, tqdm(
            total=total, initial=done, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    if expected_size and dest.stat().st_size != expected_size:
        raise RuntimeError(
            f"{dest} is {dest.stat().st_size} bytes, expected {expected_size}. Re-run to resume."
        )
    return dest


def download_url(url: str, dest_dir: Path) -> Path:
    """Download a link from a dataset email: Google Drive links via gdown, others via HTTP."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if "drive.google.com" in url or "docs.google.com" in url:
        import gdown

        if "/folders/" in url:
            gdown.download_folder(url, output=str(dest_dir), quiet=False)
            return dest_dir
        out = gdown.download(url, output=str(dest_dir) + "/", quiet=False, fuzzy=True)
        return Path(out)
    name = url.split("?")[0].rstrip("/").split("/")[-1] or "download.zip"
    return http_download(url, dest_dir / name)


def _safe_target(dest: Path, member: str) -> Path:
    target = (dest / member).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise RuntimeError(f"Unsafe path in archive: {member}")
    return target


def extract(archive: Path, dest: Path, only_prefix: str | None = None) -> None:
    """Extract a zip/tar archive, optionally only members under `only_prefix`."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive.name} -> {dest}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if not only_prefix or m.startswith(only_prefix)]
            for m in tqdm(members, unit="file"):
                _safe_target(dest, m)
                zf.extract(m, dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            members = [
                m for m in tf.getmembers() if not only_prefix or m.name.startswith(only_prefix)
            ]
            for m in tqdm(members, unit="file"):
                _safe_target(dest, m.name)
                if not (m.isfile() or m.isdir()):
                    raise RuntimeError(f"Unsafe archive member (link or special file): {m.name}")
                tf.extract(m, dest)
    else:
        raise ValueError(f"Not a zip or tar archive: {archive}")


def find_dir_with(root: Path, names: tuple[str, ...], max_depth: int = 4) -> Path | None:
    """Find the first directory under root that contains all of `names`."""
    if not root.exists():
        return None
    queue = [(root, 0)]
    while queue:
        d, depth = queue.pop(0)
        if all((d / n).exists() for n in names):
            return d
        if depth < max_depth:
            queue += [(c, depth + 1) for c in sorted(d.iterdir()) if c.is_dir()]
    return None


def count(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.rglob(pattern))


def _ingest(raw: Path, url: str | None, archive: str | None) -> None:
    """Shared --url / --archive handling for the request-form datasets."""
    if url:
        got = download_url(url, raw / "_downloads")
        archives = [got] if got.is_file() else sorted(p for p in got.rglob("*") if p.is_file())
    else:
        archives = [Path(archive).expanduser()]
    for a in archives:
        if zipfile.is_zipfile(a) or tarfile.is_tarfile(a):
            extract(a, raw)


# --------------------------------------------------------------------------- datasets

def verify_dfdc(raw: Path) -> dict:
    d = find_dir_with(raw, ("metadata.json",))
    if d is None:
        return {"ok": False, "reason": "metadata.json not found"}
    return {"ok": True, "root": str(d), "videos": count(d, "*.mp4")}


def get_dfdc(raw: Path, archive: str | None = None) -> None:
    if archive:
        extract(Path(archive).expanduser(), raw, only_prefix="train_sample_videos/")
        return
    if shutil.which("kaggle") is None:
        sys.exit("kaggle CLI not found: pip install kaggle")
    if not (Path.home() / ".kaggle" / "kaggle.json").exists() and not _kaggle_env():
        sys.exit(
            "Kaggle API token missing. Kaggle -> Settings -> API -> 'Create New Token', then put\n"
            "kaggle.json in ~/.kaggle/ (Colab: see notebooks/01_download_datasets.ipynb).\n"
            "Also open https://www.kaggle.com/c/deepfake-detection-challenge/rules and accept them."
        )
    raw.mkdir(parents=True, exist_ok=True)
    # The default competition download (~4 GB) contains train_sample_videos/ and test_videos/.
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "deepfake-detection-challenge", "-p", str(raw)],
        check=True,
    )
    zip_path = raw / "deepfake-detection-challenge.zip"
    extract(zip_path, raw, only_prefix="train_sample_videos/")
    zip_path.unlink()


def _kaggle_env() -> bool:
    import os

    return bool(os.environ.get("KAGGLE_API_TOKEN")) or bool(
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )


def verify_asvspoof(raw: Path) -> dict:
    d = find_dir_with(raw, ("ASVspoof2019_LA_cm_protocols", "ASVspoof2019_LA_train"))
    if d is None:
        return {"ok": False, "reason": "LA/ folder with protocols not found"}
    return {
        "ok": True,
        "root": str(d),
        **{s: count(d / f"ASVspoof2019_LA_{s}", "*.flac") for s in ("train", "dev", "eval")},
    }


def get_asvspoof(raw: Path, check: bool = False, keep_zip: bool = False) -> None:
    if check:
        with requests.get(
            ASVSPOOF_LA_URL,
            headers={"User-Agent": USER_AGENT, "Range": "bytes=0-3"},
            timeout=60,
            stream=True,
        ) as r:
            header = next(r.iter_content(chunk_size=4), b"")
            ok = r.status_code in (200, 206) and header[:2] == b"PK"
            print(f"ASVspoof LA.zip link: HTTP {r.status_code}, zip header {'OK' if ok else 'MISSING'}")
        sys.exit(0 if ok else 1)
    try:
        zip_path = http_download(ASVSPOOF_LA_URL, raw / "LA.zip", expected_size=ASVSPOOF_LA_SIZE)
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 403:
            raise
        print("Edinburgh returned HTTP 403; using the ASVspoof Zenodo mirror.")
        zip_path = http_download(ASVSPOOF_LA_MIRROR, raw / "LA.zip", expected_size=ASVSPOOF_LA_SIZE)
    digest = hashlib.md5()
    with zip_path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != ASVSPOOF_LA_MD5:
        raise RuntimeError(f"ASVspoof archive checksum mismatch: {zip_path}; preserve it for inspection.")
    extract(zip_path, raw)
    if not keep_zip:
        zip_path.unlink()


def verify_fakeavceleb(raw: Path) -> dict:
    d = find_dir_with(raw, FAKEAVCELEB_CATEGORIES)
    if d is None:
        return {"ok": False, "reason": "category folders (RealVideo-RealAudio, ...) not found"}
    return {"ok": True, "root": str(d), **{c: count(d / c, "*.mp4") for c in FAKEAVCELEB_CATEGORIES}}


def get_fakeavceleb(raw: Path, url: str | None, archive: str | None) -> None:
    if not (url or archive):
        sys.exit(
            "FakeAVCeleb requires approval.\n"
            f"  1. Fill in the request form: {FAKEAVCELEB_FORM}\n"
            "  2. You'll receive a download link/script by email.\n"
            "  3. Re-run with --url '<link>' or --archive /path/to/FakeAVCeleb_v1.2.zip"
        )
    _ingest(raw, url, archive)


def verify_celebdf(raw: Path) -> dict:
    d = find_dir_with(raw, CELEBDF_FOLDERS)
    if d is None:
        return {"ok": False, "reason": "Celeb-real / Celeb-synthesis / YouTube-real not found"}
    return {
        "ok": True,
        "root": str(d),
        "test_list": (d / "List_of_testing_videos.txt").exists(),
        **{f: count(d / f, "*.mp4") for f in CELEBDF_FOLDERS},
    }


def get_celebdf(raw: Path, url: str | None, archive: str | None) -> None:
    if not (url or archive):
        sys.exit(
            "Celeb-DF v2 requires approval.\n"
            f"  1. Request access via the form linked at {CELEBDF_REPO}\n"
            "  2. Re-run with --url '<link>' or --archive /path/to/Celeb-DF-v2.zip"
        )
    _ingest(raw, url, archive)


VERIFY = {
    "dfdc": verify_dfdc,
    "asvspoof": verify_asvspoof,
    "fakeavceleb": verify_fakeavceleb,
    "celebdf": verify_celebdf,
}


def status(cfg) -> None:
    print(f"data_root: {cfg['data_root']}")
    for name, fn in VERIFY.items():
        info = fn(get_path(cfg, "raw", name))
        flag = "OK     " if info.pop("ok") else "MISSING"
        print(f"  [{flag}] {name:12s} {info}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["status", *VERIFY])
    p.add_argument("--url", help="download link received by email (fakeavceleb / celebdf)")
    p.add_argument("--archive", help="already-downloaded zip/tar to extract")
    p.add_argument("--check", action="store_true", help="asvspoof: only test the download link")
    p.add_argument("--keep-zip", action="store_true", help="keep archive after extracting")
    args = p.parse_args(argv)

    cfg = load_config()
    if args.dataset == "status":
        status(cfg)
        return
    raw = get_path(cfg, "raw", args.dataset)
    if args.dataset == "dfdc":
        get_dfdc(raw, args.archive)
    elif args.dataset == "asvspoof":
        if args.archive:
            extract(Path(args.archive).expanduser(), raw)
        else:
            get_asvspoof(raw, check=args.check, keep_zip=args.keep_zip)
    elif args.dataset == "fakeavceleb":
        get_fakeavceleb(raw, args.url, args.archive)
    elif args.dataset == "celebdf":
        get_celebdf(raw, args.url, args.archive)
    info = VERIFY[args.dataset](raw)
    print(("Verified: " if info.get("ok") else "Verification FAILED: ") + str(info))


if __name__ == "__main__":
    main()
