"""Scan raw datasets into one unified manifest CSV per dataset.

Columns:
  sample_id    unique id within the dataset
  source       dataset name (dfdc / asvspoof / fakeavceleb / celebdf)
  path         media file, relative to the raw root (portable between Mac and Colab)
  modality     "av" (video with audio) | "video" | "audio"
  label        1 = fake (video or audio manipulated), 0 = real
  video_label  1 / 0, or -1 if unknown / not applicable
  audio_label  1 / 0, or -1 if unknown / not applicable
  group        identity used for leakage-free splitting (same person -> same split)
  split        official split if the dataset defines one, else empty
  category     finer label for per-category results: DFDC REAL/FAKE, FakeAVCeleb folder
               (RealVideo-FakeAudio, ...), ASVspoof attack system (bonafide, A01..A19),
               Celeb-DF folder

Usage:
  python -m deepfake.data.manifest dfdc
  python -m deepfake.data.manifest all
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from deepfake.config import get_path, load_config
from deepfake.data import download as dl

COLUMNS = ["sample_id", "source", "path", "modality", "label",
           "video_label", "audio_label", "group", "split", "category"]


def _rel(p: Path, root: Path) -> str:
    return str(p.relative_to(root))


def build_dfdc(raw: Path, root: Path) -> pd.DataFrame:
    """DFDC train sample: metadata.json maps file -> {label: REAL|FAKE, original}."""
    d = dl.find_dir_with(raw, ("metadata.json",))
    meta = json.loads((d / "metadata.json").read_text())
    rows = []
    for name, m in meta.items():
        f = d / name
        if not f.exists():
            continue
        fake = int(m["label"] == "FAKE")
        # A fake and the real video it was made from share a group, so they never
        # land in different splits.
        group = Path(m.get("original") or name).stem if fake else Path(name).stem
        rows.append(dict(sample_id=Path(name).stem, source="dfdc", path=_rel(f, root),
                         modality="av", label=fake, video_label=fake, audio_label=-1,
                         group=group, split="", category=m["label"]))
    return pd.DataFrame(rows, columns=COLUMNS)


def build_fakeavceleb(raw: Path, root: Path) -> pd.DataFrame:
    """FakeAVCeleb: <Category>/<ethnicity>/<gender>/<idXXXXX>/<file>.mp4"""
    d = dl.find_dir_with(raw, dl.FAKEAVCELEB_CATEGORIES)
    rows = []
    for cat in dl.FAKEAVCELEB_CATEGORIES:
        video_fake = int(cat.startswith("FakeVideo"))
        audio_fake = int(cat.endswith("FakeAudio"))
        for f in sorted((d / cat).rglob("*.mp4")):
            ident = next((p for p in f.parts if re.fullmatch(r"id\d+", p)), f.parent.name)
            rel = f.relative_to(d / cat)
            rows.append(dict(
                sample_id=f"{cat}/{rel.with_suffix('')}", source="fakeavceleb",
                path=_rel(f, root), modality="av", label=int(video_fake or audio_fake),
                video_label=video_fake, audio_label=audio_fake, group=ident, split="",
                category=cat,
            ))
    return pd.DataFrame(rows, columns=COLUMNS)


def build_asvspoof(raw: Path, root: Path) -> pd.DataFrame:
    """ASVspoof 2019 LA protocol lines: SPEAKER FILE - SYSTEM bonafide|spoof"""
    d = dl.find_dir_with(raw, ("ASVspoof2019_LA_cm_protocols", "ASVspoof2019_LA_train"))
    protocols = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "val": "ASVspoof2019.LA.cm.dev.trl.txt",
        "test": "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    folders = {"train": "train", "val": "dev", "test": "eval"}
    rows = []
    for split, proto in protocols.items():
        for line in (d / "ASVspoof2019_LA_cm_protocols" / proto).read_text().splitlines():
            speaker, file_id, _, system, key = line.split()
            f = d / f"ASVspoof2019_LA_{folders[split]}" / "flac" / f"{file_id}.flac"
            fake = int(key == "spoof")
            rows.append(dict(sample_id=file_id, source="asvspoof", path=_rel(f, root),
                             modality="audio", label=fake, video_label=-1, audio_label=fake,
                             group=speaker, split=split,
                             category="bonafide" if system == "-" else system))
    return pd.DataFrame(rows, columns=COLUMNS)


def build_celebdf(raw: Path, root: Path) -> pd.DataFrame:
    """Celeb-DF v2: Celeb-real/id0_0000.mp4, Celeb-synthesis/id0_id16_0000.mp4,
    YouTube-real/00000.mp4, plus the official List_of_testing_videos.txt."""
    d = dl.find_dir_with(raw, dl.CELEBDF_FOLDERS)
    test_list = d / "List_of_testing_videos.txt"
    test = set()
    if test_list.exists():
        test = {ln.split()[1] for ln in test_list.read_text().splitlines() if ln.strip()}
    rows = []
    for folder in dl.CELEBDF_FOLDERS:
        fake = int(folder == "Celeb-synthesis")
        for f in sorted((d / folder).glob("*.mp4")):
            key = f"{folder}/{f.name}"
            # Group by the identity appearing in the video (target id for synthesis).
            ids = re.findall(r"id\d+", f.stem)
            group = ids[-1] if ids else f"yt_{f.stem}"
            rows.append(dict(sample_id=key[:-4], source="celebdf", path=_rel(f, root),
                             modality="video", label=fake, video_label=fake, audio_label=-1,
                             group=group, split="test" if key in test else "", category=folder))
    return pd.DataFrame(rows, columns=COLUMNS)


BUILDERS = {
    "dfdc": build_dfdc,
    "fakeavceleb": build_fakeavceleb,
    "asvspoof": build_asvspoof,
    "celebdf": build_celebdf,
}


def build(name: str, cfg) -> Path:
    root = get_path(cfg, "raw")
    df = BUILDERS[name](root / name, root)
    out = get_path(cfg, "manifests", f"{name}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{name}: {len(df)} samples, {int(df.label.sum())} fake / {int((df.label == 0).sum())} real -> {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["all", *BUILDERS])
    args = p.parse_args(argv)
    cfg = load_config()
    names = BUILDERS if args.dataset == "all" else [args.dataset]
    for name in names:
        if not dl.VERIFY[name](get_path(cfg, "raw", name))["ok"]:
            print(f"{name}: not downloaded, skipping")
            continue
        build(name, cfg)


if __name__ == "__main__":
    main()
