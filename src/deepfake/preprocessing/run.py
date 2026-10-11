"""Batch preprocessing: video -> face crops (.npz) + log-mel spectrogram (.npy) +
16 kHz waveform (.flac, used for audio degradations in Phase 5).

Re-running skips samples that are already processed, so it is safe after a Colab disconnect.

Usage:
  # a dataset that has a manifest (see deepfake.data.manifest / splits)
  python -m deepfake.preprocessing.run dfdc --workers 2
  python -m deepfake.preprocessing.run fakeavceleb --per-class 500   # balanced subset

  # any folder of videos (quick experiments / tests)
  python -m deepfake.preprocessing.run --input path/to/videos --out path/to/output
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from deepfake.config import get_path, load_config
import soundfile as sf

from deepfake.preprocessing.audio import load_audio, log_mel
from deepfake.preprocessing.faces import FaceDetector, extract_faces, save_faces
from deepfake.preprocessing.frames import sample_frames

_detector: FaceDetector | None = None
_cfg: dict | None = None


def _init_worker(cfg: dict, use_detector: bool) -> None:
    global _detector, _cfg
    _cfg = cfg
    _detector = FaceDetector(cfg["video"]["detector_score"]) if use_detector else None


def process_one(job: tuple[str, str, str, str, str, str]) -> dict:
    """job = (sample_id, media_path, modality, faces_out, mel_out, wav_out) — absolute paths."""
    sample_id, media, modality, faces_out, mel_out, wav_out = job
    v, a = _cfg["video"], _cfg["audio"]
    info = {"sample_id": sample_id, "n_frames": 0, "n_detected": 0, "audio_seconds": 0.0, "error": ""}
    try:
        if modality in ("av", "video"):
            if Path(faces_out).exists():
                with np.load(faces_out) as z:
                    info["n_frames"], info["n_detected"] = len(z["detected"]), int(z["detected"].sum())
            else:
                frames = sample_frames(media, v["num_frames"])
                faces, detected = extract_faces(frames, _detector, v["face_size"], v["face_margin"])
                save_faces(faces_out, faces, detected, v["jpeg_quality"])
                info["n_frames"], info["n_detected"] = len(faces), int(detected.sum())
        if modality in ("av", "audio"):
            wav = None
            if not Path(wav_out).exists():
                wav = load_audio(media, a["sample_rate"], a["max_seconds"])
                if len(wav):
                    Path(wav_out).parent.mkdir(parents=True, exist_ok=True)
                    sf.write(wav_out, np.clip(wav, -1, 1), a["sample_rate"], format="FLAC", subtype="PCM_16")
            if Path(mel_out).exists():
                mel = np.load(mel_out)
            else:
                if wav is None:
                    wav = load_audio(media, a["sample_rate"], a["max_seconds"])
                mel = log_mel(wav, a["sample_rate"], a["n_fft"], a["win_length"], a["hop_length"], a["n_mels"])
                if len(wav) == 0:
                    mel = mel[:, :0]  # no audio track
                Path(mel_out).parent.mkdir(parents=True, exist_ok=True)
                np.save(mel_out, mel)
            info["audio_seconds"] = mel.shape[1] * a["hop_length"] / a["sample_rate"]
    except Exception as e:  # keep going; failures are listed in the index
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def balanced_subset(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Up to `per_class` samples for every (split, video_label, audio_label) combination."""
    keys = ["split", "video_label", "audio_label"]
    return (df.groupby(keys, group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_class), random_state=seed))
              .reset_index(drop=True))


def run(df: pd.DataFrame, media_root: Path, out_root: Path, cfg: dict, workers: int,
        use_detector: bool = True) -> pd.DataFrame:
    """Process every row of a manifest. faces_path / mel_path / wav_path in the returned index are
    relative to out_root (where index.csv is saved), so the index works from both Mac and Colab.
    Re-running on an older output folder only adds what is missing (e.g. the wav/ files)."""
    jobs = []
    if df.empty:
        raise ValueError("No samples to preprocess")
    if df.sample_id.duplicated().any():
        raise ValueError("Duplicate sample_id would overwrite processed files")
    for r in df.itertuples():
        stem = str(r.sample_id)
        if not stem or Path(stem).is_absolute() or ".." in Path(stem).parts:
            raise ValueError(f"Unsafe sample_id: {stem!r}")
        jobs.append((stem, str(media_root / r.path), r.modality,
                     str(out_root / "faces" / f"{stem}.npz"), str(out_root / "mel" / f"{stem}.npy"),
                     str(out_root / "wav" / f"{stem}.flac")))
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg, use_detector)) as ex:
            results = list(tqdm(ex.map(process_one, jobs, chunksize=4), total=len(jobs)))
    else:
        _init_worker(cfg, use_detector)
        results = [process_one(j) for j in tqdm(jobs)]
    res = pd.DataFrame(results)
    out = df.merge(res, on="sample_id")
    out["face_detection"] = use_detector
    has_v = out.modality.isin(["av", "video"]) & (out.n_frames > 0)
    has_a = out.modality.isin(["av", "audio"]) & (out.audio_seconds > 0)
    out["faces_path"] = np.where(has_v, [f"faces/{s}.npz" for s in out.sample_id], "")
    out["mel_path"] = np.where(has_a, [f"mel/{s}.npy" for s in out.sample_id], "")
    out["wav_path"] = [f"wav/{s}.flac" if (a and (out_root / "wav" / f"{s}.flac").exists()) else ""
                       for s, a in zip(out.sample_id, has_a)]
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", help="dataset name with a manifest in manifests/")
    p.add_argument("--input", help="folder of videos (instead of a dataset)")
    p.add_argument("--out", help="output folder for --input mode")
    p.add_argument("--per-class", type=int, help="balanced subset size per split/label combination")
    p.add_argument("--limit", type=int, help="process only the first N samples (smoke test)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--no-detector", action="store_true", help="centre crops instead of face detection")
    args = p.parse_args(argv)
    cfg = load_config()

    if args.input:
        media_root = Path(args.input).resolve()
        out_root = Path(args.out or get_path(cfg, "processed", "custom")).resolve()
        files = sorted(f for f in media_root.rglob("*") if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
        df = pd.DataFrame({
            "sample_id": [str(f.relative_to(media_root).with_suffix("")) for f in files],
            "source": "custom", "path": [str(f.relative_to(media_root)) for f in files],
            "modality": "av", "label": -1, "video_label": -1, "audio_label": -1,
            "group": [f.stem for f in files], "split": "test", "category": "",
        })
        index_path = out_root / "index.csv"
    elif args.dataset:
        media_root = get_path(cfg, "raw")
        out_root = get_path(cfg, "processed", args.dataset)
        df = pd.read_csv(get_path(cfg, "manifests", f"{args.dataset}.csv"), keep_default_na=False)
        if (df["split"] == "").any():
            p.error(f"run `python -m deepfake.data.splits {args.dataset}` first")
        index_path = out_root / "index.csv"
    else:
        p.error("give a dataset name or --input")

    if args.per_class:
        df = balanced_subset(df, args.per_class, cfg["seed"])
    if args.limit:
        df = df.head(args.limit)
    print(f"Processing {len(df)} samples -> {out_root}")
    out = run(df, media_root, out_root, cfg, args.workers, use_detector=not args.no_detector)
    out_root.mkdir(parents=True, exist_ok=True)
    out.to_csv(index_path, index=False)
    failed = out[out.error != ""]
    v = out[out.n_frames > 0]
    print(f"Done: {len(out) - len(failed)} ok, {len(failed)} failed. Index: {index_path}")
    if len(v):
        print(f"Face detection rate: {v.n_detected.sum() / v.n_frames.sum():.1%}")
    if len(failed):
        print(failed[["sample_id", "error"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
