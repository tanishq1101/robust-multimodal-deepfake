"""Command line for deepfake.analyzer: any video, audio or image file, or all demo samples.

Usage:
  python -m deepfake.analyze clip.mp4                       # JSON summary on stdout
  python -m deepfake.analyze photo.jpg --out result/        # + result.json and evidence PNGs
  python -m deepfake.analyze --samples --out outputs/demo_samples   # results.md + results.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from deepfake.analyzer import Analyzer
from deepfake.config import REPO_ROOT

SAMPLES_DIR = REPO_ROOT / "demo_samples"
IMAGE_KEYS = {"overlays", "spectrogram", "timeline", "annotated", "face_crop"}


def summary(res: dict) -> dict:
    """The result without image arrays, safe for JSON."""
    def clean(d):
        return None if d is None else {k: v for k, v in d.items() if k not in IMAGE_KEYS}
    out = {k: v for k, v in res.items() if k not in IMAGE_KEYS}
    out["face"], out["voice"] = clean(res["face"]), clean(res["voice"])
    return out


def save(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(summary(res), indent=2))
    if res.get("timeline") is not None:
        plt.imsave(out / "timeline.png", res["timeline"])
    if res["voice"] and res["voice"].get("spectrogram") is not None:
        plt.imsave(out / "spectrogram_gradcam.png", res["voice"]["spectrogram"])
    for i, (img, _) in enumerate((res["face"] or {}).get("overlays", [])):
        plt.imsave(out / f"face_gradcam_{i + 1}.png", img)


def _pct(p) -> str:
    return "–" if p is None else f"{100 * p:.1f}%"


def run_samples(analyzer: Analyzer, out: Path, threshold: float) -> pd.DataFrame:
    manifest = json.loads((SAMPLES_DIR / "samples.json").read_text())
    rows = []
    for s in manifest:
        t0 = time.time()
        res = analyzer.analyse(SAMPLES_DIR / s["path"], threshold=threshold, with_explanation=False)
        rows.append({"sample": s["path"], "kind": s["kind"], "truth": s["label"].upper(), "category": s["category"],
                     "face P(fake)": _pct(res["face"]["p_fake"]) if res["face"] else "–",
                     "voice P(fake)": _pct(res["voice"]["p_fake"]) if res["voice"] else "–",
                     "P(fake)": _pct(res["p_fake"]), "verdict": res["verdict"],
                     "correct": "✓" if res["verdict"] == s["label"].upper() else "✗",
                     "seconds": round(time.time() - t0, 1)})
        print(f"{s['path']:<40} truth {s['label']:<5} → {res['verdict']:<12} P(fake) {_pct(res['p_fake'])}", flush=True)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    caps = analyzer.capabilities()
    n_ok = (df.correct == "✓").sum()
    md = ["# Demo samples: results", "",
          f"- Face model: {caps['face_model'] or 'none'}",
          f"- Voice model: {caps['voice_model'] or 'none'}",
          f"- Threshold: {threshold:.2f}; video verdict = max(face, voice) unless a fusion model is loaded",
          f"- Correct: **{n_ok} / {len(df)}**", "", df.to_markdown(index=False)]
    (out / "results.md").write_text("\n".join(md) + "\n")
    print(f"\n{n_ok}/{len(df)} correct · saved {out / 'results.md'}")
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="video, audio or image file")
    p.add_argument("--samples", action="store_true", help="analyse every clip in demo_samples/")
    p.add_argument("--checkpoint", help="project checkpoint (default: best available)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", help="folder for result.json + evidence images (or results.md with --samples)")
    p.add_argument("--device")
    args = p.parse_args(argv)
    if not args.file and not args.samples:
        p.error("give a file or --samples")
    analyzer = Analyzer.from_defaults(args.checkpoint, args.device, args.threshold)
    if args.samples:
        return run_samples(analyzer, Path(args.out or "outputs/demo_samples"), args.threshold)
    res = analyzer.analyse(args.file, with_explanation=bool(args.out))
    print(json.dumps(summary(res), indent=2))
    if args.out:
        save(res, Path(args.out))
        print(f"Saved to {args.out}")
    return res


if __name__ == "__main__":
    main()
