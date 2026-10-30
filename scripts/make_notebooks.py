"""Generate the single Colab notebook notebooks/deepfake_pipeline.ipynb (edit here, not in the .ipynb).

The notebook runs the whole project top to bottom: setup, data, preprocessing, the face, voice and
fusion models, robustness, Grad-CAM, the demo samples, the report and the demo app. Every stage has
an on/off switch in the settings cell and checks its own inputs, so a stage whose data or
checkpoint is missing prints why and is skipped instead of crashing the rest of the run.

Usage:  python scripts/make_notebooks.py [out_dir]
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "notebooks"
NAME = "deepfake_pipeline.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [
    md("""
# Robust Multimodal Deepfake Analysis: full pipeline

One notebook for the whole project. Run it on Google Colab with a **T4 GPU**
(`Runtime → Change runtime type → T4 GPU`), then `Runtime → Run all`.

| § | Stage | Needs |
|---|---|---|
| 1 | Setup: GPU check, Google Drive, install, tests | nothing |
| 2 | Settings: dataset and which stages to run | nothing |
| 3 | Download datasets, manifests, identity-wise splits | Kaggle token (DFDC), request links (FakeAVCeleb, Celeb-DF) |
| 4 | Preprocess: face crops, log-mel spectrograms, waveforms | §3 in the same session |
| 5 | Face model (EfficientNet-B0) | §4 |
| 6 | Voice model (ResNet-18): ASVspoof pre-training, FakeAVCeleb fine-tuning, ablation | §4 |
| 7 | Fusion (late + intermediate) and the fair comparison table | §5 + §6 on FakeAVCeleb |
| 8 | Robustness stress test and robust fine-tuning | §7 |
| 9 | Grad-CAM explanations | any trained checkpoint |
| 10 | Pretrained models on the demo samples (video, audio, image) | nothing (downloads the weights) |
| 11 | Report and the demo app with a public link | anything above |

Each stage checks its inputs. If a dataset or checkpoint is missing, the stage prints what is
missing and is skipped, so the rest of the notebook still runs. Training **resumes** from
`last.pt` if Colab disconnects: re-run §1, §2 and the stage you were in.
"""),

    # ------------------------------------------------------------------ 1. setup
    md("## 1 · Setup\nRun this first, every time you open the notebook."),
    code('''
REPO_URL = "https://github.com/tanishq1101/robust-multimodal-deepfake.git"
CODE = "/content/robust-multimodal-deepfake"

import os, subprocess, sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    if os.path.exists(CODE):
        subprocess.run(["git", "-C", CODE, "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CODE], check=True)
    os.chdir(CODE)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
    sys.path.insert(0, f"{CODE}/src")
else:   # local Jupyter: run from the repo (pip install -r requirements.txt && pip install -e .)
    CODE = str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())
    os.chdir(CODE)

import torch
from deepfake.config import get_path, load_config
cfg = load_config()
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("data_root:", cfg["data_root"])
print("raw data :", cfg["paths"]["raw"])
'''),
    code('''
import json
import pandas as pd
from IPython.display import Image, Markdown, display

CKPT = get_path(cfg, "checkpoints")
OUTS = get_path(cfg, "outputs")
PROC = get_path(cfg, "processed")
MANI = get_path(cfg, "manifests")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run(module, *args):
    """Run `python -m module args...`, stream its output, and stop the cell if it fails."""
    command = [sys.executable, "-u", "-m", module, *map(str, args)]
    print("$", " ".join(command), flush=True)
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait():
        raise RuntimeError(f"{module} failed with exit code {proc.returncode} (see the output above)")


def processed(dataset):
    return (PROC / dataset / "index.csv").is_file()


def ckpt(run_name):
    return CKPT / run_name / "best.pt"


def need(*paths):
    """True if every path exists; otherwise prints what is missing (the stage is then skipped)."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    for m in missing:
        print("missing:", m)
    if missing:
        print("→ skipped. Run the earlier stage that creates these files first.")
    return not missing


def show_run(run_name):
    """Curves, test metrics, ROC and per-category table of a finished training run."""
    d = CKPT / run_name
    if (d / "curves.png").exists():
        display(Image(str(d / "curves.png")))
    if (d / "test" / "metrics.json").exists():
        m = json.loads((d / "test" / "metrics.json").read_text())
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()
               if k not in ("index", "checkpoint", "at_eer_threshold")})
        display(Image(str(d / "test" / "roc.png")))
    if (d / "test" / "per_category.csv").exists():
        display(pd.read_csv(d / "test" / "per_category.csv").round(4))
'''),
    md("Optional: the test suite (synthetic data, no dataset needed, about 3 minutes)."),
    code('''
RUN_TESTS = False
if RUN_TESTS:
    run("pytest", "-q")
'''),

    # ------------------------------------------------------------------ 2. settings
    md("""
## 2 · Settings
`DATASET` is the main audio-visual dataset. Use `"dfdc"` for a quick face-only baseline while the
FakeAVCeleb request is pending; fusion (§7) and robustness (§8) need `"fakeavceleb"`.
"""),
    code('''
DATASET = "fakeavceleb"      # "fakeavceleb" (face + voice) or "dfdc" (face only)
EPOCHS = 10
WORKERS = 2

STAGES = {
    "download": True,         # §3
    "preprocess": True,       # §4
    "face": True,             # §5
    "voice": True,            # §6
    "fusion": True,           # §7
    "robustness": True,       # §8
    "gradcam": True,          # §9
    "demo_samples": True,     # §10
    "report": True,           # §11
    "launch_demo": False,     # §11, keeps the cell running while the app is live
}

# Paste the links from the approval emails (leave empty to skip that dataset)
FAKEAVCELEB_URL = ""
CELEBDF_URL = ""
'''),

    # ------------------------------------------------------------------ 3. download
    md("""
## 3 · Download datasets
Raw data goes to the Colab VM disk (`/content/raw`), which is **wiped when the session ends**, so run
§3 and §4 in the same session. Manifests and processed data are saved to Google Drive.

| Dataset | Size | Access |
|---|---|---|
| ASVspoof 2019 LA | 7.6 GB | free |
| DFDC train sample | ~4 GB | Kaggle account, accepted competition rules, `kaggle.json` |
| FakeAVCeleb | several GB | request form → link by email |
| Celeb-DF v2 | ~10 GB | request form → link by email |
"""),
    code('''
if STAGES["download"]:
    run("deepfake.data.download", "asvspoof")

    # DFDC needs a Kaggle API token: kaggle.com → Settings → API → Create New Token
    kaggle_token = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_token.exists() and IN_COLAB:
        from google.colab import files
        print("Upload kaggle.json to download DFDC, or press Cancel to skip it.")
        uploaded = files.upload()
        if "kaggle.json" in uploaded:
            kaggle_token.parent.mkdir(exist_ok=True)
            kaggle_token.write_bytes(uploaded["kaggle.json"])
            kaggle_token.chmod(0o600)
            Path("kaggle.json").unlink(missing_ok=True)      # keep the token out of the repo folder
    if kaggle_token.exists():
        run("deepfake.data.download", "dfdc")
    else:
        print("No Kaggle token: DFDC skipped.")

    if FAKEAVCELEB_URL:
        run("deepfake.data.download", "fakeavceleb", "--url", FAKEAVCELEB_URL)
    if CELEBDF_URL:
        run("deepfake.data.download", "celebdf", "--url", CELEBDF_URL)
    run("deepfake.data.download", "status")
'''),
    code('''
if STAGES["download"]:
    run("deepfake.data.manifest", "all")
    for name in ["dfdc", "asvspoof", "fakeavceleb", "celebdf"]:
        if (MANI / f"{name}.csv").exists():
            run("deepfake.data.splits", name)
'''),

    # ------------------------------------------------------------------ 4. preprocess
    md("""
## 4 · Preprocess
16 face crops per video (224 × 224, YuNet), a 128-bin log-mel spectrogram and a 16 kHz waveform per
clip, saved under `processed/<dataset>/` on Drive. Re-running skips finished files.
"""),
    code('''
PER_CLASS = {"asvspoof": 2500, "fakeavceleb": 400}   # balanced subsets; DFDC and Celeb-DF in full
if STAGES["preprocess"]:
    for name in ["asvspoof", "dfdc", "fakeavceleb", "celebdf"]:
        if not (MANI / f"{name}.csv").exists():
            print(f"{name}: no manifest, skipped")
            continue
        extra = ["--per-class", PER_CLASS[name]] if name in PER_CLASS else []
        run("deepfake.preprocessing.run", name, "--workers", WORKERS, *extra)
'''),
    code('''
if STAGES["preprocess"] and processed(DATASET):
    import numpy as np, matplotlib.pyplot as plt
    from deepfake.preprocessing.faces import load_faces

    root = PROC / DATASET
    idx = pd.read_csv(root / "index.csv", keep_default_na=False)
    display(idx.groupby(["split", "label"]).size().unstack(fill_value=0))
    v = idx[idx.n_frames > 0]
    print(f"errors: {(idx.error != '').sum()} · face detection rate: {v.n_detected.sum() / max(v.n_frames.sum(), 1):.1%}")

    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    for row, label in enumerate([0, 1]):
        sample = idx[(idx.label == label) & (idx.faces_path != "")].sample(1, random_state=0).iloc[0]
        faces, _ = load_faces(root / sample.faces_path)
        for j in range(5):
            axes[row, j].imshow(faces[min(j * 3, len(faces) - 1)])
        axes[row, 0].set_title(("REAL" if label == 0 else "FAKE") + f" · {sample.sample_id}", loc="left")
        if sample.mel_path:
            axes[row, 5].imshow(np.load(root / sample.mel_path).astype(float), origin="lower", aspect="auto")
            axes[row, 5].set_title("log-mel")
        for ax in axes[row]:
            ax.axis("off")
    plt.tight_layout(); plt.show()
'''),

    # ------------------------------------------------------------------ 5. face
    md("""
## 5 · Face model (EfficientNet-B0)
ImageNet-pretrained EfficientNet-B0 fine-tuned on the face crops: AdamW, cosine schedule, backbone
frozen for the first epoch, class-balanced loss, 8 random frames per video in training and all 16
in evaluation. The best checkpoint is evaluated on the test split automatically.
"""),
    code('''
VISUAL_RUN = f"visual_{DATASET}"
if STAGES["face"] and need(PROC / DATASET / "index.csv"):
    run("deepfake.training.train_visual", "--dataset", DATASET, "--run-name", VISUAL_RUN,
        "--epochs", EPOCHS, "--batch-size", 8, "--workers", WORKERS)
    show_run(VISUAL_RUN)
'''),
    md("Cross-dataset check: the same face model on Celeb-DF (only if Celeb-DF was preprocessed)."),
    code('''
if STAGES["face"] and processed("celebdf") and ckpt(VISUAL_RUN).exists():
    run("deepfake.evaluation.evaluate", "--checkpoint", ckpt(VISUAL_RUN), "--dataset", "celebdf", "--split", "test")
'''),

    # ------------------------------------------------------------------ 6. voice
    md("""
## 6 · Voice model (ResNet-18 on log-mel spectrograms)
1. **Pre-train on ASVspoof 2019 LA**: bona fide speech vs. 19 TTS / voice-conversion systems. The
   evaluation partition has attacks never seen in training, so its per-attack table measures
   generalisation to new cloning methods.
2. **Fine-tune on FakeAVCeleb audio** from the ASVspoof weights.
3. **Ablation:** FakeAVCeleb from ImageNet weights only, to measure what pre-training adds.
"""),
    code('''
AUDIO_PRE = "audio_asvspoof"
if STAGES["voice"] and need(PROC / "asvspoof" / "index.csv"):
    run("deepfake.training.train_audio", "--dataset", "asvspoof", "--run-name", AUDIO_PRE,
        "--epochs", 15, "--batch-size", 32, "--workers", WORKERS)
    show_run(AUDIO_PRE)
'''),
    code('''
AUDIO_RUN = f"audio_{DATASET}"
if STAGES["voice"] and DATASET == "fakeavceleb" and need(PROC / DATASET / "index.csv", ckpt(AUDIO_PRE)):
    run("deepfake.training.train_audio", "--dataset", DATASET, "--run-name", AUDIO_RUN,
        "--init-from", ckpt(AUDIO_PRE), "--lr", "3e-5", "--freeze-epochs", 0,
        "--epochs", EPOCHS, "--batch-size", 32, "--workers", WORKERS)
    show_run(AUDIO_RUN)

    run("deepfake.training.train_audio", "--dataset", DATASET, "--run-name", f"{AUDIO_RUN}_noprt",
        "--epochs", EPOCHS, "--batch-size", 32, "--workers", WORKERS)
    rows = [{"run": r, **{k: json.loads((CKPT / r / "test" / "metrics.json").read_text())[k]
                         for k in ("auc", "eer", "accuracy")}}
            for r in (f"{AUDIO_RUN}_noprt", AUDIO_RUN)]
    display(pd.DataFrame(rows).round(4))
'''),

    # ------------------------------------------------------------------ 7. fusion
    md("""
## 7 · Fusion and the comparison table
| Model | What it is |
|---|---|
| visual_only | §5 face model alone |
| audio_only | §6 voice model alone |
| late | no training: best of weighted average / max / noisy-OR, tuned on validation |
| intermediate | both embeddings → MLP; branches frozen first, then fine-tuned; modality dropout 0.2 |

Every model is scored on the **same test set** against the **overall label** (fake if the face
*or* the voice is fake). Expect visual_only to miss *RealVideo-FakeAudio* and audio_only to miss
*FakeVideo-RealAudio*; fusion should handle all four categories.
"""),
    code('''
FUSION_READY = DATASET == "fakeavceleb" and STAGES["fusion"] and need(ckpt(VISUAL_RUN), ckpt(AUDIO_RUN))
if FUSION_READY:
    common = ["--dataset", DATASET, "--visual-ckpt", ckpt(VISUAL_RUN), "--audio-ckpt", ckpt(AUDIO_RUN),
              "--workers", WORKERS]
    run("deepfake.training.train_fusion", "--fusion", "late", *common)
    display(pd.read_csv(CKPT / f"fusion_late_{DATASET}" / "calibration.csv").head(10).round(4))
    run("deepfake.training.train_fusion", "--fusion", "intermediate", *common, "--epochs", EPOCHS, "--batch-size", 8)
    show_run(f"fusion_intermediate_{DATASET}")
'''),
    code('''
if FUSION_READY:
    models = {"visual_only": ckpt(VISUAL_RUN), "audio_only": ckpt(AUDIO_RUN),
              "late": ckpt(f"fusion_late_{DATASET}"), "intermediate": ckpt(f"fusion_intermediate_{DATASET}")}
    args = [a for k, v in models.items() if v.exists() for a in ("--model", f"{k}={v}")]
    run("deepfake.evaluation.compare", "--dataset", DATASET, *args, "--workers", WORKERS)
    out = OUTS / f"compare_{DATASET}_test"
    display(Markdown((out / "comparison.md").read_text()))
    display(Image(str(out / "comparison.png")))
'''),

    # ------------------------------------------------------------------ 8. robustness
    md("""
## 8 · Robustness
1. **Stress test**: clean test set vs. JPEG, H.264, blur, noise, downscaling (video) and noise, MP3
   (audio), each at 3 severities, plus face-only / voice-only inputs for the fusion models.
2. **Robust fine-tuning** with random degradations (`--robust-augment`).
3. **Standard vs. robust** on the same stress test.

H.264 and MP3 run on the CPU, so the stress test uses a random subset of `LIMIT` clips per condition.
"""),
    code('''
LIMIT = 300
ROBUST_READY = STAGES["robustness"] and DATASET == "fakeavceleb" and need(ckpt(f"fusion_intermediate_{DATASET}"))


def models_arg(names):
    return [a for n, r in names.items() if ckpt(r).exists() for a in ("--model", f"{n}={ckpt(r)}")]


def show_robustness(out):
    display(Markdown((out / "robustness.md").read_text()))
    display(Image(str(out / "robustness.png")))


if ROBUST_READY:
    base = {"visual_only": VISUAL_RUN, "audio_only": AUDIO_RUN,
            "late": f"fusion_late_{DATASET}", "intermediate": f"fusion_intermediate_{DATASET}"}
    out1 = OUTS / f"robustness_{DATASET}_standard"
    run("deepfake.evaluation.robustness_eval", "--dataset", DATASET, "--label", "overall", *models_arg(base),
        "--limit", LIMIT, "--out", out1, "--workers", WORKERS)
    show_robustness(out1)
'''),
    code('''
if ROBUST_READY:
    common = ["--dataset", DATASET, "--robust-augment", "--lr", "3e-5", "--freeze-epochs", 0,
              "--epochs", 8, "--workers", WORKERS]
    run("deepfake.training.train_visual", *common, "--run-name", f"{VISUAL_RUN}_robust", "--init-from", ckpt(VISUAL_RUN))
    run("deepfake.training.train_audio", *common, "--batch-size", 32, "--run-name", f"{AUDIO_RUN}_robust",
        "--init-from", ckpt(AUDIO_RUN))
    run("deepfake.training.train_fusion", "--fusion", "intermediate", "--robust-augment", "--dataset", DATASET,
        "--epochs", 8, "--workers", WORKERS, "--run-name", f"fusion_intermediate_{DATASET}_robust",
        "--visual-ckpt", ckpt(f"{VISUAL_RUN}_robust"), "--audio-ckpt", ckpt(f"{AUDIO_RUN}_robust"))

    pairs = {"intermediate": f"fusion_intermediate_{DATASET}", "intermediate_robust": f"fusion_intermediate_{DATASET}_robust",
             "visual_only": VISUAL_RUN, "visual_robust": f"{VISUAL_RUN}_robust",
             "audio_only": AUDIO_RUN, "audio_robust": f"{AUDIO_RUN}_robust"}
    out2 = OUTS / f"robustness_{DATASET}_robust_vs_standard"
    run("deepfake.evaluation.robustness_eval", "--dataset", DATASET, "--label", "overall", *models_arg(pairs),
        "--limit", LIMIT, "--out", out2, "--workers", WORKERS)
    show_robustness(out2)
'''),

    # ------------------------------------------------------------------ 9. Grad-CAM
    md("""
## 9 · Grad-CAM explanations
Where the model looked: heatmaps over the face frames and over the spectrogram. For a fusion model,
one backward pass of the fused decision explains both. The gallery picks, per category, the most
confident correct predictions and the most confident mistakes.
"""),
    code('''
GALLERY_RUN = next((r for r in [f"fusion_intermediate_{DATASET}_robust", f"fusion_intermediate_{DATASET}",
                                VISUAL_RUN, AUDIO_RUN, AUDIO_PRE] if ckpt(r).exists()), None)
if STAGES["gradcam"] and GALLERY_RUN is None:
    print("No trained checkpoint yet: §9 skipped.")
elif STAGES["gradcam"]:
    ds_name = "asvspoof" if GALLERY_RUN == AUDIO_PRE else DATASET
    out = OUTS / f"gallery_{GALLERY_RUN}"
    if need(PROC / ds_name / "index.csv"):
        run("deepfake.explainability.gallery", "--checkpoint", ckpt(GALLERY_RUN), "--dataset", ds_name,
            "--per-category", 2, "--out", out)
        g = pd.read_csv(out / "gallery.csv")
        display(g)
        for f in g.file[:8]:
            display(Image(str(out / f)))
'''),

    # ------------------------------------------------------------------ 10. demo samples
    md("""
## 10 · Pretrained models on the demo samples
Runs the same analyser the demo app uses on the clips in `demo_samples/` (see its README for
sources and licences): DFDC videos with consenting actors, LJ Speech / VCTK recordings against
Tacotron 2, Glow-TTS and VITS speech of the same sentences, and face images from the DFDC clips.

- **Voice:** this project's ResNet-18, trained on ASVspoof 2019 LA (published release).
- **Face:** the DFDC-winning EfficientNet-B7 by Selim Seferbekov (MIT licence), used as-is until
  this project's own face model is trained in §5.

These samples are a sanity check, not a benchmark: there are only eleven. The audio is outside the
voice model's training domain, which is where it over-flags genuine speech
(`reports/voice_generalization.md`); the face model may have seen the DFDC actors during training.
"""),
    code('''
if STAGES["demo_samples"]:
    subprocess.run([sys.executable, "scripts/download_model.py"], check=True)
    run("deepfake.analyze", "--samples", "--out", OUTS / "demo_samples")
    display(Markdown((OUTS / "demo_samples" / "results.md").read_text()))
'''),

    # ------------------------------------------------------------------ 11. report + demo
    md("""
## 11 · Report and demo app
`deepfake.report` gathers every finished run into `report/results.md` with figures; download the
zip and copy it into the repo's `reports/` folder. The demo app gets a public `gradio.live` link;
the cell keeps running while the app is live, so stop it to shut the app down.
"""),
    code('''
REPORT = Path(cfg["data_root"]) / "report"
if STAGES["report"]:
    run("deepfake.report", "--out", REPORT)
    display(Markdown((REPORT / "results.md").read_text()[:6000]))
    if IN_COLAB:
        import shutil
        from google.colab import files
        files.download(shutil.make_archive("/content/report", "zip", REPORT))
'''),
    code('''
if STAGES["launch_demo"]:
    subprocess.run([sys.executable, "scripts/download_model.py"], check=True)
    subprocess.run([sys.executable, "app/app.py", "--share"])
'''),
]

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"}},
      "cells": cells}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / NAME).write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", OUT / NAME)
