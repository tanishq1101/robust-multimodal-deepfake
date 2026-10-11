"""Generate the Colab notebooks in notebooks/ (edit here, not in the .ipynb files).

Usage:  python scripts/make_notebooks.py
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "notebooks"

SETUP = '''# --- Session setup: run this first, every time you open the notebook ---
REPO_URL = "https://github.com/tanishq1101/robust-multimodal-deepfake.git"  # "" = use code copied to Drive

import os, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")

if REPO_URL:
    CODE = "/content/DeepFake"
    if os.path.exists(CODE):
        subprocess.run(["git", "-C", CODE, "pull"], check=True)
    else:
        subprocess.run(["git", "clone", REPO_URL, CODE], check=True)
else:
    # No GitHub repo yet: upload the project folder to Google Drive at MyDrive/deepfake_project/code
    CODE = "/content/drive/MyDrive/deepfake_project/code"
    assert os.path.exists(CODE), f"Upload the project folder to {CODE} or set REPO_URL"

os.chdir(CODE)
!pip install -q -r requirements-colab.txt && pip install -q -e .
sys.path.insert(0, f"{CODE}/src")

from deepfake.config import load_config
cfg = load_config()
print("data_root (Drive):", cfg["data_root"])
print("raw data (VM disk):", cfg["paths"]["raw"])'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def notebook(cells, gpu=False):
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    return {"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": cells}


nb00 = notebook([
    md("""
# 00 · Colab setup
Checks the GPU, mounts Google Drive, installs the project and runs the test suite.

**Before running:** `Runtime → Change runtime type → T4 GPU`.
"""),
    code("!nvidia-smi"),
    code(SETUP),
    code("""
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
"""),
    md("Run the test suite (synthetic video, no dataset needed). Everything should pass."),
    code("!python -m pytest -q"),
    code("!python -m deepfake.data.download status"),
], gpu=True)

nb01 = notebook([
    md("""
# 01 · Download datasets
Raw datasets are downloaded to the Colab VM disk (`/content/raw`, large but **wiped when the session ends**).
Only manifests and processed data (notebook 02) are saved to Google Drive.
So run **01 → 02 in the same session**.

| Dataset | Size | Access |
|---|---|---|
| DFDC train sample | ~4 GB download | Kaggle account + API token |
| ASVspoof 2019 LA | 7.6 GB | free |
| FakeAVCeleb | ~ several GB | request form → link by email |
| Celeb-DF v2 | ~ 10 GB | request form → link by email |
"""),
    code(SETUP),
    md("""
## 1. DFDC sample (Kaggle)
1. Log in at kaggle.com and **accept the rules**: https://www.kaggle.com/competitions/deepfake-detection-challenge/rules
2. Kaggle → *Settings* → *API* → **Create New Token** → downloads `kaggle.json`.
3. Run the next cell and upload `kaggle.json` when asked.
"""),
    code("""
from google.colab import files
import os, shutil
if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
    os.chdir("/content")       # keep the token off Google Drive
    files.upload()             # choose kaggle.json
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    shutil.move("/content/kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
    os.chdir(CODE)
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
print("Kaggle token ready")
"""),
    code("!python -m deepfake.data.download dfdc"),
    md("## 2. ASVspoof 2019 LA (free, ~10–20 min)"),
    code("!python -m deepfake.data.download asvspoof"),
    md("""
## 3. FakeAVCeleb (after your request is approved)
Paste the link from the approval email below. If they sent a `download.sh` script instead,
upload it with the second cell.
"""),
    code("""
FAKEAVCELEB_URL = ""  # paste link from the email
if FAKEAVCELEB_URL:
    !python -m deepfake.data.download fakeavceleb --url "$FAKEAVCELEB_URL"
"""),
    code("""
# Alternative: run the download script they emailed you
# os.chdir("/content"); from google.colab import files; files.upload()   # upload download.sh
# !mkdir -p /content/raw/fakeavceleb && cd /content/raw/fakeavceleb && bash /content/download.sh
# os.chdir(CODE)
"""),
    md("## 4. Celeb-DF v2 (after approval)"),
    code("""
CELEBDF_URL = ""  # paste link from the email
if CELEBDF_URL:
    !python -m deepfake.data.download celebdf --url "$CELEBDF_URL"
"""),
    md("## 5. Check, then build manifests + identity-wise splits (saved to Drive)"),
    code("!python -m deepfake.data.download status"),
    code("""
!python -m deepfake.data.manifest all
import os
for name in ["dfdc", "asvspoof", "fakeavceleb", "celebdf"]:
    if os.path.exists(f"{cfg['paths']['manifests']}/{name}.csv"):
        print(f"--- {name}")
        !python -m deepfake.data.splits {name}
"""),
])

nb02 = notebook([
    md("""
# 02 · Preprocess
Face crops (16 frames/video, 224×224) and log-mel spectrograms → saved to Google Drive
under `deepfake_project/processed/<dataset>/`. Raw data from notebook 01 must be present
in this session. Re-running skips files that are already done.
"""),
    code(SETUP),
    md("## Smoke test: 10 DFDC videos"),
    code("!python -m deepfake.preprocessing.run dfdc --limit 10 --workers 2"),
    md("## Full runs"),
    code("""
# DFDC sample: all ~400 videos
!python -m deepfake.preprocessing.run dfdc --workers 2
"""),
    code("""
# ASVspoof: balanced subset (the eval split alone has ~71k files)
!python -m deepfake.preprocessing.run asvspoof --per-class 2500 --workers 2
"""),
    code("""
# FakeAVCeleb: balanced across the 4 categories (only 500 real videos exist)
import os
if os.path.exists(f"{cfg['paths']['manifests']}/fakeavceleb.csv"):
    !python -m deepfake.preprocessing.run fakeavceleb --per-class 400 --workers 2
"""),
    md("## Inspect the results"),
    code("""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from deepfake.preprocessing.faces import load_faces

DATASET = "dfdc"
root = Path(cfg["paths"]["processed"]) / DATASET
idx = pd.read_csv(root / "index.csv", keep_default_na=False)
print(idx.groupby(["split", "label"]).size().unstack(fill_value=0))
print("errors:", (idx.error != "").sum())
v = idx[idx.n_frames > 0]
print(f"face detection rate: {v.n_detected.sum() / v.n_frames.sum():.1%}")
"""),
    code("""
fig, axes = plt.subplots(2, 6, figsize=(15, 5))
for row, label in enumerate([0, 1]):
    sample = idx[(idx.label == label) & (idx.faces_path != "")].sample(1, random_state=0).iloc[0]
    faces, detected = load_faces(root / sample.faces_path)
    for j in range(5):
        axes[row, j].imshow(faces[j * 3]); axes[row, j].axis("off")
    axes[row, 0].set_title(("REAL" if label == 0 else "FAKE") + f" · {sample.sample_id}", loc="left")
    if sample.mel_path:
        axes[row, 5].imshow(np.load(root / sample.mel_path).astype(float), origin="lower", aspect="auto")
        axes[row, 5].set_title("log-mel")
    axes[row, 5].axis("off")
plt.tight_layout()
"""),
    md("## Dataset sanity check (what the models will receive)"),
    code("""
from deepfake.datasets import MultimodalDataset
ds = MultimodalDataset(root / "index.csv", "train", cfg, train=True)
item = ds[0]
print(len(ds), "train samples")
print({k: tuple(v.shape) for k, v in item.items()})
"""),
])

nb03 = notebook([
    md("""
# 03 · Phase 2: train the visual branch
EfficientNet-B0 (ImageNet-pretrained) fine-tuned on the face crops from notebook 02.

- **Runtime:** `Runtime → Change runtime type → T4 GPU`.
- **Needs:** `processed/<dataset>/index.csv` on Drive (notebook 02). Raw data is *not* needed.
- **Disconnected?** Re-run the setup cell and the training cell: training **resumes** from `last.pt`.
"""),
    code(SETUP),
    code("""
DATASET = "dfdc"          # "fakeavceleb" once it is downloaded + preprocessed
RUN = f"visual_{DATASET}"
EPOCHS = 10
BATCH = 8                 # videos per batch (x 8 frames each); lower to 4 if CUDA runs out of memory
"""),
    md("## Train (+ automatic test-set evaluation of the best checkpoint)"),
    code("""
!python -m deepfake.training.train_visual --dataset $DATASET --run-name $RUN --epochs $EPOCHS --batch-size $BATCH --workers 2
"""),
    md("## Training curves"),
    code("""
import json, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Image, display

run_dir = Path(cfg["paths"]["checkpoints"]) / RUN
display(Image(str(run_dir / "curves.png")))   # saved by the training script
pd.read_csv(run_dir / "history.csv")
"""),
    md("## Test results"),
    code("""
m = json.loads((run_dir / "test" / "metrics.json").read_text())
print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k not in ("index", "checkpoint")})
display(Image(str(run_dir / "test" / "roc.png")))
"""),
    code("""
# Score distribution: a good model pushes real towards 0 and fake towards 1
p = pd.read_csv(run_dir / "test" / "predictions.csv")
plt.figure(figsize=(6, 3))
for lab, name in [(0, "real"), (1, "fake")]:
    plt.hist(p[p.label == lab].score, bins=20, range=(0, 1), alpha=0.6, label=name)
plt.xlabel("P(fake)"); plt.legend(); plt.tight_layout()
"""),
    md("""
## Cross-dataset check (optional)
Evaluate this model on another processed dataset, e.g. Celeb-DF, to measure generalisation.
"""),
    code("""
OTHER = "celebdf"
import os
if os.path.exists(f"{cfg['paths']['processed']}/{OTHER}/index.csv"):
    !python -m deepfake.evaluation.evaluate --checkpoint {run_dir}/best.pt --dataset $OTHER --split test
"""),
], gpu=True)

nb04 = notebook([
    md("""
# 04 · Phase 3: train the audio branch
ResNet-18 (ImageNet-pretrained) on log-mel spectrograms, two steps:
1. **Pre-train on ASVspoof 2019 LA**: lots of bona fide vs. synthetic/converted speech.
2. **Fine-tune on FakeAVCeleb audio**: adapts to the cloned voices in our main dataset.

- **Runtime:** T4 GPU. **Needs:** `processed/asvspoof/index.csv` (and `processed/fakeavceleb/`) from notebook 02.
- DFDC has no separate audio labels, so it is not used for this branch.
- **Disconnected?** Re-run setup + the same cell: training resumes from `last.pt`.
"""),
    code(SETUP),
    code("""
import json, os, pandas as pd
from pathlib import Path
from IPython.display import Image, display
CKPT = Path(cfg["paths"]["checkpoints"])

def show_run(run):
    run_dir = CKPT / run
    display(Image(str(run_dir / "curves.png")))
    m = json.loads((run_dir / "test" / "metrics.json").read_text())
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k not in ("index", "checkpoint")})
    display(Image(str(run_dir / "test" / "roc.png")))
    if (run_dir / "test" / "per_category.csv").exists():
        display(pd.read_csv(run_dir / "test" / "per_category.csv").round(4))
"""),
    md("## Step 1: pre-train on ASVspoof 2019 LA"),
    code("""
!python -m deepfake.training.train_audio --dataset asvspoof --run-name audio_asvspoof --epochs 15 --batch-size 32 --workers 2
"""),
    md("""
Results. `per_category` lists the **EER for each attack system**. The ASVspoof eval set contains
attacks (A07–A19) never seen in training, so this measures generalisation to new voice-cloning methods.
"""),
    code('show_run("audio_asvspoof")'),
    md("## Step 2: fine-tune on FakeAVCeleb audio"),
    code("""
if os.path.exists(f"{cfg['paths']['processed']}/fakeavceleb/index.csv"):
    !python -m deepfake.training.train_audio --dataset fakeavceleb --run-name audio_fakeavceleb --init-from {CKPT}/audio_asvspoof/best.pt --lr 3e-5 --freeze-epochs 0 --epochs 10 --batch-size 32 --workers 2
else:
    print("FakeAVCeleb not preprocessed yet: run notebooks 01 + 02 once access is approved.")
"""),
    code("""
if (CKPT / "audio_fakeavceleb" / "test" / "metrics.json").exists():
    show_run("audio_fakeavceleb")
"""),
    md("""
## Ablation: does ASVspoof pre-training help?
Train on FakeAVCeleb from ImageNet weights only and compare test AUC/EER with step 2.
"""),
    code("""
if os.path.exists(f"{cfg['paths']['processed']}/fakeavceleb/index.csv"):
    !python -m deepfake.training.train_audio --dataset fakeavceleb --run-name audio_fakeavceleb_noprt --epochs 10 --batch-size 32 --workers 2
    rows = []
    for run in ["audio_fakeavceleb_noprt", "audio_fakeavceleb"]:
        m = json.loads((CKPT / run / "test" / "metrics.json").read_text())
        rows.append({"run": run, "auc": m["auc"], "eer": m["eer"], "accuracy": m["accuracy"]})
    display(pd.DataFrame(rows).round(4))
"""),
], gpu=True)

nb05 = notebook([
    md("""
# 05 · Phase 4: multimodal fusion
Combines the best visual (Phase 2) and audio (Phase 3) models, trained on FakeAVCeleb, and
compares everything on the **same test set** with the **overall label**: fake if the face OR the voice is fake.

| Model | What it is |
|---|---|
| visual_only | Phase 2 face model alone |
| audio_only | Phase 3 voice model alone |
| late | no training: best of weighted-average / max / noisy-OR, tuned on validation |
| intermediate | embeddings of both branches → MLP; branches frozen first, then fine-tuned; modality dropout 0.2 |

**Needs:** `checkpoints/visual_fakeavceleb/best.pt` (notebook 03 with `DATASET="fakeavceleb"`) and
`checkpoints/audio_fakeavceleb/best.pt` (notebook 04, step 2). T4 GPU runtime.
"""),
    code(SETUP),
    code("""
import json, pandas as pd
from pathlib import Path
from IPython.display import Image, Markdown, display
CKPT = Path(cfg["paths"]["checkpoints"])
VISUAL = CKPT / "visual_fakeavceleb" / "best.pt"
AUDIO = CKPT / "audio_fakeavceleb" / "best.pt"
for p in (VISUAL, AUDIO):
    print(("OK      " if p.exists() else "MISSING ") + str(p))
"""),
    md("## Late fusion (calibration only, takes ~1 min)"),
    code("""
!python -m deepfake.training.train_fusion --fusion late --visual-ckpt {VISUAL} --audio-ckpt {AUDIO} --workers 2
display(pd.read_csv(CKPT / "fusion_late_fakeavceleb" / "calibration.csv").head(10).round(4))
"""),
    md("## Intermediate fusion (training)"),
    code("""
!python -m deepfake.training.train_fusion --fusion intermediate --visual-ckpt {VISUAL} --audio-ckpt {AUDIO} --epochs 10 --batch-size 8 --workers 2
display(Image(str(CKPT / "fusion_intermediate_fakeavceleb" / "curves.png")))
"""),
    md("## Comparison table: the core result of the project"),
    code("""
models = {
    "visual_only": VISUAL,
    "audio_only": AUDIO,
    "late": CKPT / "fusion_late_fakeavceleb" / "best.pt",
    "intermediate": CKPT / "fusion_intermediate_fakeavceleb" / "best.pt",
}
args = " ".join(f"--model {k}={v}" for k, v in models.items() if Path(v).exists())
!python -m deepfake.evaluation.compare --dataset fakeavceleb {args} --workers 2
OUT = Path(cfg["paths"]["outputs"]) / "compare_fakeavceleb_test"
display(Markdown((OUT / "comparison.md").read_text()))
display(Image(str(OUT / "comparison.png")))
"""),
    md("""
**How to read it:** visual_only should fail on *RealVideo-FakeAudio* and audio_only on
*FakeVideo-RealAudio*. Fusion should handle all four categories. Copy `comparison.md` into
`reports/results.md`.
"""),
], gpu=True)

OUT.mkdir(parents=True, exist_ok=True)
for name, nb in [("00_colab_setup", nb00), ("01_download_datasets", nb01), ("02_preprocess", nb02),
                 ("03_train_visual", nb03), ("04_train_audio", nb04),
                 ("05_train_fusion", nb05)]:
    (OUT / f"{name}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", name)
