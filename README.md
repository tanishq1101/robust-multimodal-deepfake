# Robust Multimodal Deepfake Analysis

Detects deepfake videos using both the **face** (video) and the **voice** (audio). It is meant to stay reliable under compression and noise, and it explains its decisions with Grad-CAM heatmaps.

```
video ──► 16 face crops ──► EfficientNet-B0 ──┐
     └──► log-mel spectrogram ──► ResNet-18 ──┴─► fusion ──► P(fake) + per-modality scores + heatmaps
```

See [PLAN.md](PLAN.md) for the full roadmap and [DATASETS.md](DATASETS.md) for how to get the data.

## Quick start

### On your Mac (writing code, running tests)
```bash
uv venv --python 3.11 .venv          # or: python3.11 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
pip install -e .
pytest                               # 19 tests, uses a synthetic video (needs ffmpeg)
```

### On Google Colab (downloading data, training)
1. Put the code where Colab can reach it, using either:
   - **GitHub** (recommended): push this repo, then set `REPO_URL` in the notebooks.
   - **Google Drive**: upload this folder to `MyDrive/deepfake_project/code` (leave out `.venv/` and `data/`).
2. Open the notebooks in order:
   - `notebooks/00_colab_setup.ipynb`: GPU check, install, tests.
   - `notebooks/01_download_datasets.ipynb`: downloads, manifests, splits.
   - `notebooks/02_preprocess.ipynb`: face crops and spectrograms saved to Drive.

## Command-line tools
```bash
python -m deepfake.data.download status            # what's downloaded
python -m deepfake.data.download asvspoof          # dfdc | asvspoof | fakeavceleb | celebdf
python -m deepfake.data.manifest all               # unified CSV per dataset
python -m deepfake.data.splits dfdc                # identity-wise train/val/test
python -m deepfake.preprocessing.run dfdc          # faces + spectrograms
python -m deepfake.preprocessing.run --input my_videos/ --out out/   # any folder of videos
```

## Project layout
```
configs/default.yaml      all settings (frames, image size, audio params, training)
src/deepfake/
  config.py               config + Mac/Colab path resolution
  data/                   download.py · manifest.py · splits.py
  preprocessing/          frames.py · faces.py (YuNet) · audio.py · run.py
  datasets/               PyTorch datasets (visual / audio / multimodal)
  models/                 visual.py · audio.py · fusion.py
  evaluation/metrics.py   AUC, EER, accuracy, F1
  robustness/             degradations (Phase 5)
  explainability/         Grad-CAM (Phase 6)
  training/               training loops (Phases 2–4)
app/app.py                Gradio demo (Phase 7)
notebooks/                Colab runners
tests/                    pytest suite
```

## Status
- [x] Phase 0: project setup
- [x] Phase 1: dataset download and preprocessing pipeline (code done; waiting on dataset access)
- [ ] Phase 2: visual branch
- [ ] Phase 3: audio branch
- [ ] Phase 4: multimodal fusion
- [ ] Phase 5: robustness evaluation
- [ ] Phase 6: explainability
- [ ] Phase 7: demo app
- [ ] Phase 8: report
