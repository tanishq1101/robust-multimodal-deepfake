# Robust Multimodal Deepfake Analysis: Project Plan

**Type:** College project · **Compute:** Google Colab (GPU) for training, Mac for coding
**Status:** Phase 0 ✅ · Phase 1 code ✅ (waiting on dataset access) · Phase 2 code ✅ · Phase 3 code ✅ (training runs pending) · next: Phase 4
**Task tracker:** [TASKS.md](TASKS.md)

---

## 1. Goal

Build a system that takes a video and decides **real vs. fake**:

- It analyses **both the face (video) and the voice (audio)**.
- It stays reliable when the video is **compressed, noisy or degraded**.
- It **explains its decision** with heatmaps.

It will be delivered as a working web demo.

### Success criteria
| Item | Target |
|---|---|
| Working pipeline | video in → real/fake score + heatmap out |
| Visual branch AUC (in-dataset) | ≥ 0.85 |
| Audio branch EER (in-dataset) | ≤ 10% |
| Fusion | beats the better single branch on the multimodal test set |
| Robustness | accuracy-drop report under compression/noise, with and without robust training |
| Demo | Gradio app: upload video → verdict, per-modality scores, Grad-CAM |

---

## 2. Architecture

```
                         Input video (.mp4)
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
     PREPROCESSING (visual)            PREPROCESSING (audio)
     16 frames sampled uniformly       ffmpeg → 16 kHz mono
     YuNet face detection              log-mel spectrogram
     224×224 crops (JPEG in .npz)      128 mels, 10 ms hop, 4 s window
                │                                │
                ▼                                ▼
     VISUAL BRANCH                     AUDIO BRANCH
     EfficientNet-B0 (ImageNet)        ResNet-18 (ImageNet) on spectrogram
     per-frame → masked average        → 512-d embedding + score
     → 1280-d embedding + score
                │                                │
                └───────────────┬────────────────┘
                                ▼
                         FUSION MODULE
            late:         weighted average of branch probabilities
            intermediate: [visual emb | audio emb | presence flags] → MLP
                                ▼
            Output: P(fake) overall + visual score + audio score
                                ▼
            EXPLAINABILITY: Grad-CAM on face crops and spectrogram
```

**Robustness** is handled in three ways:
- **Training-time augmentation:** compression, blur, noise, downscaling, audio noise and MP3 compression.
- **Test-time stress suite:** the same degradations at three severity levels.
- **Missing-modality handling:** clips with no face or no audio are handled through the presence flags.

---

## 3. Datasets (details in [DATASETS.md](DATASETS.md))

| Dataset | Role | Access |
|---|---|---|
| **FakeAVCeleb v1.2** | Main multimodal train/test (4 categories: RV-RA, RV-FA, FV-RA, FV-FA) | Request form |
| **DFDC train sample** | Pipeline development while waiting for approvals | Kaggle token |
| **ASVspoof 2019 LA** | Audio branch pre-training | Free (link verified) |
| **Celeb-DF v2** | Cross-dataset visual test | Request form |

- **Splits:** by identity, never by file. Official splits are kept where they exist (ASVspoof, Celeb-DF test list).
- **Storage:** raw data goes on the Colab VM disk. Processed data and checkpoints go on Google Drive.

---

## 4. Tech stack

| Area | Tools |
|---|---|
| Language / env | Python 3.11, `uv` venv on Mac; Colab for GPU |
| Deep learning | PyTorch, torchvision, `timm` |
| Video / faces | OpenCV (YuNet face detector, Haar fallback), ffmpeg |
| Audio | librosa, soundfile, ffmpeg |
| Augmentation | albumentations, audiomentations, ffmpeg re-encoding |
| Explainability | `pytorch-grad-cam` |
| Metrics | scikit-learn (AUC, EER, ACC, F1) |
| Demo | Gradio |
| Data access | requests (resumable), kaggle CLI, gdown |
| Testing | pytest (synthetic video generated with ffmpeg) |

### Workflow
```
Mac (VS Code) ──git push──► GitHub ──git clone──► Colab notebook
                                                   │
                     Google Drive ◄────────────────┘  manifests, processed data, checkpoints
                     Colab VM disk                     raw datasets (temporary)
```

---

## 5. Repository structure

```
DeepFake/
├── README.md  PLAN.md  TASKS.md  DATASETS.md
├── requirements.txt  requirements-colab.txt  pyproject.toml
├── configs/default.yaml
├── src/deepfake/
│   ├── config.py
│   ├── data/            download.py · manifest.py · splits.py
│   ├── preprocessing/   frames.py · faces.py · audio.py · run.py
│   ├── datasets/        multimodal.py · visual.py · audio.py
│   ├── models/          visual.py · audio.py · fusion.py
│   ├── evaluation/      metrics.py · evaluate.py · plots.py   (+ robustness_eval.py later)
│   ├── robustness/      degradations.py  (Phase 5)
│   ├── explainability/  gradcam.py       (Phase 6)
│   └── training/        engine.py · tasks.py · runner.py · train_visual.py · train_audio.py   (+ fusion)
├── app/app.py           (Phase 7)
├── notebooks/           00_colab_setup · 01_download_datasets · 02_preprocess · 03_train_visual · 04_train_audio
├── scripts/             make_notebooks.py
├── tests/               test_metrics · test_preprocessing · test_models
└── reports/             figures + results (later)
```

---

## 6. Phases

### Phase 0: Setup ✅
- Repo structure, requirements, config, README, `.gitignore`, installable package.
- Local Python 3.11 venv; pytest suite.
- Colab setup notebook.

### Phase 1: Data pipeline ✅ code / ⏳ data
- Download tools for all 4 datasets (resumable, verified).
- Unified manifests, identity-wise splits.
- Frame sampling, YuNet face crops, log-mel spectrograms; resumable, parallel batch runner.
- PyTorch datasets returning `frames, frame_mask, mel, has_video, has_audio, labels`.
- **Remaining (you):** submit the FakeAVCeleb and Celeb-DF forms, create a Kaggle token, run notebooks 01 and 02 in Colab.

### Phase 2: Visual branch ✅ code / ⬜ training run
- Shared training engine (`training/engine.py`): auto device, AMP, checkpoint every epoch, resume, early stopping on val AUC.
- `training/train_visual.py`: EfficientNet-B0 with a frozen-backbone warm-up, class-balanced BCE and 8 random frames per video in training.
- `evaluation/evaluate.py`: test metrics, predictions, ROC; also cross-dataset.
- Notebook `03_train_visual.ipynb`.
- **Remaining:** run on DFDC (baseline), then FakeAVCeleb (main), then cross-dataset on Celeb-DF.
- **Deliverable:** checkpoint, AUC/ACC/F1, training curves.

### Phase 3: Audio branch ✅ code / ⬜ training runs
- `training/train_audio.py` on the shared `training/runner.py`: ResNet-18 on log-mel, SpecAugment, random 4 s windows.
- Pre-train on ASVspoof 2019 LA, then fine-tune on FakeAVCeleb audio with `--init-from`.
- Per-category evaluation: EER per ASVspoof attack (A07–A19) and per FakeAVCeleb category.
- Notebook `04_train_audio.ipynb`, including an ablation with and without pre-training.
- (Stretch) frozen wav2vec2 features as a comparison.
- **Deliverable:** checkpoint, EER/AUC, per-attack table, curves.

### Phase 4: Multimodal fusion
- Late fusion (weight tuned on val) vs. intermediate fusion (branches initialised from Phases 2–3).
- Results per FakeAVCeleb category (which fakes each branch catches).
- **Deliverable:** comparison table. This is the core result.

### Phase 5: Robustness
- Implement the degradations in `robustness/degradations.py`.
- Stress-test the baseline models, retrain with augmentation, then stress-test again.
- Run the missing-modality test.
- Run a cross-dataset test on Celeb-DF.
- **Deliverable:** robustness tables and "metric vs. severity" plots.

### Phase 6: Explainability
- Grad-CAM on face crops and spectrograms.
- Gallery of correct and incorrect predictions.

### Phase 7: Demo app
- Gradio app: upload a video → verdict, per-modality scores, heatmaps.
- Runs in Colab (share link) and locally.

### Phase 8: Report & presentation
- Results write-up, figures, final PPT, clean README.

---

## 7. Evaluation protocol
- **Metrics:** AUC (primary), EER (audio), accuracy, precision, recall, F1, confusion matrix.
- **Levels:** frame-level and video-level.
- **Experiments:**
  1. Single-modality baselines.
  2. Fusion variants.
  3. Robustness with and without augmentation.
  4. Missing modality.
  5. Cross-dataset.
- **Reproducibility:** fixed seeds; every result is reproducible from the notebooks.

---

## 8. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Dataset approval delayed | DFDC + ASVspoof need no approval; the pipeline is built on them first |
| Colab disconnects | Resumable downloads, preprocessing skips finished files, checkpoints every epoch |
| Drive space (15 GB free) | Raw data on VM disk; faces stored as JPEG-in-npz; balanced subsets (`--per-class`) |
| FakeAVCeleb imbalance (500 real vs 19,500 fake) | Balanced subsets per category; AUC as the main metric |
| Overfitting | Pretrained backbones, augmentation, identity-wise splits, early stopping |
| Face detection failures | Previous box reused, centre-crop fallback, detection rate logged |
