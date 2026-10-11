# Task Tracker

Status legend: ✅ done · 🔄 in progress · ⏳ blocked / waiting · ⬜ todo
Owner: **You** = needs your account / action · **Code** = implementation in this repo

Last updated: 2026-10-11

---

## Now / next up

| # | Task | Owner | Status |
|---|---|---|---|
| D1 | Submit FakeAVCeleb request form | You | ⏳ |
| D2 | Kaggle account + accept DFDC rules + `kaggle.json` | You | ⏳ |
| D3 | (Optional) Submit Celeb-DF v2 request form | You | ⏳ |
| R1 | Run notebooks 00 → 01 → 02 in Colab on DFDC + ASVspoof | You | ⬜ |
| R2 | Run notebook 03: first visual baseline on DFDC | You | ⬜ |
| R3 | Run notebook 04 step 1: audio pre-training on ASVspoof | You | ⬜ (no approval needed) |
| P6 | Phase 6: explainability (Grad-CAM) | Code | ⬜ next |

---

## Phase 0: Setup ✅

| # | Task | Spec | Status |
|---|---|---|---|
| 0.1 | Repo structure + installable package | `src/deepfake/`, `pyproject.toml`, `pip install -e .` | ✅ |
| 0.2 | Requirements | `requirements.txt` (Mac), `requirements-colab.txt` (no torch) | ✅ |
| 0.3 | Config | `configs/default.yaml`; `deepfake.config` resolves Mac / Colab / env-var paths | ✅ |
| 0.4 | Local env | Python 3.11 venv in `.venv` (uv) | ✅ |
| 0.5 | Test suite | pytest with ffmpeg-generated synthetic videos; no dataset needed | ✅ |
| 0.6 | GitHub | public repo `tanishq1101/robust-multimodal-deepfake` | ✅ |
| 0.7 | Colab setup notebook | `00_colab_setup.ipynb`: GPU check, clone, install, tests | ✅ |

## Phase 1: Data pipeline ✅ code · ⏳ data

| # | Task | Spec | Status |
|---|---|---|---|
| 1.1 | Download tools | `deepfake.data.download {status,dfdc,asvspoof,fakeavceleb,celebdf}`; resumable HTTP, gdown for Drive links, safe extraction, verification | ✅ |
| 1.2 | Manifests | `deepfake.data.manifest`: unified CSV (`sample_id, path, modality, label, video_label, audio_label, group, split`) | ✅ |
| 1.3 | Identity-wise splits | `deepfake.data.splits`: 70/15/15 by group; official splits kept | ✅ |
| 1.4 | Frames + faces | 16 uniform frames, YuNet detector (Haar fallback), 1.3× margin, 224², JPEG-in-npz | ✅ |
| 1.5 | Audio | ffmpeg → 16 kHz mono → 128-bin log-mel (25 ms win, 10 ms hop), ≤10 s, float16 | ✅ |
| 1.6 | Batch runner | `deepfake.preprocessing.run`: parallel, resumable, `--per-class` balanced subsets, error log | ✅ |
| 1.7 | PyTorch datasets | `MultimodalDataset` / `VisualDataset` / `AudioDataset`, missing-modality flags | ✅ |
| 1.8 | Notebooks | `01_download_datasets`, `02_preprocess` | ✅ |
| 1.9 | Download DFDC sample + ASVspoof, preprocess | Colab run (R1) | ⏳ needs D2 |
| 1.10 | Download + preprocess FakeAVCeleb | `--per-class 400` | ⏳ needs D1 |
| 1.11 | Download + preprocess Celeb-DF (test list only) | optional | ⏳ needs D3 |

## Phase 2: Visual branch 🔄 (code ✅, training run pending)

| # | Task | Spec | Status |
|---|---|---|---|
| 2.1 | Training engine | `training/engine.py`: auto device (CUDA/MPS/CPU), AMP on CUDA, grad clipping, atomic checkpoints (`last.pt`, `best.pt`), resume, early stopping on val AUC, `history.csv` | ✅ |
| 2.2 | Task registry | `training/tasks.py`: model + dataset + forward fn + label key per branch | ✅ |
| 2.3 | Visual training CLI | `training/train_visual.py` (thin wrapper over the shared `training/runner.py`): AdamW (head LR 10×), cosine LR, frozen backbone for epoch 1, class-balanced BCE (`pos_weight = n_real/n_fake`), 8 random frames/video in training, all 16 in eval, flip + brightness/contrast augmentation | ✅ |
| 2.4 | Evaluation | `evaluation/evaluate.py`: `metrics.json` (AUC, EER, ACC, P/R/F1, confusion matrix, metrics at EER threshold), `predictions.csv`, `roc.png`; works cross-dataset | ✅ |
| 2.5 | Notebook | `03_train_visual.ipynb`: train, curves, test metrics, ROC, score histogram, cross-dataset | ✅ |
| 2.6 | Tests | train → resume → evaluate smoke test (CPU and MPS) | ✅ |
| 2.7 | Baseline run on DFDC sample | target val AUC ≥ 0.80 (small dataset) | ⬜ R2 |
| 2.8 | Main run on FakeAVCeleb | target test AUC ≥ 0.85; record in `reports/results.md` | ⏳ needs 1.10 |
| 2.9 | Cross-dataset eval on Celeb-DF | report the AUC drop | ⏳ needs 1.11 |
| 2.10 | Optional tuning | try `--frames-per-video 16`, `efficientnet_b2`, more epochs, LR 3e-4 | ⬜ |

## Phase 3: Audio branch 🔄 (code ✅, training runs pending)

| # | Task | Spec | Status |
|---|---|---|---|
| 3.1 | `audio` task in registry | `AudioModel` (ResNet-18, spectrogram repeated to 3 channels for ImageNet weights), label key `audio_label`; rows without audio or with `audio_label = -1` (DFDC) dropped | ✅ |
| 3.2 | Shared runner + `train_audio.py` | `training/runner.py` now drives both branches: same CLI, optimizer, freeze schedule, curves, test eval. Audio defaults: dataset `asvspoof`, batch 32. Random 4 s window in training, centre window in eval | ✅ |
| 3.3 | `--init-from` | load weights from any checkpoint (optimizer not loaded; ignored when resuming) | ✅ |
| 3.4 | SpecAugment | 2 frequency masks ≤16 bins + 2 time masks ≤40 frames, filled with the mean; training only | ✅ |
| 3.5 | Per-category evaluation | manifests now have a `category` column; `test/per_category.csv` gives n, mean score and accuracy per category, plus AUC/EER of each fake category vs all reals (= EER per ASVspoof attack A07–A19, per FakeAVCeleb category) | ✅ |
| 3.6 | Notebook + tests | `04_train_audio.ipynb` (pre-train, fine-tune, pre-training ablation); tests for audio training, init-from, label filtering, SpecAugment, per-category | ✅ |
| 3.7 | Pre-train on ASVspoof 2019 LA | target eval EER ≤ 10%; record per-attack EER | ⬜ R3 |
| 3.8 | Fine-tune on FakeAVCeleb audio | `--init-from audio_asvspoof/best.pt --lr 3e-5 --freeze-epochs 0`; report AUC/EER | ⏳ needs 1.10 |
| 3.9 | Ablation: with vs. without ASVspoof pre-training | table in `reports/results.md` | ⏳ needs 1.10 |
| 3.10 | (Stretch) frozen wav2vec2-base features + linear head | comparison row | ⬜ |

## Phase 4: Multimodal fusion 🔄 (code ✅, training runs pending)

| # | Task | Spec | Status |
|---|---|---|---|
| 4.1 | `fusion` task + `train_fusion.py` | `MultimodalModel` on the overall `label`; branches initialised from `--visual-ckpt` / `--audio-ckpt` (backbone names read from those checkpoints); reuses `training/runner.py` (task-specific param groups + frozen params) | ✅ |
| 4.2 | Late fusion | no training; grid over rules `weighted` (visual weight 0…1, step 0.05), `max`, `noisy_or`; best val AUC chosen; `calibration.csv` saved. Absent modality excluded; none present → 0.5 | ✅ |
| 4.3 | Intermediate fusion | `[visual emb 1280 \| audio emb 512 \| 2 presence flags]` → Linear 256 → ReLU → Dropout → Linear 1; branches frozen for 2 epochs (default), then joint fine-tuning (branches at `--lr`, MLP at 10×) | ✅ |
| 4.4 | Modality dropout | `--modality-dropout 0.2`: for samples with both modalities, hide one at random (training only); never removes a sample's only modality | ✅ |
| 4.5 | Fair comparison tool | `evaluation/compare.py`: every model scored on the same full test set against the overall label; `comparison.csv/.md/.png` + per-category accuracy pivot | ✅ |
| 4.6 | Notebook + tests | `05_train_fusion.ipynb`; tests: late rules, modality dropout, frozen-branch init, calibration, comparison | ✅ |
| 4.7 | Run late + intermediate fusion on FakeAVCeleb | targets: fusion beats the best single branch in AUC and is good on all 4 categories | ⏳ needs 2.8 + 3.8 |
| 4.8 | Record comparison in `reports/results.md` | copy `comparison.md` | ⏳ needs 4.7 |
| 4.9 | (Optional) attention / gated fusion variant | extra comparison row | ⬜ |

## Phase 5: Robustness 🔄 (code ✅, runs pending)

| # | Task | Spec | Status |
|---|---|---|---|
| 5.1 | Degradations | `robustness/degradations.py`. Video (on the face clip): JPEG q 90/50/20, H.264 CRF 23/30/40 (libx264 round trip), blur σ 1/2/3, Gaussian noise std 5/15/30, downscale ×2/3/4. Audio (on the waveform): white noise SNR 20/10/5 dB, MP3 64k/32k/16k (libmp3lame). `Degradation(name, level)` and `RandomDegradation` are picklable | ✅ |
| 5.2 | Waveform storage | preprocessing also writes `wav/<id>.flac` (16 kHz, 16-bit, ≤10 s); re-running on old data only adds the missing files; `wav_path` column in the index | ✅ |
| 5.3 | Dataset hooks | `clip_transform` (whole face clip) and `wav_transform` (waveform → degraded → log-mel recomputed) in `MultimodalDataset` | ✅ |
| 5.4 | Stress-test suite | `evaluation/robustness_eval.py`: clean + each applicable degradation × severity per model; `--label native/overall`; `--limit` subset; same random noise for every model; outputs `robustness.csv/.md/.png` with AUC drop vs clean | ✅ |
| 5.5 | Missing-modality test | `video_only` / `audio_only` conditions for fusion models | ✅ |
| 5.6 | Robust training | `--robust-augment` (+ `--robust-p`, default 0.5) for all three training scripts | ✅ |
| 5.7 | Notebook + tests | `06_robustness.ipynb` (stress test → robust fine-tuning → standard vs robust); 13 new tests (severity monotonicity, SNR accuracy, MP3 round trip, hooks, robust training, suite) | ✅ |
| 5.8 | Run stress test on standard models | `--label overall`, `--limit 300` | ⏳ needs 4.7 |
| 5.9 | Robust fine-tuning + stress test again | target: lower mean AUC drop with ≤ 0.01 clean-AUC loss | ⏳ needs 5.8 |
| 5.10 | Cross-dataset test on Celeb-DF | `evaluation.evaluate --dataset celebdf` (standard vs robust visual model) | ⏳ needs 1.11 |
| 5.11 | Figures into `reports/figures/` + numbers into `reports/results.md` | | ⏳ |

## Phase 6: Explainability ⬜

| # | Task | Spec | Status |
|---|---|---|---|
| 6.1 | Visual Grad-CAM | target `visual.backbone.conv_head`; overlay on face crops | ⬜ |
| 6.2 | Audio Grad-CAM | target `audio.backbone.layer4`; overlay on spectrogram | ⬜ |
| 6.3 | Gallery | correct and incorrect examples per category | ⬜ |

## Phase 7: Demo app ⬜

| # | Task | Spec | Status |
|---|---|---|---|
| 7.1 | Inference pipeline | single video → faces + mel → model → scores | ⬜ |
| 7.2 | Gradio UI | verdict, visual/audio scores, heatmaps; Colab share link + local | ⬜ |
| 7.3 | Screen recording for the review | | ⬜ |

## Phase 8: Report & presentation ⬜

| # | Task | Spec | Status |
|---|---|---|---|
| 8.1 | `reports/results.md` | all tables + figures (template created; fill in as runs finish) | 🔄 |
| 8.2 | Final PPT | you'll share the format | ⬜ |
| 8.3 | README polish | results summary, demo GIF | ⬜ |

---

## Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-11 | YuNet (OpenCV) instead of MediaPipe for face detection | Newer MediaPipe dropped the legacy API; YuNet is tiny and needs no extra dependency |
| 2026-10-11 | Raw data on the Colab VM disk; processed data on Drive | Free Drive is 15 GB; ASVspoof alone is 7.6 GB zipped |
| 2026-10-11 | Faces stored as JPEG (q95) inside one `.npz` per video | ~10× smaller than raw arrays; far fewer files on Drive |
| 2026-10-11 | Visual branch trained on `video_label` | e.g. RealVideo-FakeAudio has a real face; the visual branch must not be blamed for fake audio |
| 2026-10-11 | Primary metric AUC; early stopping on val AUC | Threshold-free, robust to the heavy class imbalance |
| 2026-10-11 | One shared training runner for all branches | Identical CLI/behaviour for visual and audio, less code to maintain |
| 2026-10-11 | Audio branch trained on `audio_label`; DFDC excluded from it | DFDC does not label audio fakes separately |
| 2026-10-11 | Audio: ASVspoof pre-training → FakeAVCeleb fine-tuning | ASVspoof is much larger and speech-focused; ablation planned to verify the benefit |
| 2026-10-11 | Late fusion offers max / noisy-OR, not only a weighted average | The overall label is an OR of face-fake and voice-fake; averaging dilutes a single confident fake branch |
| 2026-10-11 | Comparison scores all models against the overall label on the same samples | Fair table: single-branch models are penalised for the fake type they cannot see |
| 2026-10-11 | Video degradations applied to stored face crops, not the full original frames | No raw video needed after preprocessing and much cheaper; documented as an approximation of platform re-compression |
| 2026-10-11 | Store a 16 kHz FLAC waveform per clip | MP3 / additive noise must act on the waveform, not the spectrogram; ~32 KB/s, acceptable on Drive |
