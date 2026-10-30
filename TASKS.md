# Task Tracker

Status legend: ✅ done · 🔄 in progress · ⏳ blocked / waiting · ⬜ todo
Owner: **You** = needs your account / action · **Code** = implementation in this repo

Last updated: 2026-10-29

---

## Now / next up

| # | Task | Owner | Status |
|---|---|---|---|
| D1 | Submit FakeAVCeleb request form | You | ⏳ |
| D2 | Kaggle account + accept DFDC rules + `kaggle.json` | You | ⏳ |
| D3 | (Optional) Submit Celeb-DF v2 request form | You | ⏳ |
| R1 | Run `deepfake_pipeline.ipynb` §1–4 in Colab on DFDC + ASVspoof | You | 🔄 ASVspoof done (2026-10-11, former notebook 10); DFDC pending |
| R2 | Notebook §5 with `DATASET="dfdc"`: first face baseline | You | ⬜ |
| R3 | Notebook §6 step 1: voice pre-training on ASVspoof | You | ✅ 2026-10-11, run `audio_asvspoof_3a53cf51` |
| R4 | After FakeAVCeleb approval: notebook §3–6 with `DATASET="fakeavceleb"` | You | ⏳ needs D1 |
| R5 | Notebook §7 (fusion) → §8 (robustness) → §9 (Grad-CAM) | You | ⏳ needs R4 |
| R6 | Notebook §11: build the report, copy it into `reports/`, record a demo video | You | ⏳ needs R5 |
| R7 | Voice model that generalises: retrain with more varied genuine speech, silence trimming and codec augmentation; test on In-the-Wild | You + Code | ⬜ see `reports/voice_generalization.md` |
| P8 | Final PPT in your format; fill in report sections 6–8 from `results.md` | Code | ⏳ needs your template + R6 |

All code for Phases 0–8 is implemented and tested. What remains is running it on the real data
(R1–R6) and the final write-up (P8).

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
| 3.7 | Pre-train on ASVspoof 2019 LA | target eval EER ≤ 10%; record per-attack EER | ✅ balanced subset (5,000 clips/split): test AUC 0.977, EER 6.2%; per-attack EER in `reports/results.md` (A17 hardest, EER 34.4%). Checkpoint `checkpoints/audio_asvspoof_3a53cf51/best.pt` |
| 3.8 | Fine-tune on FakeAVCeleb audio | `--init-from audio_asvspoof_3a53cf51/best.pt --lr 3e-5 --freeze-epochs 0`; report AUC/EER | ⏳ needs 1.10 |
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

## Phase 6: Explainability 🔄 (code ✅, runs pending)

| # | Task | Spec | Status |
|---|---|---|---|
| 6.1 | Grad-CAM core | `explainability/gradcam.py`, own hook-based implementation (the fusion model has several inputs). `explain(model, task, batch, target)` works for visual, audio and fusion; target = predicted / fake / real | ✅ |
| 6.2 | Visual Grad-CAM | layer `backbone.bn2` (EfficientNet, 7×7 per frame) → upsampled to 224², one shared scale across frames, padded frames zeroed | ✅ |
| 6.3 | Audio Grad-CAM | layer `backbone.layer4` (ResNet-18, 4×13) → upsampled to 128 × 401 | ✅ |
| 6.4 | Fusion Grad-CAM | one backward pass of the fused logit gives face + spectrogram maps; a missing modality gets no map | ✅ |
| 6.5 | Rendering | `render_explanation`: frames, overlays (JET), spectrogram + CAM, P(fake) and explained class in the title | ✅ |
| 6.6 | Gallery CLI | `explainability/gallery.py`: per category, the most confident correct and wrong examples → PNGs + `gallery.csv` + `index.md` | ✅ |
| 6.7 | Notebook + tests | `07_explainability.ipynb`; 7 tests incl. a faithfulness check (toy model, CAM must cover exactly the evidence region and be empty for the opposite class) | ✅ |
| 6.8 | Run gallery on the final fusion model | pick 4–8 figures for the report | ⏳ needs 4.7 |
| 6.9 | (Optional) sanity checks | model-randomisation test (Adebayo et al., 2018); deletion curve | ⬜ |

## Phase 7: Demo app ✅ code · ⏳ needs a trained model

| # | Task | Spec | Status |
|---|---|---|---|
| 7.1 | Inference pipeline | `deepfake/inference.py`, `DeepfakeDetector`: same preprocessing as training, in memory (16 frames, YuNet crops, centre 4 s of the first 10 s of audio); any checkpoint type; branch scores; Grad-CAM; warnings for no face / no audio; audio files supported for audio/fusion models | ✅ |
| 7.2 | CLI | `python -m deepfake.inference clip.mp4 [--checkpoint] [--out]` → JSON + `explanation.png`; picks the best available checkpoint (robust intermediate fusion > intermediate > late > visual > audio) | ✅ |
| 7.3 | Gradio UI | `deepfake/demo.py` + `app/app.py`: upload, heatmap target, verdict label, score table, face Grad-CAM gallery, spectrogram Grad-CAM; `--share` for Colab; disclaimer | ✅ |
| 7.4 | Tests + live check | 7 tests (fusion/no-face/no-audio/audio-file inference, CLI, checkpoint picker, UI handler); app launched locally and queried end-to-end through the Gradio client | ✅ |
| 7.5 | Demo with the trained model + screen recording for the review | notebook 08 | 🔄 demo runs with the published ASVspoof voice model (2026-10-11); fusion model + screen recording need R5 |
| 7.6 | UI/UX redesign | Tabs: Analyse (video / audio upload / microphone, verdict card with probability gauge, per-branch evidence, input checks, adjustable threshold, Grad-CAM), Model & training (`deepfake/model_card.py`: data splits, settings, test metrics, confusion matrix, per-category table, curves, ROC, history), Code (training command + model/training source), About. Responsive; tested in desktop and mobile widths | ✅ |
| 7.8 | One notebook | the ten Colab notebooks merged into `notebooks/deepfake_pipeline.ipynb` (generated by `scripts/make_notebooks.py`); every stage has a switch and skips with a message when its inputs are missing | ✅ 2026-10-29 |
| 7.9 | Video errors fixed | YuNet download fixed (Git LFS URL + SHA-256) and no crash on OpenCV 5 (no Haar cascades there); Gradio 5/6 keyword compatibility; videos are now judged on the face too (pretrained DFDC face model), not on the voice alone | ✅ 2026-10-29 |
| 7.10 | Image analysis + pretrained face model | `face_model.py` (DFDC-winning EfficientNet-B7, weights-only loading, Grad-CAM); `analyzer.py` + `analyze.py` CLI for video / audio / image; voice scored on 4 s windows over 30 s | ✅ 2026-10-29 |
| 7.11 | Demo samples | `demo_samples/`: 4 videos, 4 audio, 3 images with ground truth (`scripts/make_demo_samples.py`); 9/11 correct, `reports/demo_samples/results.md` | ✅ 2026-10-29 |
| 7.12 | UI redesign v2 | Video / Audio / Image / Samples / Models / Code / About tabs; one-click labelled samples with ground truth; score-over-time chart; video decision rule; mobile layout without horizontal scroll | ✅ 2026-10-29 |
| 7.13 | Voice model out-of-domain check | `scripts/eval_voice_generalization.py`: AUC 0.59 on 95 VITS-paper clips, 0/10 genuine recordings accepted → caution note in the app | ✅ 2026-10-29 |
| 7.7 | Publish trained models | `models/<run>/` in git (config, data, history, metrics, per-category, predictions, figures, model card); weights as GitHub Release assets; `scripts/download_model.py` verifies SHA-256; app finds `models/` automatically | ✅ |

## Phase 8: Report & presentation 🔄 (tooling + drafts ✅, final numbers pending)

| # | Task | Spec | Status |
|---|---|---|---|
| 8.1 | Report builder | `deepfake/report.py`: dataset sizes, all runs (`runs.csv`), per-run metrics / per-category / calibration, comparison, robustness, Grad-CAM gallery, figures copied; "Not run yet" for missing parts (never invents numbers); includes `observations.md` | ✅ |
| 8.2 | `reports/results.md` | generated by the builder; currently all "Not run yet" | ⏳ regenerate after R5 |
| 8.3 | Report draft | `reports/report.md`: abstract, introduction, related work, datasets, method, setup, limitations, 17 references; results / discussion / conclusion marked to fill from `results.md` | ✅ draft |
| 8.4 | Presentation outline | `reports/presentation_outline.md`: 18 slides + 2-minute demo script | ✅ |
| 8.5 | Final PPT | build in your format once shared | ⏳ needs your template |
| 8.6 | README polish | architecture diagram, features, notebook table, demo, layout, status | ✅ (add a demo GIF/screenshot after R6) |
| 8.7 | Notebook | `08_demo_and_report.ipynb`: build + download report, CLI check, launch demo with share link | ✅ |
| 8.8 | Tests | report builder with and without results | ✅ |

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
| 2026-10-11 | Own Grad-CAM implementation instead of `pytorch-grad-cam` | The fusion model has 5 inputs; hooks give face + audio maps from one backward pass of the fused decision |
| 2026-10-11 | Heatmaps explain the predicted class by default | Shows why the model decided what it did; fake / real targets available for contrast |
| 2026-10-11 | `results.md` is generated, not hand-written | Every number is traceable to a saved run; no placeholders filled by hand |
| 2026-10-11 | Demo judges audio on the centre 4 s of the first 10 s | Matches the training / evaluation preprocessing exactly |
| 2026-10-11 | PPT not built yet | You will share the required format; an outline is ready in `reports/presentation_outline.md` |
