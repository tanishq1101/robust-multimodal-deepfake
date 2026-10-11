# Results

> Template. Fill in each table as the Colab runs finish. Every number should come from a
> `metrics.json` / `comparison.md` file in `MyDrive/deepfake_project/` (path noted per table).
> Nothing here is filled in yet: no training run has been done.

## Setup

| Item | Value |
|---|---|
| Hardware | Google Colab T4 GPU |
| Frames per video | 16 (8 random per video in training) · 224×224 face crops (YuNet) |
| Audio | 16 kHz · 128-bin log-mel · 4 s window |
| Splits | identity-wise 70/15/15 (official splits for ASVspoof) |
| Seed | 42 |

## Datasets used

| Dataset | Train | Val | Test | Notes |
|---|---|---|---|---|
| DFDC sample | | | | `processed/dfdc/index.csv` |
| ASVspoof 2019 LA | | | | balanced subset, `--per-class 2500` |
| FakeAVCeleb | | | | balanced subset, `--per-class 400` |
| Celeb-DF v2 (test only) | | | | |

## Phase 2: Visual branch (`checkpoints/visual_*/test/metrics.json`)

| Train on | Test on | AUC | EER | Accuracy | F1 |
|---|---|---|---|---|---|
| DFDC | DFDC | | | | |
| FakeAVCeleb | FakeAVCeleb | | | | |
| FakeAVCeleb | Celeb-DF (cross-dataset) | | | | |

## Phase 3: Audio branch (`checkpoints/audio_*/test/`)

| Run | Test on | AUC | EER | Accuracy |
|---|---|---|---|---|
| ASVspoof only | ASVspoof eval | | | |
| ImageNet → FakeAVCeleb (no pre-training) | FakeAVCeleb | | | |
| ASVspoof → FakeAVCeleb | FakeAVCeleb | | | |

EER per ASVspoof attack (`audio_asvspoof/test/per_category.csv`):

| Attack | A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EER | | | | | | | | | | | | | |

## Phase 4: Fusion comparison (`outputs/compare_fakeavceleb_test/comparison.md`)

Overall label = fake if face or voice is fake. All models are scored on the same test set.

| Model | AUC | EER | Accuracy | F1 |
|---|---|---|---|---|
| visual_only | | | | |
| audio_only | | | | |
| late (rule: ___) | | | | |
| intermediate | | | | |

Accuracy per category:

| Category | visual_only | audio_only | late | intermediate |
|---|---|---|---|---|
| RealVideo-RealAudio | | | | |
| RealVideo-FakeAudio | | | | |
| FakeVideo-RealAudio | | | | |
| FakeVideo-FakeAudio | | | | |

## Phase 5: Robustness (todo)

## Phase 6: Explainability (todo)

## Observations

-
