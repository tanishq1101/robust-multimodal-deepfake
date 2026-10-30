# Datasets

| Dataset | Role in the project | Size | Access | Status |
|---|---|---|---|---|
| **FakeAVCeleb v1.2** | **Main multimodal dataset.** Separate labels for fake video and fake audio | 500 real + 19,500 fake videos | Request form | ⏳ submit form |
| **DFDC train sample** | Pipeline development while FakeAVCeleb approval is pending | ~400 videos with audio (~4 GB download) | Kaggle account + rules | ⏳ needs your Kaggle token |
| **ASVspoof 2019 LA** | Audio branch pre-training (bona fide vs. synthetic/converted speech) | 7.6 GB zip | Free | ✅ link verified, download in Colab |
| **Celeb-DF v2** | Cross-dataset test of the visual branch | 590 real + 5,639 fake videos | Request form | ⏳ submit form (optional) |

All downloads run in **Colab** (`notebooks/deepfake_pipeline.ipynb`, §3). Your Mac only needs the code.

---

## 1. FakeAVCeleb (do this first: approval can take days)

1. Fill in the Google request form linked from the official repo:
   https://github.com/DASH-Lab/FakeAVCeleb
   (direct form: https://docs.google.com/forms/d/e/1FAIpQLSfPDd3oV0auqmmWEgCSaTEQ6CGpFeB-ozQJ35x-B_0Xjd93bw/viewform)
   Use your college email and mention it's for an academic project.
2. When approved, they email a download link or a `download.sh` script.
3. In the notebook settings cell (§2), paste the link into `FAKEAVCELEB_URL`.

Folder layout after extraction (the code finds it automatically):
```
FakeAVCeleb_v1.2/
├── RealVideo-RealAudio/<ethnicity>/<gender>/idXXXXX/*.mp4   label 0
├── RealVideo-FakeAudio/...                                    label 1 (audio fake)
├── FakeVideo-RealAudio/...                                    label 1 (video fake)
└── FakeVideo-FakeAudio/...                                    label 1 (both fake)
```

## 2. DFDC train sample (Kaggle)

1. Create a free account at https://www.kaggle.com
2. Accept the competition rules: https://www.kaggle.com/competitions/deepfake-detection-challenge/rules
3. Kaggle → **Settings** → **API** → **Create New Token**. This downloads `kaggle.json`.
   Keep it private and never commit it (`.gitignore` already excludes it).
4. Run §3 of the notebook and upload `kaggle.json` when asked.

The script downloads the default competition archive and keeps only `train_sample_videos/` with `metadata.json` (labels REAL/FAKE, plus which real video each fake was made from).

## 3. ASVspoof 2019 LA

No action needed. Notebook 01 downloads `LA.zip` from Edinburgh DataShare
(https://datashare.ed.ac.uk/handle/10283/3336), resuming if Colab disconnects.
Official splits are kept: train → train, dev → val, eval → test. The eval split must never be used for training.

## 4. Celeb-DF v2 (optional, for cross-dataset testing)

1. Request access via the form linked at https://github.com/yuezunli/celeb-deepfakeforensics
2. Paste the link into `CELEBDF_URL` in the notebook settings cell (§2).

---

## Where data lives

| What | Location | Why |
|---|---|---|
| Raw datasets | Colab VM disk `/content/raw/` | Too big for free 15 GB Drive; wiped when the session ends |
| Manifests, processed faces / spectrograms / 16 kHz FLAC audio, checkpoints | `MyDrive/deepfake_project/` | Small, persistent |

So run notebook **§3 then §4 in the same Colab session**. After preprocessing, the raw data is no longer needed.
Override the locations with the `DEEPFAKE_RAW_ROOT` / `DEEPFAKE_DATA_ROOT` environment variables.

## Leakage-free splits

Splits use FakeAVCeleb person IDs, DFDC source-video groups, and ASVspoof's official speaker
protocols. The splitter keeps each group in one active split, and training rejects duplicate
samples or groups/media paths that cross splits. DFDC source-video grouping does **not** prove
that every person's identity is disjoint: different source videos may show the same person.

The Celeb-DF official test list is preserved exactly. Other videos with those test identities
receive `split=excluded` so they cannot enter training or validation. No files are deleted.
Check the resulting class counts before training; the training commands require both classes
in train and validation.

## Citations

- Khalid et al., *FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset*, NeurIPS Datasets & Benchmarks 2021.
- Dolhansky et al., *The DeepFake Detection Challenge (DFDC) Dataset*, arXiv:2006.07397, 2020.
- Wang et al., *ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech*, Computer Speech & Language, 2020.
- Li et al., *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics*, CVPR 2020.
