# Deepfake pipeline review — 11 October 2026

## Finding

This is a **PyTorch training and inference pipeline**, not a verified trained detector yet.
The configured local data root does not exist, no trained checkpoints are present there,
`reports/results.md` contains no completed runs, and all nine supplied notebooks have zero
saved code-cell outputs. Those observations cannot establish whether a separate Colab run
exists or how well a model detects real deepfakes.

No project files or existing results were deleted. The changes retain the visual, audio,
fusion, robustness, explainability, demo, and reporting workflows.

## Problems fixed

| Problem | Correction |
|---|---|
| `.gitignore` hid `src/deepfake/data`, leaving the downloader, manifest builder, and splitter out of Git | Restricted generated-data exclusions to the repository root. The existing source files now appear in the change list and must be included when publishing the fixes. |
| The training dataset treated centre crops from clips with no detected face as visual evidence, while inference treated those clips as missing video | Persisted whether preprocessing used face detection; aligned dataset evidence flags with inference. Explicit `--no-detector` experiments remain supported. |
| Missing required evidence still produced a REAL/FAKE verdict | Return `INCONCLUSIVE`, a null score, and a warning; the demo handles that result. |
| Frozen backbones still updated BatchNorm buffers and used dropout | Keep fully frozen modules in evaluation mode during warm-up. Tests compare every branch state tensor before and after frozen fusion training. |
| Padded frames and absent modalities entered the CNNs and affected training statistics | Encode only valid frames and available modalities. Reconstruct Grad-CAM maps at their original frame positions. |
| Weighted late fusion returned zero when its sole available branch had weight 0 or 1 | Use the available branch's score at either endpoint. |
| Resuming could silently apply changed data/model/optimizer settings and overwrite the recorded config | Check compatibility before loading state or rewriting config; preserve supported random-number states. |
| Extending a completed cosine schedule could run additional epochs at zero learning rate | Update the schedule horizon and restore the learning rate for the extended run. Regression checks verify weights actually change after resuming. |
| Partially assigned official splits could leak related samples into training | Preserve official assignments and exclude unassigned relatives of official test identities. The exact official test list remains unchanged. |
| Training accepted overlapping groups/media, duplicate samples, or single-class validation | Validate splits and labels before building a model; late-fusion calibration also requires both classes. |
| Evaluation/inference depended on the current config rather than the trained preprocessing settings | Save video/audio settings in new checkpoints and reuse them in evaluation, robustness, galleries, and raw-file inference. Older checkpoints retain the existing config fallback. |
| Grad-CAM could fail for frozen parameters or explain a different class from a custom-threshold verdict | Enable input gradients and pass the actual predicted class to the explanation. |
| Archive checks allowed sibling-prefix traversal and unsafe tar links | Check path ancestry and reject links/special tar members. Preprocessing also rejects duplicate or unsafe sample IDs. |
| Fusion/gallery notebooks failed unclearly when checkpoints were missing | Add explicit prerequisite errors in the generator and regenerate the affected notebooks. |
| Documentation described real training as completed without results | Identify the real-data training and GPU evaluation as unverified. |

## Notebook and training review

All nine notebooks pass notebook-schema validation, and the regular Python portions of
every code cell compile. Shell-magic commands were inspected, not executed in a Colab kernel.
All nine files match `scripts/make_notebooks.py` exactly.

The training flow remains:

1. `01_download_datasets` and `02_preprocess` prepare raw data, splits, face crops, log-mel
   arrays, and waveforms.
2. `03_train_visual` trains the EfficientNet visual branch using PyTorch and `video_label`.
3. `04_train_audio` pre-trains the ResNet audio branch on ASVspoof, then fine-tunes on
   FakeAVCeleb using `audio_label`.
4. `05_train_fusion` calibrates late fusion on validation data or trains the intermediate
   MLP using the overall label. Late fusion is calibration, not gradient training.
5. `06_robustness` stress-tests and fine-tunes the models; `07_explainability` and
   `08_demo_and_report` consume saved checkpoints.

PyTorch provides DataLoaders, binary cross-entropy loss with class balancing, AdamW,
gradient clipping, the cosine schedule, CUDA mixed precision, and `.pt` checkpoints.
Checkpoint selection uses validation AUC; test data is evaluated afterward. Existing tests
exercise actual synthetic visual/audio/fusion training, checkpoint reload, resume,
fine-tuning, inference, explanations, robustness, the Gradio interface, and report output.

## Verification limits and remaining work

Verified after the fixes:

- `.venv/bin/python -m pytest -q`: **74 passed in 222.49 seconds**, no failures or skips.
- Python source compilation: passed for `src`, `app`, and `scripts`.
- Notebook schema validation and regular Python-cell compilation: passed for all nine files.
- Notebook-generator consistency: all nine generated files match.
- Training CLI help and dataset-status commands: passed; all four datasets report missing.
- `git diff --check`: passed.

- The runtime used for this review supports CPU. CUDA/MPS training, mixed precision on a
  GPU, and execution inside Colab were not verified.
- Synthetic patterns and tones verify software behavior, not deepfake detection accuracy.
  No real-data accuracy, AUC, EER, generalization, or robustness claim is justified yet.
- DFDC groups identify source videos, not necessarily every person's identity. The code
  checks recorded groups and paths; donor identities or duplicated content absent from
  metadata require a dataset audit.
- RNG state restoration does not establish bitwise reproducibility with persistent
  DataLoader workers, augmentation-library RNGs, or nondeterministic GPU kernels.
- Old processed data created with `--no-detector` has no explicit detector-mode metadata;
  re-run preprocessing with that same flag before relying on its evidence flags.
- Automatic checkpoint discovery selects by naming preference, not measured validation
  quality. Select an explicit checkpoint for a final evaluation or demonstration.
- To certify training, run the updated notebooks against the real datasets and inspect
  `history.csv`, `best.pt`, validation/test metrics, class counts, per-category results,
  and held-out robustness/cross-dataset evaluations. More epochs alone do not prove a
  better model. Probability calibration and deployment thresholds need separate validation.

The updated code is local; it has not been pushed to the repository cloned by Colab.
