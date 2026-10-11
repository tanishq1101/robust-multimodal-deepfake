# Voice model: `audio_asvspoof_3a53cf51`

ResNet-18 (ImageNet initialisation) that scores 4-second log-mel spectrograms as bona fide or
spoofed speech. It was trained on a Google Colab T4 on 11 October 2026 with
`notebooks/10_reviewed_colab_training.ipynb`.

## Weights

`best.pt` (134,258,271 bytes) is attached to the GitHub Release
[`model-audio-asvspoof-v1`](https://github.com/tanishq1101/robust-multimodal-deepfake/releases/tag/model-audio-asvspoof-v1).
```bash
python scripts/download_model.py audio_asvspoof_3a53cf51
```
SHA-256: `bdd5f5ec458936669161459fb9c9b084a069caaa1c226d8e5d31b688c7f4acc8`

## Training data

ASVspoof 2019 Logical Access, balanced subset: up to 2,500 clips per class from each official
partition, seed 42 (`data.json`).

| Split | Official partition | Clips | Spoof | Bona fide |
|---|---|---:|---:|---:|
| train | train | 5,000 | 2,500 | 2,500 |
| val | development | 5,000 | 2,500 | 2,500 |
| test | evaluation | 5,000 | 2,500 | 2,500 |

The evaluation partition contains attack systems A07–A19. Except for A16 and A19, which reuse
training algorithms A04 and A06, they are absent from training.

## Training

```bash
python -m deepfake.training.train_audio --dataset asvspoof --run-name audio_asvspoof_3a53cf51 \
    --epochs 15 --batch-size 32 --lr 0.0001 --weight-decay 0.0001 --dropout 0.3 \
    --freeze-epochs 1 --patience 3 --seed 42 --device cuda
```
Up to 15 epochs with early-stopping patience 3, SpecAugment on, and the backbone frozen for epoch 1.
The run stopped after epoch 6. The epoch-3 checkpoint (best validation AUC) was evaluated once on
the test split. Validation AUC reached 1.0 by epoch 3 because the development partition shares
its attack systems with training (`history.csv`, `curves.png`).

## Test results (`test/metrics.json`)

| Metric | Value |
|---|---:|
| AUC | 0.9771 |
| EER | 6.20% |
| Accuracy / precision / recall / F1 at threshold 0.5 | 91.24% / 0.991 / 0.832 / 0.905 |
| Accuracy at the EER threshold (0.0102) | 93.80% |
| Confusion matrix [[TN, FP], [FN, TP]] | [[2481, 19], [419, 2081]] |

Per attack system: `test/per_category.csv`. A17 is the hardest attack (1.2% detected, AUC 0.725,
EER 34.4%). Every test prediction: `test/predictions.csv` (ASVspoof clip ID, attack, label, score).
ROC: `test/roc.png`. Full discussion: [`reports/report.md`](../../reports/report.md) §6.2 and §7.

## Files

| File | Contents |
|---|---|
| `config.json` | full training configuration, including the audio preprocessing settings |
| `data.json` | dataset and split sizes |
| `history.csv`, `curves.png` | loss and validation metrics per epoch |
| `test/metrics.json` | test metrics, confusion matrix, EER threshold |
| `test/per_category.csv` | results per ASVspoof attack system and for bona fide speech |
| `test/predictions.csv` | score for each of the 5,000 test clips |
| `test/roc.png` | ROC curve on the test split |

## Limitations

- Subset results (5,000 clips per split), not full-benchmark ASVspoof scores.
- The model was trained on studio-quality speech. Other microphones, rooms or compression are
  untested.
- Threshold 0.5 under-detects several unseen attacks. Calibrate a threshold before relying on
  verdicts.

ASVspoof 2019 is distributed under the Open Data Commons Attribution License. Cite: Wang, X.,
Yamagishi, J., Todisco, M., et al. *ASVspoof 2019: A large-scale public database of synthesized,
converted and replayed speech.* Computer Speech & Language 64, 2020.
