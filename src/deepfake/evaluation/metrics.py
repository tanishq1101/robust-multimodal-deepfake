"""Evaluation metrics. Convention: label 1 = fake, score = predicted probability of fake."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and the threshold where false-accept rate == false-reject rate."""
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def compute_metrics(labels, scores, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    keep = labels >= 0  # drop unknown labels (-1)
    labels, scores = labels[keep], scores[keep]
    preds = (scores >= threshold).astype(int)
    out = {
        "n": int(len(labels)),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }
    if len(np.unique(labels)) == 2:
        out["auc"] = roc_auc_score(labels, scores)
        out["eer"], out["eer_threshold"] = eer(labels, scores)
    else:
        out["auc"] = out["eer"] = out["eer_threshold"] = float("nan")
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}
