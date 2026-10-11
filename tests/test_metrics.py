import numpy as np

from deepfake.evaluation.metrics import compute_metrics, eer


def test_perfect_separation():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auc"] == 1.0
    assert m["eer"] == 0.0
    assert m["accuracy"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_inverted_scores():
    m = compute_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert m["auc"] == 0.0
    assert m["eer"] == 1.0


def test_eer_random_is_about_half():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 20000)
    scores = rng.random(20000)
    assert abs(eer(labels, scores)[0] - 0.5) < 0.02


def test_unknown_labels_ignored():
    m = compute_metrics([0, 1, -1], [0.1, 0.9, 0.5])
    assert m["n"] == 2 and m["auc"] == 1.0
