"""Model page of the demo: everything is read from a run folder; missing files are optional."""

import json

import pandas as pd
import pytest

from deepfake import model_card
from deepfake.config import REPO_ROOT
from deepfake.demo import confusion_html, metric_tiles, training_command

PUBLISHED = REPO_ROOT / "models" / "audio_asvspoof_3a53cf51"


def _run(tmp_path, **files):
    run = tmp_path / "audio_x"
    (run / "test").mkdir(parents=True)
    (run / "best.pt").touch()
    for name, content in files.items():
        path = run / name.replace("__", "/")
        path.write_text(content if isinstance(content, str) else json.dumps(content))
    return run / "best.pt"


def test_minimal_run_folder(tmp_path):
    ckpt = _run(tmp_path, **{"config.json": {"task": "audio", "index": "/x/processed/asvspoof/index.csv"},
                             "history.csv": "epoch,train_loss,val_loss,val_auc,val_eer,val_acc\n1,0.5,0.4,0.9,0.1,0.8\n"})
    run = model_card.load_run(ckpt)
    assert run["task"] == "audio" and run["dataset"] == "ASVspoof 2019 Logical Access"
    assert run["metrics"] is None and run["per_category"] is None and run["roc"] is None
    assert model_card.category_table(run) is None and model_card.split_table(run).empty
    assert list(model_card.history_table(run).iloc[0]) == [1, "0.5000", "0.4000", "0.9000", "10.00%", "80.00%"]
    assert "No test metrics" in metric_tiles(run) and confusion_html(run) == ""
    assert "train_audio" in training_command(run) and "--dataset asvspoof" in training_command(run)


def test_split_table_from_confusion_matrix(tmp_path):
    ckpt = _run(tmp_path, **{"config.json": {"task": "audio"}, "history.csv": "epoch,val_auc\n1,0.5\n",
                             "data.json": {"dataset": "asvspoof", "splits": {"train": {"clips": 4, "fake": 2, "real": 2}}},
                             "test__metrics.json": {"n": 10, "confusion_matrix": [[4, 1], [2, 3]], "auc": 0.8,
                                                    "eer": 0.2, "accuracy": 0.7, "precision": 0.75, "recall": 0.6,
                                                    "f1": 0.67}})
    run = model_card.load_run(ckpt)
    table = model_card.split_table(run)
    assert table.to_dict("records")[1] == {"Split": "test", "Source": "", "Clips": 10, "Fake": 5, "Real": 5}
    assert "1 false alarms on 5 real clips; 2 missed fakes out of 5" in confusion_html(run)


@pytest.mark.skipif(not (PUBLISHED / "config.json").exists(), reason="published run folder not present")
def test_published_asvspoof_run():
    """The run folder committed to the repository is complete and self-consistent."""
    run = model_card.load_run(PUBLISHED / "best.pt")
    m = run["metrics"]
    assert run["task"] == "audio" and m["n"] == 5000 and round(m["auc"], 3) == 0.977 and run["best_epoch"] == 3
    assert sum(map(sum, m["confusion_matrix"])) == m["n"]
    cats = model_card.category_table(run)
    assert len(cats) == 14 and cats.Clips.sum() == m["n"]
    preds = pd.read_csv(PUBLISHED / "test" / "predictions.csv")
    assert len(preds) == m["n"] and set(preds.label) == {0, 1}
    assert list(model_card.split_table(run).Clips) == [5000, 5000, 5000]
    assert run["curves"] and run["roc"] and len(run["history"]) == 6
