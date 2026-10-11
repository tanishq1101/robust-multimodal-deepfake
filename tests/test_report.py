"""Phase 8: report builder."""

import pandas as pd

from deepfake import report
from deepfake.config import load_config
from deepfake.evaluation import compare
from deepfake.explainability.gallery import build_gallery

from conftest import DEVICE


def test_report_without_any_results(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFAKE_DATA_ROOT", str(tmp_path / "empty"))
    md = report.build_report(load_config(), tmp_path / "rep").read_text()
    assert md.count(report.NOT_RUN) >= 6  # datasets, runs, phases 2-6
    assert "## Setup" in md


def test_report_collects_runs(branches, late_fusion_ckpt, tiny_index, tmp_path, monkeypatch):
    root, v, a = branches
    monkeypatch.setenv("DEEPFAKE_DATA_ROOT", str(root))
    cfg = load_config()
    compare.main(["--index", str(tiny_index), "--out", str(root / "outputs" / "compare_tiny_test"), "--device", DEVICE,
                  "--model", f"visual_only={v}", "--model", f"late={late_fusion_ckpt}"])
    build_gallery(v, tiny_index, "test", cfg, root / "outputs" / "gallery_v", per_category=1, device=DEVICE)
    (tmp_path / "observations.md").write_text("Fusion helps on RealVideo-FakeAudio.")

    md = report.build_report(cfg, tmp_path).read_text()
    runs = pd.read_csv(tmp_path / "runs.csv")
    assert {"v", "a", "late_fixture"} <= set(runs.run)
    assert set(runs.task) >= {"visual", "audio", "fusion"}
    for text in ("### `v`", "### `late_fixture`", "Late-fusion rule chosen", "compare_tiny_test",
                 "gallery_v", "Fusion helps on RealVideo-FakeAudio."):
        assert text in md, text
    # every image referenced in the report was copied next to it
    for line in md.splitlines():
        if line.startswith("![]("):
            assert (tmp_path / line[4:-1]).exists(), line
    assert (tmp_path / "figures" / "compare_tiny_test.png").exists()
