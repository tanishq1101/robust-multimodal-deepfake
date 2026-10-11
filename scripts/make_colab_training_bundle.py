"""Package the current reviewed source into a self-contained Colab training notebook.

No GitHub push, credentials, datasets, Git history, or virtual environment is included.
Usage: python scripts/make_colab_training_bundle.py /tmp/deepfake_colab_training.ipynb
"""

import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import textwrap
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(source).strip()}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(out):
    files = sorted(ROOT.glob("src/deepfake/**/*.py")) + sorted(ROOT.glob("configs/*.yaml"))
    files += [ROOT / "pyproject.toml", ROOT / "requirements-colab.txt"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.writestr(str(path.relative_to(ROOT)), path.read_bytes())
    payload = buf.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    setup = f'''
import base64, hashlib, io, json, os, subprocess, sys, zipfile
from pathlib import Path
import torch
assert torch.cuda.is_available(), "Select Runtime > Change runtime type > T4 GPU before running."
print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))
SOURCE_SHA256 = {digest!r}
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA256
CODE = Path("/content") / ("DeepFake_reviewed_" + SOURCE_SHA256[:12])
CODE.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        assert (CODE / name).resolve().is_relative_to(CODE.resolve())
    archive.extractall(CODE)
os.chdir(CODE)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
sys.path.insert(0, str(CODE / "src"))
print("Reviewed source installed:", SOURCE_SHA256)
'''
    cells = [cell("markdown", """
        # Reviewed deepfake pipeline: real-data PyTorch training
        This notebook embeds the current reviewed source. Select a **T4 GPU**, then run cells
        in order. Checkpoints and reports go to `MyDrive/deepfake_project`.

        The first run uses the existing project's ASVspoof balanced-subset recipe: up to
        **2,500 samples per class per official split**, seed 42. Its scores are subset results,
        not a full ASVspoof benchmark. Face/audio/fusion training also runs if FakeAVCeleb
        processed data or an approved archive is available in the project folder.
        No credentials are embedded. Drive mounting may require your authorization.
        """), cell("code", setup), cell("code", """
        from google.colab import drive
        drive.mount("/content/drive")
        DATA = Path("/content/drive/MyDrive/deepfake_project")
        DATA.mkdir(parents=True, exist_ok=True)
        os.environ["DEEPFAKE_DATA_ROOT"] = str(DATA)
        os.environ["DEEPFAKE_RAW_ROOT"] = "/content/raw"
        from deepfake.config import load_config, get_path
        cfg = load_config()
        PER_CLASS = 2500
        AUDIO_RUN = "audio_asvspoof_" + SOURCE_SHA256[:8]
        logs = DATA / "outputs" / ("training_" + SOURCE_SHA256[:8])
        logs.mkdir(parents=True, exist_ok=True)
        def run_module(module, *args):
            command = [sys.executable, "-u", "-m", module, *map(str, args)]
            print("Running:", " ".join(command), flush=True)
            with (logs / (module.rsplit(".", 1)[-1] + ".log")).open("a") as log:
                proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line); log.flush()
                status = proc.wait()
            if status:
                raise RuntimeError(f"{module} failed (exit {status}); see {logs}")
        (logs / "source.json").write_text(json.dumps({"source_sha256": SOURCE_SHA256,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "asvspoof_per_class_per_split": PER_CLASS, "seed": cfg["seed"]}, indent=2))
        run_module("deepfake.data.download", "status")
        """), cell("code", """
        # Public real speech/spoof dataset; downloads resume after a disconnect.
        run_module("deepfake.data.download", "asvspoof")
        run_module("deepfake.data.manifest", "asvspoof")
        run_module("deepfake.data.splits", "asvspoof")
        run_module("deepfake.preprocessing.run", "asvspoof", "--per-class", PER_CLASS, "--workers", 2)
        import pandas as pd
        asv_index = get_path(cfg, "processed", "asvspoof", "index.csv")
        asv = pd.read_csv(asv_index, keep_default_na=False)
        display(asv.groupby(["split", "audio_label"]).size().unstack(fill_value=0))
        errors = asv[asv.error != ""]
        if len(errors):
            display(errors[["sample_id", "error"]].head(20))
            raise RuntimeError("Preprocessing errors must be resolved before training.")
        """), cell("code", """
        run_module("deepfake.training.train_audio", "--dataset", "asvspoof", "--run-name", AUDIO_RUN,
                   "--epochs", 15, "--batch-size", 32, "--workers", 2, "--device", "cuda")
        audio_dir = get_path(cfg, "checkpoints", AUDIO_RUN)
        required = ["best.pt", "last.pt", "history.csv", "curves.png",
                    "test/metrics.json", "test/predictions.csv", "test/per_category.csv"]
        for name in required:
            assert (audio_dir / name).is_file(), f"Missing training output: {name}"
        from IPython.display import Image, display
        display(pd.read_csv(audio_dir / "history.csv"))
        metrics = json.loads((audio_dir / "test/metrics.json").read_text())
        print(json.dumps(metrics, indent=2))
        display(Image(str(audio_dir / "curves.png")))
        display(Image(str(audio_dir / "test/roc.png")))
        display(pd.read_csv(audio_dir / "test/per_category.csv"))
        """), cell("code", """
        # Look only in the project's own folder for approved FakeAVCeleb archives/data.
        from deepfake.data import download as dl
        fake_index = get_path(cfg, "processed", "fakeavceleb", "index.csv")
        if not fake_index.exists():
            archives = sorted(p for p in DATA.rglob("*.zip") if "fakeav" in p.name.lower())
            if archives:
                run_module("deepfake.data.download", "fakeavceleb", "--archive", archives[0])
            if dl.verify_fakeavceleb(get_path(cfg, "raw", "fakeavceleb"))["ok"]:
                run_module("deepfake.data.manifest", "fakeavceleb")
                run_module("deepfake.data.splits", "fakeavceleb")
                run_module("deepfake.preprocessing.run", "fakeavceleb", "--per-class", 400, "--workers", 2)
        if fake_index.exists():
            visual_run = "visual_fakeavceleb_" + SOURCE_SHA256[:8]
            audio_ft = "audio_fakeavceleb_" + SOURCE_SHA256[:8]
            common = ["--dataset", "fakeavceleb", "--epochs", "10", "--workers", "2", "--device", "cuda"]
            run_module("deepfake.training.train_visual", *common, "--run-name", visual_run)
            run_module("deepfake.training.train_audio", *common, "--run-name", audio_ft,
                       "--batch-size", 32, "--lr", "3e-5", "--freeze-epochs", 0,
                       "--init-from", audio_dir / "best.pt")
            visual_ckpt = get_path(cfg, "checkpoints", visual_run, "best.pt")
            audio_ckpt = get_path(cfg, "checkpoints", audio_ft, "best.pt")
            models = ["--model", f"visual_only={visual_ckpt}", "--model", f"audio_only={audio_ckpt}"]
            for fusion in ["late", "intermediate"]:
                name = f"fusion_{fusion}_fakeavceleb_" + SOURCE_SHA256[:8]
                run_module("deepfake.training.train_fusion", *common, "--run-name", name,
                           "--fusion", fusion, "--visual-ckpt", visual_ckpt, "--audio-ckpt", audio_ckpt)
                models += ["--model", f"{fusion}={get_path(cfg, 'checkpoints', name, 'best.pt')}"]
            run_module("deepfake.evaluation.compare", "--dataset", "fakeavceleb", "--device", "cuda", *models)
            print("Multimodal training completed; inspect the saved validation/test results.")
        else:
            print("MULTIMODAL PENDING: FakeAVCeleb is not available. The audio checkpoint remains saved.")
            print("Official access: https://github.com/DASH-Lab/FakeAVCeleb")
        """), cell("code", """
        run_module("deepfake.report", "--out", DATA / "report")
        from IPython.display import Markdown
        display(Markdown((DATA / "report/results.md").read_text()))
        status = {"source_sha256": SOURCE_SHA256, "audio_checkpoint": str(audio_dir / "best.pt"),
                  "audio_test_metrics": str(audio_dir / "test/metrics.json"),
                  "multimodal_available": fake_index.exists(), "report": str(DATA / "report/results.md")}
        (logs / "completion.json").write_text(json.dumps(status, indent=2))
        print("Outputs saved persistently:", json.dumps(status, indent=2))
        """)]
    notebook = {"nbformat": 4, "nbformat_minor": 0,
                "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}, "accelerator": "GPU",
                             "colab": {"gpuType": "T4", "provenance": []}}, "cells": cells}
    out = Path(out)
    out.write_text(json.dumps(notebook, indent=1) + "\n")
    print(json.dumps({"notebook": str(out), "source_sha256": digest,
                      "source_files": len(files), "bytes": out.stat().st_size}, indent=2))
    return notebook


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/deepfake_colab_training.ipynb")
