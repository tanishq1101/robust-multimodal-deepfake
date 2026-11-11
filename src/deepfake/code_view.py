"""Source snippets for the demo's Code tab, shown without comments or docstrings.

`clean_source` keeps the code itself and its layout, and drops:
  - docstrings of modules, classes and functions
  - whole-line comments and trailing comments
  - runs of blank lines (collapsed to one)

Usage:
  from deepfake.code_view import clean_source, snippets
  print(clean_source(inspect.getsource(fn)))
  for s in snippets(run): print(s["title"], s["code"])
"""

from __future__ import annotations

import ast
import inspect
import io
import textwrap
import tokenize
from pathlib import Path


def _docstring_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if len(body) < 2:                       # a body that is only a docstring must keep it
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def clean_source(source: str) -> str:
    source = textwrap.dedent(source)
    lines = source.splitlines()
    drop = _docstring_lines(ast.parse(source))
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        before = lines[row - 1][:col]
        if before.strip():
            lines[row - 1] = before.rstrip()
        else:
            drop.add(row)
    kept: list[str] = []
    for i, line in enumerate(lines, start=1):
        if i in drop:
            continue
        line = line.rstrip()
        if not line and (not kept or not kept[-1]):
            continue
        kept.append(line)
    while kept and not kept[-1]:
        kept.pop()
    return "\n".join(kept)


def _where(obj) -> str:
    path = Path(inspect.getsourcefile(obj))
    parts = path.parts
    rel = "/".join(parts[parts.index("deepfake") - 1:]) if "deepfake" in parts else path.name
    line = inspect.getsourcelines(obj)[1]
    return f"{rel}:{line}"


PYTHON_USAGE = '''from deepfake.analyzer import Analyzer

analyzer = Analyzer.from_defaults()
res = analyzer.analyse("clip.mp4")
print(res["verdict"], res["p_fake"], res["decided_by"])
print("face:", res["face"]["p_fake"], "voice:", res["voice"]["p_fake"])'''

CLI_USAGE = '''python -m deepfake.analyze clip.mp4 --out result/
python -m deepfake.analyze --samples'''


def training_steps(run: dict) -> list[dict]:
    """The commands that reproduce a run, one step per entry (no shell comments)."""
    c = run["config"]
    task = run["task"]
    script = {"audio": "train_audio", "visual": "train_visual", "fusion": "train_fusion"}.get(task, f"train_{task}")
    dataset = (run["data"] or {}).get("dataset") or Path(c.get("index", "")).parent.name or "<dataset>"
    args = [f"--dataset {dataset}", f"--run-name {run['name']}"]
    for flag, key in (("--epochs", "epochs"), ("--batch-size", "batch_size"), ("--lr", "lr"),
                      ("--weight-decay", "weight_decay"), ("--dropout", "dropout"),
                      ("--freeze-epochs", "freeze_epochs"), ("--patience", "patience"), ("--seed", "seed")):
        if c.get(key) is not None:
            args.append(f"{flag} {c[key]}")
    if c.get("init_from"):
        args.append(f"--init-from {c['init_from']}")
    if c.get("augment") is False:
        args.append("--no-augment")
    if c.get("robust_augment"):
        args.append(f"--robust-augment --robust-p {c.get('robust_p', 0.5)}")
    prep = f"python -m deepfake.preprocessing.run {dataset}" + (" --per-class 2500" if dataset == "asvspoof" else "")
    return [
        {"title": "Download, index and split the dataset",
         "text": "Notebook sections 3 and 4. Splits keep each speaker or identity in one split only.",
         "language": "shell",
         "code": "\n".join([f"python -m deepfake.data.download {dataset}",
                            f"python -m deepfake.data.manifest {dataset}",
                            f"python -m deepfake.data.splits {dataset}", prep])},
        {"title": f"Train the {task} model",
         "text": "Resumable; the best epoch by validation AUC is kept as best.pt.",
         "language": "shell",
         "code": f"python -m deepfake.training.{script} \\\n    " + " \\\n    ".join(args) + " \\\n    --device cuda"},
        {"title": "Build the results report",
         "text": "Test metrics, the per-category table and the ROC curve are written to <run>/test/ during "
                 "training; this collects every run into reports/results.md.",
         "language": "shell", "code": "python -m deepfake.report"},
    ]


def snippets(run: dict | None = None) -> list[dict]:
    """Every block of the Code tab, in reading order: {group, title, text, language, code, where}."""
    from deepfake import analyzer as analyzer_mod
    from deepfake.face_model import DFDCClassifier, FaceSwapDetector, confident_mean
    from deepfake.models.audio import AudioModel
    from deepfake.preprocessing.audio import log_mel
    from deepfake.training.engine import train_one_epoch

    out = [
        {"group": "Run it", "title": "Use the analyser from Python", "language": "python", "code": PYTHON_USAGE, "where": "",
         "text": "Loads the voice model from models/ and the face model if it was downloaded."},
        {"group": "Run it", "title": "Use it from the command line", "language": "shell", "code": CLI_USAGE, "where": "",
         "text": "First line: one file, JSON plus evidence images. Second line: every labelled demo sample."},
    ]
    source_blocks = [
        ("Analyser: video, audio and image", analyzer_mod.Analyzer.analyse,
         "Runs the face and voice branches, fuses them and returns one result shape for every input type."),
        ("Video fusion rule", analyzer_mod.fuse_video,
         "Face and voice scores to one video score: fake if either is fake, but a borderline score "
         "cannot overrule a confident one."),
        ("Speech windows", analyzer_mod.speech_windows,
         "Picks the 4-second windows that contain speech, so silence does not drive the voice score."),
        ("Voice model", AudioModel, "ResNet-18 over a one-channel log-mel spectrogram, one logit out."),
        ("Audio preprocessing", log_mel, "16 kHz waveform to a 128-band log-mel spectrogram."),
        ("Face model", DFDCClassifier, "EfficientNet-B7 from the DFDC-winning solution, loaded strictly."),
        ("Frame scores to a video score", confident_mean,
         "If many frames are confidently fake, average only those; a swap is often visible in some frames only."),
        ("Face scoring with its mirror image", FaceSwapDetector.score_canvases,
         "Scores every face and its mirror image and averages the two logits."),
        ("Training loop", train_one_epoch, "One epoch: mixed precision on a GPU, gradient clipping, frozen layers kept frozen."),
    ]
    for title, obj, text in source_blocks:
        obj = inspect.unwrap(obj)
        out.append({"group": "How it works", "title": title, "text": text, "language": "python",
                    "code": clean_source(inspect.getsource(obj)), "where": _where(obj)})
    if run is not None:
        for step in training_steps(run):
            out.append({**step, "group": "Reproduce the voice model", "where": ""})
    return out
