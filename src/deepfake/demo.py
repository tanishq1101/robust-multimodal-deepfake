"""Gradio demo UI (Phase 7). Launch with app/app.py.

Four tabs:
  Analyse            upload a video or an audio file (or record your voice) -> verdict, branch
                     scores, input checks, Grad-CAM evidence
  Model & training   what the loaded model is: training data, hyperparameters, test metrics,
                     per-category results, training curves and ROC
  Code               the exact command that trained the model and the source of its key parts
  About              pipeline, limitations, datasets
"""

from __future__ import annotations

import html
import inspect
from pathlib import Path

import gradio as gr
import pandas as pd

from deepfake import model_card
from deepfake.inference import DeepfakeDetector

TITLE = "Robust Multimodal Deepfake Analysis"
THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
CSS = """
.gradio-container {max-width: 1240px !important; margin: 0 auto !important;}
.df-hero {padding: 4px 2px 10px;}
.df-hero h1 {font-size: 1.65rem; font-weight: 700; margin: 0 0 4px; letter-spacing: -0.01em;}
.df-hero p {margin: 0; color: var(--body-text-color-subdued); max-width: 760px;}
.df-chips {display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px;}
.df-chip {display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font-size: 0.8rem; border: 1px solid var(--border-color-primary); background: var(--block-background-fill);}
.df-dot {width: 8px; height: 8px; border-radius: 50%; display: inline-block;}
.df-on {background: #16a34a;} .df-off {background: #94a3b8;}
.df-card {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 18px 20px;
  background: var(--block-background-fill);}
.df-empty {color: var(--body-text-color-subdued); text-align: center; padding: 42px 16px;}
.df-empty b {display: block; color: var(--body-text-color); font-size: 1.05rem; margin-bottom: 4px;}
.df-verdict {display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;}
.df-badge {font-size: 1.9rem; font-weight: 800; letter-spacing: 0.02em; line-height: 1;}
.df-sub {color: var(--body-text-color-subdued); font-size: 0.9rem; margin-top: 6px;}
.df-pct {font-size: 2.1rem; font-weight: 700; font-variant-numeric: tabular-nums; text-align: right; line-height: 1;}
.df-pct small {display: block; font-size: 0.75rem; font-weight: 500; color: var(--body-text-color-subdued); margin-top: 4px;}
.df-gauge {position: relative; height: 12px; border-radius: 999px; margin: 16px 0 6px;
  background: linear-gradient(90deg, #16a34a 0%, #eab308 50%, #dc2626 100%); opacity: 0.9;}
.df-needle {position: absolute; top: -5px; width: 4px; height: 22px; border-radius: 2px;
  background: var(--body-text-color); transform: translateX(-50%); box-shadow: 0 0 0 2px var(--block-background-fill);}
.df-thr {position: absolute; top: -3px; height: 18px; border-left: 2px dashed var(--body-text-color-subdued);}
.df-scale {display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--body-text-color-subdued);}
.df-explain {margin-top: 12px; font-size: 0.95rem;}
.df-section {font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--body-text-color-subdued); margin: 0 0 8px;}
.df-row {display: grid; grid-template-columns: 150px 1fr 64px; align-items: center; gap: 10px; padding: 7px 0;
  border-bottom: 1px solid var(--border-color-primary); font-size: 0.92rem;}
.df-row:last-child {border-bottom: none;}
.df-bar {height: 8px; border-radius: 999px; background: var(--border-color-primary); overflow: hidden;}
.df-bar > span {display: block; height: 100%; border-radius: 999px;}
.df-val {text-align: right; font-variant-numeric: tabular-nums; font-weight: 600;}
.df-na {color: var(--body-text-color-subdued); font-size: 0.85rem;}
.df-facts {display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-top: 14px;}
.df-fact {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 8px 10px;}
.df-fact span {display: block; font-size: 0.75rem; color: var(--body-text-color-subdued);}
.df-fact b {font-size: 0.95rem; overflow-wrap: anywhere;}
.df-warn {margin-top: 12px; padding: 10px 12px; border-radius: 10px; font-size: 0.9rem;
  background: rgba(234, 179, 8, 0.12); border: 1px solid rgba(234, 179, 8, 0.45);}
.df-warn li {margin-left: 16px;}
.df-tiles {display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px;}
.df-tile {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px;
  background: var(--block-background-fill);}
.df-tile span {display: block; font-size: 0.78rem; color: var(--body-text-color-subdued);}
.df-tile b {font-size: 1.45rem; font-variant-numeric: tabular-nums;}
.df-tile small {display: block; font-size: 0.75rem; color: var(--body-text-color-subdued); margin-top: 2px;}
.df-cm {border-collapse: separate !important; border-spacing: 6px; font-variant-numeric: tabular-nums;
  border: none !important; margin: 0 0 8px !important; width: auto !important;}
.df-cm tr, .df-cm td, .df-cm th {border: none !important; background-clip: padding-box;}
.df-cm td, .df-cm th {padding: 8px 12px; text-align: center; border-radius: 8px; font-size: 0.9rem;}
.df-cm th {font-weight: 600; color: var(--body-text-color-subdued); background: transparent !important;}
.df-caption {font-size: 0.9rem; font-weight: 600; margin: 4px 0 6px;}
.df-ok {background: rgba(22, 163, 74, 0.16);} .df-bad {background: rgba(220, 38, 38, 0.16);}
.df-note {font-size: 0.88rem; color: var(--body-text-color-subdued);}
.df-views .wrap {gap: 6px !important;}
"""
VERDICT_COLORS = {"FAKE": "#dc2626", "REAL": "#16a34a", "INCONCLUSIVE": "#d97706"}
BRANCH_LABELS = [("overall", "Overall (fusion)"), ("visual", "Face model"), ("audio", "Voice model")]
TASK_BRANCHES = {"audio": {"audio"}, "visual": {"visual"}, "fusion": {"overall", "visual", "audio"}}


def _pct(p) -> str:
    return "n/a" if p is None else f"{100 * p:.1f}%"


def _score_color(p: float, threshold: float) -> str:
    return VERDICT_COLORS["FAKE"] if p >= threshold else VERDICT_COLORS["REAL"]


# --------------------------------------------------------------------------- analysis results

def format_details(res: dict) -> str:
    """Markdown summary of a DeepfakeDetector.analyse() result (also used for the copyable summary)."""
    s, info = res["scores"], res["info"]
    lines = [f"### Verdict: **{res['verdict']}** (P(fake) = {_pct(res['p_fake'])}, threshold {res['threshold']:.2f})", ""]
    lines += ["| Evidence | P(fake) |", "|---|---|"]
    lines += [f"| {name} | {_pct(s[key])} |" for key, name in BRANCH_LABELS if key in s]
    lines += ["", f"- Faces detected in {info['faces_detected']} / {info['frames_sampled']} sampled frames",
              f"- Audio: {info['audio_seconds']} s analysed" if info["has_audio"] else "- Audio: none found",
              f"- Model: `{res['task']}` (run `{Path(res['checkpoint']).parent.name}`)"]
    if res.get("explained_class"):
        lines.append(f"- Heatmaps show evidence for: **{res['explained_class'].upper()}**")
    for w in res["warnings"]:
        lines.append(f"- ⚠️ {w}")
    return "\n".join(lines)


def _interpretation(res: dict, model_name: str) -> str:
    p, t = res["p_fake"], res["threshold"]
    if p is None:
        return "The loaded model found nothing it can judge in this file (see the notes below)."
    who = {"audio": "voice", "visual": "face", "fusion": "face and voice"}.get(res["task"], "input")
    if res["verdict"] == "FAKE":
        return (f"The {model_name} rates the {who} as <b>{_pct(p)} likely fake</b>, at or above the "
                f"decision threshold of {_pct(t)}.")
    return (f"The {model_name} rates the {who} as <b>{_pct(p)} likely fake</b>, below the decision "
            f"threshold of {_pct(t)}, so it is treated as real.")


def verdict_html(res: dict | None, model_name: str = "model") -> str:
    if res is None:
        return ('<div class="df-card df-empty"><b>No analysis yet</b>Upload a video or an audio file, '
                'or record your voice, then press <i>Analyse</i>.</div>')
    color = VERDICT_COLORS[res["verdict"]]
    p, t = res["p_fake"], res["threshold"]
    needle = "" if p is None else f'<div class="df-needle" style="left:{100 * p:.1f}%"></div>'
    gauge = (f'<div class="df-gauge"><div class="df-thr" style="left:{100 * t:.1f}%" title="threshold"></div>{needle}</div>'
             '<div class="df-scale"><span>0% · real</span><span>threshold ┆</span><span>fake · 100%</span></div>')
    return (f'<div class="df-card"><div class="df-verdict"><div>'
            f'<div class="df-badge" style="color:{color}">{res["verdict"]}</div>'
            f'<div class="df-sub">P(fake) from the {html.escape(model_name)}</div></div>'
            f'<div class="df-pct" style="color:{color}">{_pct(p)}<small>probability of fake</small></div></div>'
            f'{gauge}<div class="df-explain">{_interpretation(res, html.escape(model_name))}</div></div>')


def evidence_html(res: dict | None) -> str:
    if res is None:
        return ""
    s, info, t = res["scores"], res["info"], res["threshold"]
    available = TASK_BRANCHES.get(res["task"], set(s))
    rows = []
    for key, name in BRANCH_LABELS:
        if key not in available:
            rows.append(f'<div class="df-row"><span>{name}</span><span class="df-na">not part of the loaded model'
                        f'</span><span class="df-val df-na">–</span></div>')
        elif s.get(key) is None:
            rows.append(f'<div class="df-row"><span>{name}</span><span class="df-na">no usable input in this file'
                        f'</span><span class="df-val df-na">n/a</span></div>')
        else:
            p = s[key]
            rows.append(f'<div class="df-row"><span>{name}</span><div class="df-bar"><span style="width:{100 * p:.1f}%;'
                        f'background:{_score_color(p, t)}"></span></div><span class="df-val">{_pct(p)}</span></div>')
    facts = [("File", html.escape(str(info.get("file", "")))),
             ("Audio analysed", f'{info["audio_seconds"]} s' if info["has_audio"] else "none"),
             ("Faces found", f'{info["faces_detected"]} / {info["frames_sampled"]} frames'
              if info["frames_sampled"] else "no video frames"),
             ("Threshold", _pct(t))]
    if res.get("explained_class"):
        facts.append(("Heatmaps explain", res["explained_class"].upper()))
    fact_html = "".join(f'<div class="df-fact"><span>{k}</span><b>{v}</b></div>' for k, v in facts)
    warn = ""
    if res["warnings"]:
        warn = '<div class="df-warn"><b>Notes</b><ul>' + "".join(
            f"<li>{html.escape(w)}</li>" for w in res["warnings"]) + "</ul></div>"
    return (f'<div class="df-card"><div class="df-section">Evidence by model branch</div>{"".join(rows)}'
            f'<div class="df-facts">{fact_html}</div>{warn}</div>')


def make_handler(detector: DeepfakeDetector, model_name: str = "model"):
    """Returns analyse(mode, video, audio, threshold, target) -> UI updates."""
    def analyse(mode: str, video_path: str | None, audio_path: str | None, threshold: float, target: str):
        path = video_path if mode == "video" else audio_path
        if not path:
            raise gr.Error("Please upload a video first." if mode == "video"
                           else "Please upload or record audio first.")
        res = detector.analyse(path, with_explanation=True, target=target, threshold=float(threshold))
        faces = res.get("face_overlays", [])
        spec = res.get("spectrogram")
        summary = {k: v for k, v in res.items() if k not in ("face_overlays", "spectrogram", "figure")}
        return (verdict_html(res, model_name), evidence_html(res),
                gr.update(value=spec, visible=spec is not None),
                gr.update(value=faces, visible=bool(faces)),
                gr.update(visible=spec is None and not faces),
                summary, format_details(res))
    return analyse


# --------------------------------------------------------------------------- model page

def header_html(run: dict) -> str:
    m = run["metrics"] or {}
    task = run["task"]
    chips = [f'<span class="df-chip"><span class="df-dot {"df-on" if task in ("audio", "fusion") else "df-off"}"></span>'
             f'Voice model {"loaded" if task in ("audio", "fusion") else "not loaded"}</span>',
             f'<span class="df-chip"><span class="df-dot {"df-on" if task in ("visual", "fusion") else "df-off"}"></span>'
             f'Face model {"loaded" if task in ("visual", "fusion") else "not loaded"}</span>',
             f'<span class="df-chip">Trained on {html.escape(run["dataset"])}</span>']
    if m.get("auc") is not None:
        chips.append(f'<span class="df-chip">Test AUC {m["auc"]:.3f} · EER {100 * m["eer"]:.1f}%</span>')
    return (f'<div class="df-hero"><h1>{TITLE}</h1><p>Checks whether a video or a voice recording has been '
            f'manipulated, and shows the evidence behind each decision. A research prototype: scores are '
            f'model estimates, not forensic proof.</p><div class="df-chips">{"".join(chips)}</div></div>')


def metric_tiles(run: dict) -> str:
    m = run["metrics"]
    if not m:
        return '<div class="df-card df-empty"><b>No test metrics saved for this run</b>Evaluate it with ' \
               '<code>python -m deepfake.evaluation.evaluate</code>.</div>'
    eer_thr = m.get("at_eer_threshold", {}).get("accuracy")
    tiles = [("Test AUC", f'{m["auc"]:.3f}', "ranking quality, 1.0 = perfect"),
             ("EER", f'{100 * m["eer"]:.1f}%', "equal error rate, lower is better"),
             ("Accuracy", f'{100 * m["accuracy"]:.1f}%', "at threshold 0.5"),
             ("Precision", f'{m["precision"]:.3f}', "of clips flagged fake"),
             ("Recall", f'{m["recall"]:.3f}', "share of fakes caught"),
             ("F1", f'{m["f1"]:.3f}', "at threshold 0.5")]
    if eer_thr is not None:
        tiles.append(("Accuracy at EER threshold", f"{100 * eer_thr:.1f}%",
                      f'threshold {m.get("eer_threshold", 0):.4f}'))
    tiles.append(("Test clips", f'{m["n"]:,}', f'best epoch {m.get("best_epoch", "–")}'))
    return '<div class="df-tiles">' + "".join(
        f'<div class="df-tile"><span>{k}</span><b>{v}</b><small>{s}</small></div>' for k, v, s in tiles) + "</div>"


def confusion_html(run: dict) -> str:
    m = run["metrics"]
    if not m or "confusion_matrix" not in m:
        return ""
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    return ('<div class="df-card"><div class="df-section">Confusion matrix (test, threshold 0.5)</div>'
            '<table class="df-cm"><tr><th></th><th>Predicted real</th><th>Predicted fake</th></tr>'
            f'<tr><th>Actually real</th><td class="df-ok">{tn:,}</td><td class="df-bad">{fp:,}</td></tr>'
            f'<tr><th>Actually fake</th><td class="df-bad">{fn:,}</td><td class="df-ok">{tp:,}</td></tr></table>'
            f'<div class="df-note">{fp:,} false alarms on {tn + fp:,} real clips; {fn:,} missed fakes out of '
            f'{fn + tp:,}.</div></div>')


# --------------------------------------------------------------------------- code page

def training_command(run: dict) -> str:
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
    lines = ["# 1. Download, index and split the dataset (Colab: notebooks 01 and 02)",
             f"python -m deepfake.data.download {dataset}",
             f"python -m deepfake.data.manifest {dataset}",
             f"python -m deepfake.data.splits {dataset}",
             f"python -m deepfake.preprocessing.run {dataset}" + (" --per-class 2500" if dataset == "asvspoof" else ""),
             "", f"# 2. Train ({run['branch']})",
             f"python -m deepfake.training.{script} \\\n    " + " \\\n    ".join(args) + " \\\n    --device cuda",
             "", "# 3. Test-set metrics, per-category table and ROC are written to <run>/test/ automatically.",
             "#    Build the results report:", "python -m deepfake.report"]
    return "\n".join(lines)


def code_views(run: dict) -> dict[str, tuple[str, str]]:
    """Name -> (language, code) for the Code tab; sources are read from the installed package."""
    from deepfake.datasets.multimodal import MultimodalDataset
    from deepfake.models.audio import AudioModel
    from deepfake.models.fusion import MultimodalModel, late_combine
    from deepfake.models.visual import VisualModel
    from deepfake.preprocessing.audio import log_mel
    from deepfake.training.engine import fit, train_one_epoch

    views = {"How this model was trained (commands)": ("shell", training_command(run)),
             "Use the model from Python": ("python", _python_usage(run))}
    for name, obj in (("Voice model: AudioModel", AudioModel), ("Face model: VisualModel", VisualModel),
                      ("Fusion model: MultimodalModel", MultimodalModel), ("Late-fusion rules", late_combine),
                      ("Audio preprocessing: log-mel", log_mel), ("Dataset loader: MultimodalDataset", MultimodalDataset),
                      ("Training loop: one epoch", train_one_epoch), ("Training loop: fit", fit)):
        views[name] = ("python", inspect.getsource(obj))
    return views


def _python_usage(run: dict) -> str:
    return (f'from deepfake.inference import DeepfakeDetector\n\n'
            f'det = DeepfakeDetector("{Path(run["checkpoint"]).as_posix()}")\n'
            f'result = det.analyse("clip.mp4", threshold=0.5)   # video or audio file\n'
            f'print(result["verdict"], result["p_fake"], result["scores"], result["warnings"])\n\n'
            f'# Command line: JSON summary + explanation.png\n'
            f'# python -m deepfake.inference clip.mp4 --checkpoint {Path(run["checkpoint"]).as_posix()} --out result/')


ABOUT = """
### How it works
1. **Input checks.** 16 frames are sampled from the video and faces are found with YuNet. The first
   10 s of audio are converted to a 16 kHz, 128-bin log-mel spectrogram.
2. **Branch models.** The *face model* (EfficientNet-B0) scores each face crop. The *voice model*
   (ResNet-18) scores the central 4-second spectrogram window.
3. **Fusion.** When both branches are trained, a fusion model combines them: a video is fake if
   the face **or** the voice is fake.
4. **Evidence.** Grad-CAM heatmaps show which face regions and which time–frequency regions drove
   the decision.

### Reading the result
- **P(fake)** is the model's probability that the input is manipulated. The **threshold** decides
  the verdict. Lowering it catches more fakes but raises more false alarms on real recordings.
- **INCONCLUSIVE** means the loaded model had nothing it could judge, for example a video without
  an audio track when only the voice model is loaded.
- Heatmaps show *where* the model looked, not proof of manipulation.

### Limitations
- The current voice model was trained on ASVspoof 2019 LA studio recordings. Different
  microphones, rooms or compression may change its scores.
- It misses some unseen attack types; the *Model & training* tab lists per-attack results.
- The face and fusion models are pending FakeAVCeleb access.

### Datasets
ASVspoof 2019 LA (Wang et al., 2020) · FakeAVCeleb (Khalid et al., 2021) · DFDC (Dolhansky et al., 2020) ·
Celeb-DF v2 (Li et al., 2020). See `DATASETS.md` and `reports/report.md` in the repository.
"""


# --------------------------------------------------------------------------- layout

def create_demo(detector: DeepfakeDetector) -> gr.Blocks:
    run = model_card.load_run(detector.checkpoint, detector.run_config)
    model_name = {"audio": "voice model", "visual": "face model", "fusion": "fusion model"}.get(detector.task, "model")
    views = code_views(run)
    first_view = next(iter(views))

    with gr.Blocks(title=TITLE) as demo:
        gr.HTML(header_html(run))
        with gr.Tabs():
            with gr.Tab("Analyse"):
                mode = gr.State("audio" if detector.task == "audio" else "video")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=320):
                        with gr.Tabs(selected="audio" if detector.task == "audio" else "video"):
                            with gr.Tab("Video", id="video") as video_tab:
                                video = gr.Video(label="Video file", sources=["upload"], height=280)
                            with gr.Tab("Audio / voice", id="audio") as audio_tab:
                                audio = gr.Audio(label="Audio file or microphone recording",
                                                 sources=["upload", "microphone"], type="filepath")
                        with gr.Accordion("Decision settings", open=False):
                            threshold = gr.Slider(0.01, 0.99, value=detector.threshold, step=0.01,
                                                  label="Decision threshold",
                                                  info="P(fake) at or above this value counts as FAKE.")
                            target = gr.Radio(["predicted", "fake", "real"], value="predicted",
                                              label="Heatmaps show evidence for")
                        with gr.Row():
                            run_btn = gr.Button("Analyse", variant="primary", scale=3)
                            clear_btn = gr.Button("Clear", scale=1)
                        gr.Markdown(
                            f"<span class='df-note'>Loaded: <b>{run['branch']}</b>. "
                            + ("Videos are judged on their audio track." if detector.task == "audio" else
                               "Videos are judged on their faces." if detector.task == "visual" else
                               "Videos are judged on faces and voice.")
                            + " Files stay on this computer.</span>")
                    with gr.Column(scale=7, min_width=320):
                        verdict = gr.HTML(verdict_html(None))
                        evidence = gr.HTML()
                gr.Markdown("### Why the model decided this\n<span class='df-note'>Grad-CAM heatmaps: red marks "
                            "the regions that most influenced the decision. On the spectrogram, time runs left to "
                            "right and pitch rises upwards.</span>")
                no_evidence = gr.Markdown("<span class='df-note'>Heatmaps appear here after an analysis.</span>")
                spectrogram = gr.Image(show_label=False, visible=False, interactive=False,
                                       buttons=["fullscreen", "download"])
                faces = gr.Gallery(label="Face evidence: Grad-CAM on sampled frames", columns=6, height="auto",
                                   visible=False)
                with gr.Accordion("Full result (JSON and summary)", open=False):
                    raw = gr.JSON(label="Result")
                    summary = gr.Markdown()

            with gr.Tab("Model & training"):
                gr.Markdown(f"### {run['branch']}\nRun `{run['name']}` · "
                            + ((run["data"] or {}).get("description") or f"trained on **{run['dataset']}**"))
                gr.HTML(metric_tiles(run))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("#### Data")
                        gr.Dataframe(model_card.split_table(run), interactive=False, wrap=True, max_height=260)
                        gr.HTML(confusion_html(run))
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("#### Training settings")
                        gr.Dataframe(model_card.config_table(run), interactive=False, wrap=True, max_height=640,
                                     column_widths=["42%", "58%"])
                cats = model_card.category_table(run)
                if cats is not None:
                    gr.Markdown("#### Results per category" + (
                        " (ASVspoof attack systems A07–A19 and bona fide speech)" if "asvspoof" in run["dataset"].lower()
                        else ""))
                    gr.Dataframe(cats, interactive=False, wrap=True, max_height=720)
                if run["curves"]:
                    gr.HTML('<div class="df-caption">Training curves: loss and validation metrics per epoch</div>')
                    gr.Image(run["curves"], show_label=False, interactive=False, buttons=["fullscreen", "download"])
                hist = model_card.history_table(run)
                with gr.Row(equal_height=False):
                    if run["roc"]:
                        with gr.Column(scale=1, min_width=300):
                            gr.HTML('<div class="df-caption">ROC curve on the test set</div>')
                            gr.Image(run["roc"], show_label=False, interactive=False, height=420,
                                     buttons=["fullscreen", "download"])
                    if hist is not None:
                        with gr.Column(scale=1, min_width=300):
                            gr.HTML('<div class="df-caption">Training history per epoch</div>')
                            gr.Dataframe(hist, interactive=False, max_height=420)

            with gr.Tab("Code"):
                gr.Markdown("The command that trained the loaded model, and the source of its main parts, "
                            "read directly from the installed `deepfake` package.")
                view = gr.Radio(list(views), value=first_view, label="Show", elem_classes="df-views")
                code = gr.Code(views[first_view][1], language=views[first_view][0], interactive=False)
                view.change(lambda name: gr.update(value=views[name][1], language=views[name][0]), view, code)

            with gr.Tab("About"):
                gr.Markdown(ABOUT)

        video_tab.select(lambda: "video", None, mode)
        audio_tab.select(lambda: "audio", None, mode)
        outputs = [verdict, evidence, spectrogram, faces, no_evidence, raw, summary]
        run_btn.click(make_handler(detector, model_name), [mode, video, audio, threshold, target], outputs)
        clear_btn.click(lambda: (None, None, verdict_html(None), "", gr.update(value=None, visible=False),
                                 gr.update(value=None, visible=False), gr.update(visible=True), None, ""),
                        None, [video, audio, *outputs])
    return demo
