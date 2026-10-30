"""Gradio demo UI. Launch with app/app.py.

Tabs:
  Video    upload a clip or pick a demo sample -> verdict, face and voice scores, timeline,
           Grad-CAM on faces and on the spectrogram
  Audio    upload or record speech -> verdict, score per 4 s window, spectrogram Grad-CAM
  Image    upload a photo -> verdict, detected face, Grad-CAM on the face
  Samples  every labelled demo clip analysed in one go, against its ground truth
  Models   what each loaded model is, how it was trained and how well it did
  Code     the command that trained the voice model and the source of its key parts
  About    pipeline, how to read the results, limitations, sources

Works with Gradio 5 and 6: keyword arguments a version does not know are dropped.
"""

from __future__ import annotations

import html
import inspect
import json
from pathlib import Path

import gradio as gr
import pandas as pd

from deepfake import model_card
from deepfake.analyze import SAMPLES_DIR, summary
from deepfake.analyzer import FAKE, INCONCLUSIVE, REAL, VIDEO_RULES, Analyzer
from deepfake.face_model import FaceSwapDetector

TITLE = "Robust Multimodal Deepfake Analysis"
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]).set(
    button_primary_background_fill="*primary_600", button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white")
CSS = """
.gradio-container {max-width: 1280px !important; margin: 0 auto !important;}
.df-hero {display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap;
  padding: 6px 2px 14px;}
.df-hero h1 {font-size: 1.7rem; font-weight: 750; margin: 0 0 4px; letter-spacing: -0.015em;}
.df-hero p {margin: 0; color: var(--body-text-color-subdued); max-width: 720px; font-size: 0.95rem;}
.df-chips {display: flex; flex-wrap: wrap; gap: 6px;}
.df-chip {display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 999px;
  font-size: 0.8rem; border: 1px solid var(--border-color-primary); background: var(--block-background-fill);}
.df-dot {width: 8px; height: 8px; border-radius: 50%; display: inline-block;}
.df-on {background: #16a34a;} .df-off {background: #94a3b8;}
.df-card {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 18px 20px;
  background: var(--block-background-fill);}
.df-empty {color: var(--body-text-color-subdued); text-align: center; padding: 46px 18px;}
.df-empty b {display: block; color: var(--body-text-color); font-size: 1.05rem; margin-bottom: 6px;}
.df-steps {display: flex; justify-content: center; gap: 18px; flex-wrap: wrap; margin-top: 14px; font-size: 0.85rem;}
.df-steps span {display: inline-flex; align-items: center; gap: 6px;}
.df-steps i {font-style: normal; width: 20px; height: 20px; border-radius: 50%; display: inline-grid;
  place-items: center; background: var(--color-accent-soft); color: var(--color-accent); font-weight: 700; font-size: 0.75rem;}
.df-verdict {display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;}
.df-badge {display: inline-flex; align-items: center; gap: 10px; font-size: 1.9rem; font-weight: 800;
  letter-spacing: 0.02em; line-height: 1;}
.df-sub {color: var(--body-text-color-subdued); font-size: 0.88rem; margin-top: 6px;}
.df-pct {font-size: 2.2rem; font-weight: 750; font-variant-numeric: tabular-nums; text-align: right; line-height: 1;}
.df-pct small {display: block; font-size: 0.75rem; font-weight: 500; color: var(--body-text-color-subdued); margin-top: 4px;}
.df-gauge {position: relative; height: 12px; border-radius: 999px; margin: 18px 0 6px;
  background: linear-gradient(90deg, #16a34a 0%, #eab308 50%, #dc2626 100%); opacity: 0.9;}
.df-needle {position: absolute; top: -5px; width: 4px; height: 22px; border-radius: 2px;
  background: var(--body-text-color); transform: translateX(-50%); box-shadow: 0 0 0 2px var(--block-background-fill);}
.df-thr {position: absolute; top: -3px; height: 18px; border-left: 2px dashed var(--body-text-color-subdued);}
.df-scale {display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--body-text-color-subdued);}
.df-explain {margin-top: 12px; font-size: 0.94rem; line-height: 1.5;}
.df-truth {margin-top: 12px; padding: 9px 12px; border-radius: 10px; font-size: 0.88rem;
  border: 1px solid var(--border-color-primary);}
.df-truth.ok {background: rgba(22, 163, 74, 0.10); border-color: rgba(22, 163, 74, 0.4);}
.df-truth.bad {background: rgba(220, 38, 38, 0.08); border-color: rgba(220, 38, 38, 0.4);}
.df-section {font-weight: 650; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--body-text-color-subdued); margin: 0 0 8px;}
.df-row {display: grid; grid-template-columns: minmax(110px, 160px) 1fr 64px; align-items: center; gap: 12px;
  padding: 9px 0; border-bottom: 1px solid var(--border-color-primary); font-size: 0.92rem;}
.df-row:last-of-type {border-bottom: none;}
.df-row small {display: block; color: var(--body-text-color-subdued); font-size: 0.75rem; line-height: 1.3;}
.df-bar {height: 8px; border-radius: 999px; background: var(--border-color-primary); overflow: hidden;}
.df-bar > span {display: block; height: 100%; border-radius: 999px;}
.df-val {text-align: right; font-variant-numeric: tabular-nums; font-weight: 650;}
.df-na {color: var(--body-text-color-subdued); font-size: 0.85rem;}
.df-facts {display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-top: 14px;}
.df-fact {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 8px 10px;}
.df-fact span {display: block; font-size: 0.72rem; color: var(--body-text-color-subdued);}
.df-fact b {font-size: 0.92rem; overflow-wrap: anywhere;}
.df-warn {margin-top: 12px; padding: 10px 12px; border-radius: 10px; font-size: 0.88rem;
  background: rgba(234, 179, 8, 0.12); border: 1px solid rgba(234, 179, 8, 0.45);}
.df-warn ul {margin: 4px 0 0;} .df-warn li {margin-left: 16px;}
.df-tiles {display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px;}
.df-tile {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px;
  background: var(--block-background-fill);}
.df-tile span {display: block; font-size: 0.78rem; color: var(--body-text-color-subdued);}
.df-tile b {font-size: 1.4rem; font-variant-numeric: tabular-nums;}
.df-tile small {display: block; font-size: 0.74rem; color: var(--body-text-color-subdued); margin-top: 2px;}
.df-cm {border-collapse: separate !important; border-spacing: 6px; font-variant-numeric: tabular-nums;
  border: none !important; margin: 0 0 8px !important; width: auto !important;}
.df-cm tr, .df-cm td, .df-cm th {border: none !important; background-clip: padding-box;}
.df-cm td, .df-cm th {padding: 8px 12px; text-align: center; border-radius: 8px; font-size: 0.9rem;}
.df-cm th {font-weight: 600; color: var(--body-text-color-subdued); background: transparent !important;}
.df-ok {background: rgba(22, 163, 74, 0.16);} .df-bad {background: rgba(220, 38, 38, 0.16);}
.df-caption {font-size: 0.9rem; font-weight: 650; margin: 6px 0 6px;}
.df-note {font-size: 0.86rem; color: var(--body-text-color-subdued);}
.df-modelcard {display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px;}
.df-modelcard h3 {margin: 0 0 6px; font-size: 1.05rem;}
.df-modelcard dl {display: grid; grid-template-columns: 130px 1fr; gap: 6px 12px; margin: 10px 0 0; font-size: 0.88rem;}
.df-modelcard dt {color: var(--body-text-color-subdued);} .df-modelcard dd {margin: 0;}
.df-tag {display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 650;
  margin-left: 6px; vertical-align: middle;}
.df-tag.own {background: rgba(79, 70, 229, 0.14); color: #4f46e5;}
.df-tag.ext {background: rgba(8, 145, 178, 0.14); color: #0e7490;}
.df-views .wrap {gap: 6px !important;}
@media (max-width: 760px) {
  .tab-wrapper, .tab-container {min-width: 0 !important; max-width: 100% !important;}
  .tab-container {flex-wrap: wrap !important; height: auto !important; row-gap: 2px;}
  .df-stack {flex-direction: column !important; flex-wrap: nowrap !important;}
  .df-stack > * {min-width: 0 !important; width: 100% !important; flex: 1 1 auto !important;}
}
@media (max-width: 640px) {
  .df-hero h1 {font-size: 1.35rem;} .df-badge {font-size: 1.5rem;} .df-pct {font-size: 1.7rem;}
  .df-row {grid-template-columns: 96px 1fr 54px; gap: 8px;}
  .df-modelcard dl {grid-template-columns: 1fr;}
}
"""
VERDICT_COLORS = {FAKE: "#dc2626", REAL: "#16a34a", INCONCLUSIVE: "#d97706"}
VERDICT_ICONS = {FAKE: "⚠", REAL: "✔", INCONCLUSIVE: "?"}
KIND_NOUN = {"video": "video", "audio": "recording", "image": "image"}


def _kw(component, **kwargs) -> dict:
    """Keep only the keyword arguments this Gradio version's component accepts."""
    params = inspect.signature(component).parameters          # works for classes and functions
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        params = inspect.signature(component.__init__).parameters if inspect.isclass(component) else params
    return {k: v for k, v in kwargs.items() if k in params}


def _image(**kwargs):
    return gr.Image(**_kw(gr.Image, buttons=["fullscreen", "download"], show_download_button=True, **kwargs))


def _pct(p) -> str:
    return "n/a" if p is None else f"{100 * p:.1f}%"


def _score_color(p: float, threshold: float) -> str:
    return VERDICT_COLORS[FAKE] if p >= threshold else VERDICT_COLORS[REAL]


# --------------------------------------------------------------------------- ground truth of demo samples

def load_samples() -> list[dict]:
    path = SAMPLES_DIR / "samples.json"
    if not path.is_file():
        return []
    return [{**s, "abs": str(SAMPLES_DIR / s["path"])} for s in json.loads(path.read_text())
            if (SAMPLES_DIR / s["path"]).is_file()]


def sample_truth(path: str | Path, samples: list[dict]) -> dict | None:
    """The demo sample a (possibly copied) file is, matched by file name and size."""
    p = Path(path)
    for s in samples:
        src = Path(s["abs"])
        if p.name == src.name and p.stat().st_size == src.stat().st_size:
            return s
    return None


# --------------------------------------------------------------------------- result cards

def empty_html(kind: str) -> str:
    what = {"video": "a video", "audio": "an audio file or a recording", "image": "a photo with a face"}[kind]
    return (f'<div class="df-card df-empty"><b>No analysis yet</b>Upload {what}, or pick one of the labelled '
            f'samples, then press <i>Analyse</i>.<div class="df-steps"><span><i>1</i>Choose a file</span>'
            f'<span><i>2</i>Analyse</span><span><i>3</i>Read the verdict and the evidence</span></div></div>')


def _interpretation(res: dict) -> str:
    p, t, noun = res["p_fake"], res["threshold"], KIND_NOUN[res["kind"]]
    if p is None:
        return "Nothing in this file could be judged by the loaded models; see the notes below."
    by = res["decided_by"]
    if by == "face + voice (max rule)":
        part = "face" if res["face"]["p_fake"] >= res["voice"]["p_fake"] else "voice"
        source = f"The {part} is the more suspicious part, so it sets the score (a {noun} is fake if its face <b>or</b> its voice is fake)."
    else:
        source = f"Decided by the {by}."
    side = ("at or above" if res["verdict"] == FAKE else "below")
    return f"Rated <b>{_pct(p)} likely fake</b>, {side} the {_pct(t)} decision threshold. {source}"


def verdict_html(res: dict | None, kind: str = "video", truth: dict | None = None) -> str:
    if res is None:
        return empty_html(kind)
    v = res["verdict"]
    color, p, t = VERDICT_COLORS[v], res["p_fake"], res["threshold"]
    needle = "" if p is None else f'<div class="df-needle" style="left:{100 * p:.1f}%"></div>'
    gauge = (f'<div class="df-gauge"><div class="df-thr" style="left:{100 * t:.1f}%" title="threshold"></div>{needle}</div>'
             '<div class="df-scale"><span>0% · real</span><span>dashed line = threshold</span><span>fake · 100%</span></div>')
    label = {FAKE: "Likely manipulated", REAL: "No manipulation detected", INCONCLUSIVE: "Could not be judged"}[v]
    truth_html = ""
    if truth is not None:
        ok = v == truth["label"].upper()
        truth_html = (f'<div class="df-truth {"ok" if ok else "bad"}"><b>Demo sample · ground truth: '
                      f'{truth["label"].upper()}</b> ({html.escape(truth["category"])}). '
                      f'{"The model is right." if ok else "The model gets this one wrong."} '
                      f'<span class="df-note">{html.escape(truth["description"])}</span></div>')
    return (f'<div class="df-card"><div class="df-verdict"><div>'
            f'<div class="df-badge" style="color:{color}">{VERDICT_ICONS[v]} {v}</div>'
            f'<div class="df-sub">{label}</div></div>'
            f'<div class="df-pct" style="color:{color}">{_pct(p)}<small>probability of fake</small></div></div>'
            f'{gauge}<div class="df-explain">{_interpretation(res)}</div>{truth_html}</div>')


def _branch_row(name: str, sub: str, branch: dict | None, loaded: bool, t: float, missing: str) -> str:
    if not loaded:
        return (f'<div class="df-row"><span>{name}<small>not loaded</small></span>'
                f'<span class="df-na">no model for this part</span><span class="df-val df-na">–</span></div>')
    if branch is None or branch["p_fake"] is None:
        return (f'<div class="df-row"><span>{name}<small>{html.escape(sub)}</small></span>'
                f'<span class="df-na">{missing}</span><span class="df-val df-na">n/a</span></div>')
    p = branch["p_fake"]
    return (f'<div class="df-row"><span>{name}<small>{html.escape(sub)}</small></span><div class="df-bar">'
            f'<span style="width:{100 * p:.1f}%;background:{_score_color(p, t)}"></span></div>'
            f'<span class="df-val">{_pct(p)}</span></div>')


def evidence_html(res: dict | None, caps: dict) -> str:
    if res is None:
        return ""
    t, info, kind = res["threshold"], res["info"], res["kind"]
    rows = []
    if kind in ("video", "image"):
        face = res["face"]
        sub = f"{face['faces_found']} / {face['frames_sampled']} frames with a face" if face and kind == "video" else "most prominent face"
        rows.append(_branch_row("Face", sub, face, bool(caps["face_model"]), t, "no face found"))
    if kind in ("video", "audio"):
        voice = res["voice"]
        sub = f"{len(voice['windows'])} windows of 4 s" if voice and voice["windows"] else "speech"
        rows.append(_branch_row("Voice", sub, voice, bool(caps["voice_model"]), t,
                                "no audio track" if kind == "video" else "no usable audio"))
    facts = [("File", html.escape(str(info.get("file", ""))))]
    if info.get("duration"):
        facts.append(("Length", f'{info["duration"]:.1f} s'))
    if info.get("resolution"):
        facts.append(("Resolution", info["resolution"]))
    if info.get("audio_seconds"):
        facts.append(("Audio analysed", f'{info["audio_seconds"]:.1f} s'))
    facts += [("Decided by", html.escape(res["decided_by"])), ("Threshold", _pct(t))]
    fact_html = "".join(f'<div class="df-fact"><span>{k}</span><b>{v}</b></div>' for k, v in facts)
    warn = ""
    if res["warnings"]:
        warn = '<div class="df-warn"><b>Notes</b><ul>' + "".join(
            f"<li>{html.escape(w)}</li>" for w in res["warnings"]) + "</ul></div>"
    return (f'<div class="df-card"><div class="df-section">Evidence by model</div>{"".join(rows)}'
            f'<div class="df-facts">{fact_html}</div>{warn}</div>')


def format_details(res: dict) -> str:
    """Plain Markdown summary of a result (shown under 'Full result')."""
    lines = [f"**{res['verdict']}** · P(fake) {_pct(res['p_fake'])} · threshold {res['threshold']:.2f} · "
             f"decided by {res['decided_by']}", ""]
    for key in ("face", "voice"):
        b = res[key]
        if b:
            lines.append(f"- {key.title()} ({b['model']}): {_pct(b['p_fake'])}")
    lines += [f"- ⚠️ {w}" for w in res["warnings"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------- handlers

def make_handler(analyzer: Analyzer, kind: str, samples: list[dict]):
    """analyse(file, threshold) -> updates for the result widgets of one tab."""
    caps = analyzer.capabilities()

    def analyse(path, threshold, rule="max", progress=gr.Progress()):
        if not path:
            raise gr.Error({"video": "Upload a video or pick a sample first.",
                            "audio": "Upload or record audio, or pick a sample first.",
                            "image": "Upload a photo or pick a sample first."}[kind])
        progress(0.1, desc="Reading the file and finding faces" if kind != "audio" else "Reading the audio")
        try:
            res = analyzer.analyse(path, threshold=float(threshold), video_rule=rule or "max")
        except (ValueError, OSError) as e:
            raise gr.Error(f"Could not analyse this file: {e}")
        progress(1.0, desc="Done")
        truth = sample_truth(path, samples)
        face, voice = res["face"] or {}, res["voice"] or {}
        overlays = face.get("overlays") or []
        spec = voice.get("spectrogram")
        timeline = res.get("timeline")
        out = [verdict_html(res, kind, truth), evidence_html(res, caps)]
        if kind != "image":
            out.append(gr.update(value=timeline, visible=timeline is not None))
        if kind != "audio":
            out.append(gr.update(value=overlays or None, visible=bool(overlays)))
        if kind == "image":
            out.append(gr.update(value=face.get("annotated"), visible=face.get("annotated") is not None))
        if kind != "image":
            out.append(gr.update(value=spec, visible=spec is not None))
        out += [summary(res), format_details(res)]
        return out
    return analyse


def run_all_samples(analyzer: Analyzer, samples: list[dict], threshold: float, progress=gr.Progress()):
    rows = []
    for i, s in enumerate(progress.tqdm(samples, desc="Analysing samples")):
        res = analyzer.analyse(s["abs"], threshold=float(threshold), with_explanation=False)
        rows.append({"Type": s["kind"], "Sample": Path(s["path"]).name, "Truth": s["label"].upper(),
                     "Category": s["category"],
                     "Face": _pct(res["face"]["p_fake"]) if res["face"] else "–",
                     "Voice": _pct(res["voice"]["p_fake"]) if res["voice"] else "–",
                     "P(fake)": _pct(res["p_fake"]), "Verdict": res["verdict"],
                     "Correct": "✓" if res["verdict"] == s["label"].upper() else "✗"})
    df = pd.DataFrame(rows)
    n_ok = (df["Correct"] == "✓").sum() if len(df) else 0
    return df, f"**{n_ok} of {len(df)} samples judged correctly** at threshold {float(threshold):.2f}."


# --------------------------------------------------------------------------- header + model pages

def header_html(caps: dict, run: dict | None) -> str:
    chips = []
    for label, name in (("Face model", caps["face_model"]), ("Voice model", caps["voice_model"])):
        state = "df-on" if name else "df-off"
        chips.append(f'<span class="df-chip"><span class="df-dot {state}"></span>{label}: '
                     f'{"loaded" if name else "not loaded"}</span>')
    m = (run or {}).get("metrics") or {}
    if m.get("auc") is not None:
        chips.append(f'<span class="df-chip">Voice test AUC {m["auc"]:.3f} · EER {100 * m["eer"]:.1f}%</span>')
    return (f'<div class="df-hero"><div><h1>{TITLE}</h1><p>Checks videos, voice recordings and photos for '
            f'face swaps and synthetic speech, and shows the evidence behind each decision. A research '
            f'prototype: scores are model estimates, not forensic proof.</p></div>'
            f'<div class="df-chips">{"".join(chips)}</div></div>')


def model_cards_html(analyzer: Analyzer, run: dict | None) -> str:
    cards = []
    if analyzer.face_source == "pretrained":
        cards.append(
            '<div class="df-card"><h3>Face model <span class="df-tag ext">pretrained, external</span></h3>'
            f'<div class="df-note">{FaceSwapDetector.name}. Used until this project\'s own face model is '
            'trained on FakeAVCeleb.</div><dl>'
            '<dt>Architecture</dt><dd>EfficientNet-B7 (Noisy Student), 1 output, 380 × 380 input</dd>'
            '<dt>Trained on</dt><dd>DFDC training set (over 100,000 clips of 3,426 consenting paid actors)</dd>'
            '<dt>Pedigree</dt><dd>Winner of the Kaggle Deepfake Detection Challenge (2020). This is one of the '
            '7 networks of the winning ensemble, so it is a little weaker than the full entry</dd>'
            '<dt>Input here</dt><dd>YuNet face box + ⅓ margin per side, 16 frames per video</dd>'
            '<dt>Licence</dt><dd>MIT, github.com/selimsef/dfdc_deepfake_challenge</dd>'
            '<dt>Blind spots</dt><dd>Fully AI-generated faces (GAN / diffusion), lip-sync-only edits, '
            'tiny or heavily compressed faces</dd></dl></div>')
    elif analyzer.face_source == "project":
        cards.append(f'<div class="df-card"><h3>Face model <span class="df-tag own">this project</span></h3>'
                     f'<div class="df-note">{html.escape(analyzer.face_name)}</div></div>')
    if run is not None:
        m = run["metrics"] or {}
        perf = (f'AUC {m["auc"]:.3f} · EER {100 * m["eer"]:.1f}% on {m["n"]:,} held-out clips '
                f'(official evaluation partition, unseen attacks A07–A19)') if m else "no test metrics saved"
        cards.append(
            '<div class="df-card"><h3>Voice model <span class="df-tag own">this project</span></h3>'
            f'<div class="df-note">{html.escape(run["branch"])}, run <code>{html.escape(run["name"])}</code></div><dl>'
            f'<dt>Trained on</dt><dd>{html.escape(run["dataset"])}</dd>'
            f'<dt>Test result</dt><dd>{perf}</dd>'
            '<dt>Input here</dt><dd>16 kHz mono, 128-bin log-mel, 4 s windows every 2 s over the first 30 s; '
            'P(fake) is the window mean</dd>'
            '<dt>Blind spots</dt><dd>Studio-quality training data: phone microphones, rooms, music and heavy '
            'compression shift its scores; some attack types (A17) are often missed</dd></dl></div>')
    return f'<div class="df-modelcard">{"".join(cards)}</div>'


def metric_tiles(run: dict) -> str:
    m = run["metrics"]
    if not m:
        return '<div class="df-card df-empty"><b>No test metrics saved for this run</b></div>'
    eer_thr = m.get("at_eer_threshold", {}).get("accuracy")
    tiles = [("Test AUC", f'{m["auc"]:.3f}', "ranking quality, 1.0 = perfect"),
             ("EER", f'{100 * m["eer"]:.1f}%', "equal error rate, lower is better"),
             ("Accuracy", f'{100 * m["accuracy"]:.1f}%', "at threshold 0.5"),
             ("Precision", f'{m["precision"]:.3f}', "of clips flagged fake"),
             ("Recall", f'{m["recall"]:.3f}', "share of fakes caught"),
             ("F1", f'{m["f1"]:.3f}', "at threshold 0.5")]
    if eer_thr is not None:
        tiles.append(("Accuracy at EER threshold", f"{100 * eer_thr:.1f}%", f'threshold {m.get("eer_threshold", 0):.4f}'))
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
    lines = ["# 1. Download, index and split the dataset (notebook §3 and §4)",
             f"python -m deepfake.data.download {dataset}",
             f"python -m deepfake.data.manifest {dataset}",
             f"python -m deepfake.data.splits {dataset}",
             f"python -m deepfake.preprocessing.run {dataset}" + (" --per-class 2500" if dataset == "asvspoof" else ""),
             "", f"# 2. Train ({run['branch']})",
             f"python -m deepfake.training.{script} \\\n    " + " \\\n    ".join(args) + " \\\n    --device cuda",
             "", "# 3. Test-set metrics, per-category table and ROC are written to <run>/test/ automatically.",
             "#    Build the results report:", "python -m deepfake.report"]
    return "\n".join(lines)


def code_views(run: dict | None) -> dict[str, tuple[str, str]]:
    """Name -> (language, code) for the Code tab; sources are read from the installed package."""
    from deepfake import analyzer as analyzer_mod
    from deepfake.face_model import DFDCClassifier, confident_mean
    from deepfake.models.audio import AudioModel
    from deepfake.models.fusion import late_combine
    from deepfake.preprocessing.audio import log_mel
    from deepfake.training.engine import train_one_epoch

    views = {}
    if run is not None:
        views["How the voice model was trained"] = ("shell", training_command(run))
    views["Use the analyser from Python"] = ("python", PYTHON_USAGE)
    for name, obj in (("Analyser: video / audio / image", analyzer_mod.Analyzer.analyse),
                      ("Voice model: AudioModel", AudioModel), ("Audio preprocessing: log-mel", log_mel),
                      ("Face model: DFDC EfficientNet-B7", DFDCClassifier),
                      ("Video score from frame scores", confident_mean), ("Late-fusion rules", late_combine),
                      ("Training loop: one epoch", train_one_epoch)):
        views[name] = ("python", inspect.getsource(obj))
    return views


PYTHON_USAGE = '''from deepfake.analyzer import Analyzer

analyzer = Analyzer.from_defaults()          # voice model from models/, face model if downloaded
res = analyzer.analyse("clip.mp4")           # or .wav / .mp3 / .jpg / .png
print(res["verdict"], res["p_fake"], res["decided_by"])
print("face:", res["face"]["p_fake"], "voice:", res["voice"]["p_fake"])

# Command line: JSON summary + timeline and Grad-CAM images
# python -m deepfake.analyze clip.mp4 --out result/
# python -m deepfake.analyze --samples        # all labelled demo clips
'''

ABOUT = """
### How it works
| Input | What happens |
|---|---|
| **Video** | 16 evenly spaced frames → YuNet face detector → face model on every face. The soundtrack (first 30 s) → 16 kHz log-mel spectrogram → voice model on 4-second windows. The video is fake if the face **or** the voice is fake (max rule), unless a trained fusion model is loaded. |
| **Audio** | Voice model on overlapping 4-second windows; the score is the mean over windows. |
| **Image** | The most prominent face is cropped and scored by the face model. |

**Evidence.** Grad-CAM heatmaps show which face regions and which time–frequency regions pushed
the decision. The timeline shows the score of every frame and every audio window, so a
manipulation in part of a clip stands out.

### Reading the result
- **P(fake)** is the model's probability that the input is manipulated. The **threshold** sets the
  verdict: lower it to catch more fakes at the cost of more false alarms.
- **INCONCLUSIVE** means nothing could be judged, e.g. no face in a photo, or a silent clip.
- Demo samples show their **ground truth** next to the verdict, including the ones the models get wrong.
- Heatmaps show *where* the model looked, not proof of manipulation.

### Limitations
- The voice model was trained on ASVspoof 2019 LA studio recordings. New microphones, rooms,
  background music or compression change its scores, and it misses some unseen attack types.
- The face model detects **face swaps** of the DFDC kind. It is not built for fully AI-generated
  faces (GAN, diffusion) or lip-sync-only edits.
- This project's own face and fusion models are pending FakeAVCeleb access; the face model shown here
  is the external DFDC winner, credited in the *Models* tab.

### Sources
Demo clips: DFDC preview set (Dolhansky et al., 2019; consenting paid actors) and the VITS paper's
audio samples (Kim et al., 2021; LJ Speech, public domain). Training data: ASVspoof 2019 LA (Wang et al.,
2020). See `demo_samples/README.md`, `DATASETS.md` and `reports/report.md` in the repository.
"""


# --------------------------------------------------------------------------- layout

def _examples(component, samples: list[dict], kind: str):
    items = [s for s in samples if s["kind"] == kind]
    if not items:
        return
    labels = [f'{"✔ REAL" if s["label"] == "real" else "⚠ FAKE"} · {s["category"]}' for s in items]
    gr.Examples(**_kw(gr.Examples, examples=[[s["abs"]] for s in items], inputs=[component],
                      label="Labelled samples: click one to load it", example_labels=labels,
                      examples_per_page=8, cache_examples=False))


def _analysis_tab(analyzer: Analyzer, kind: str, samples: list[dict], threshold_default: float):
    """Builds one input/result tab and wires its button. Returns nothing; all state is local."""
    with gr.Row(equal_height=False, elem_classes="df-stack"):
        with gr.Column(scale=5, min_width=300):
            if kind == "video":
                inp = gr.Video(label="Video", sources=["upload"], height=300)
            elif kind == "audio":
                inp = gr.Audio(label="Audio file or microphone recording", sources=["upload", "microphone"],
                               type="filepath")
            else:
                inp = gr.Image(label="Photo", sources=["upload", "clipboard"], type="filepath", height=300)
            _examples(inp, samples, kind)
            with gr.Accordion("Decision settings", open=False):
                threshold = gr.Slider(0.05, 0.95, value=threshold_default, step=0.01, label="Threshold",
                                      info="P(fake) at or above this value counts as FAKE.")
                inputs = [inp, threshold]
                if kind == "video" and analyzer.face_source and analyzer.voice_source:
                    rule = gr.Radio([(label.capitalize(), key) for key, label in VIDEO_RULES.items()],
                                    value="max", label="The video verdict uses",
                                    info="Both scores are always shown. 'Face only' ignores the voice model, "
                                         "which over-flags real speech recorded outside its training data.")
                    inputs.append(rule)
            with gr.Row():
                run_btn = gr.Button("Analyse", variant="primary", scale=3)
                clear_btn = gr.Button("Clear", scale=1)
        with gr.Column(scale=7, min_width=300):
            verdict = gr.HTML(empty_html(kind))
            evidence = gr.HTML()

    outputs = [verdict, evidence]
    if kind != "image":
        gr.HTML('<div class="df-caption">Score over time</div><div class="df-note">Each dot is a sampled '
                'frame (face) or a 4-second window (voice). Points above the dashed line count as fake.</div>')
        timeline = _image(show_label=False, visible=False, interactive=False)
        outputs.append(timeline)
    if kind != "audio":
        gr.HTML('<div class="df-caption">Face evidence (Grad-CAM)</div><div class="df-note">Red marks the face '
                'regions that most influenced the decision.</div>')
        with gr.Row(equal_height=False, elem_classes="df-stack"):
            if kind == "image":
                annotated = _image(label="Detected face", visible=False, interactive=False, height=340)
            faces = gr.Gallery(**_kw(gr.Gallery, label="Grad-CAM on the face", columns=4 if kind == "video" else 1,
                                     height=300 if kind == "video" else 340, visible=False,
                                     object_fit="contain", allow_preview=True))
        outputs.append(faces)
        if kind == "image":
            outputs.append(annotated)
    if kind != "image":
        gr.HTML('<div class="df-caption">Voice evidence (Grad-CAM)</div><div class="df-note">Time runs left '
                'to right, pitch rises upwards; red marks what drove the voice score.</div>')
        spectrogram = _image(show_label=False, visible=False, interactive=False)
        outputs.append(spectrogram)
    with gr.Accordion("Full result (JSON)", open=False):
        raw = gr.JSON(label="Result")
        details = gr.Markdown()
    outputs += [raw, details]

    run_btn.click(make_handler(analyzer, kind, samples), inputs, outputs)
    resets = [None, empty_html(kind), ""] + [gr.update(value=None, visible=False)] * (len(outputs) - 4) + [None, ""]
    clear_btn.click(lambda: resets, None, [inp, *outputs])


def create_demo(analyzer: Analyzer) -> gr.Blocks:
    caps = analyzer.capabilities()
    project = analyzer.project
    run = model_card.load_run(project.checkpoint, project.run_config) if project else None
    samples = load_samples()
    views = code_views(run)
    first_view = next(iter(views))

    blocks_kw = {"title": TITLE}
    if GRADIO_MAJOR < 6:                                # Gradio 6 takes theme and css in launch()
        blocks_kw.update(theme=THEME, css=CSS)
    with gr.Blocks(**blocks_kw) as demo:
        gr.HTML(header_html(caps, run))
        with gr.Tabs():
            with gr.Tab("🎬 Video"):
                _analysis_tab(analyzer, "video", samples, analyzer.threshold)
            with gr.Tab("🎙️ Audio"):
                _analysis_tab(analyzer, "audio", samples, analyzer.threshold)
            with gr.Tab("🖼️ Image"):
                _analysis_tab(analyzer, "image", samples, analyzer.threshold)

            with gr.Tab("🧪 Samples"):
                gr.Markdown("All labelled demo clips, analysed in one go and compared with their ground truth. "
                            "On a CPU this takes about a minute. Sources and licences: `demo_samples/README.md`.")
                with gr.Row():
                    s_thr = gr.Slider(0.05, 0.95, value=analyzer.threshold, step=0.01, label="Threshold", scale=3)
                    s_btn = gr.Button("Analyse all samples", variant="primary", scale=1)
                s_score = gr.Markdown()
                s_table = gr.Dataframe(**_kw(gr.Dataframe, interactive=False, wrap=True))
                s_btn.click(lambda thr, progress=gr.Progress(): run_all_samples(analyzer, samples, thr, progress),
                            [s_thr], [s_table, s_score])

            with gr.Tab("📊 Models"):
                gr.HTML(model_cards_html(analyzer, run))
                if run is not None:
                    gr.Markdown(f"### Voice model in detail\nRun `{run['name']}` · "
                                + ((run["data"] or {}).get("description") or f"trained on **{run['dataset']}**"))
                    gr.HTML(metric_tiles(run))
                    with gr.Row(equal_height=False, elem_classes="df-stack"):
                        with gr.Column(scale=1, min_width=300):
                            gr.Markdown("#### Data")
                            gr.Dataframe(**_kw(gr.Dataframe, value=model_card.split_table(run), interactive=False,
                                               wrap=True, max_height=260))
                            gr.HTML(confusion_html(run))
                        with gr.Column(scale=1, min_width=300):
                            gr.Markdown("#### Training settings")
                            gr.Dataframe(**_kw(gr.Dataframe, value=model_card.config_table(run), interactive=False,
                                               wrap=True, max_height=640, column_widths=["42%", "58%"]))
                    cats = model_card.category_table(run)
                    if cats is not None:
                        gr.Markdown("#### Results per attack system" + (
                            " (ASVspoof A07–A19 and bona fide speech)" if "asvspoof" in run["dataset"].lower() else ""))
                        gr.Dataframe(**_kw(gr.Dataframe, value=cats, interactive=False, wrap=True, max_height=720))
                    with gr.Row(equal_height=False, elem_classes="df-stack"):
                        if run["curves"]:
                            with gr.Column(scale=1, min_width=300):
                                gr.HTML('<div class="df-caption">Training curves</div>')
                                _image(value=run["curves"], show_label=False, interactive=False, height=340)
                        if run["roc"]:
                            with gr.Column(scale=1, min_width=300):
                                gr.HTML('<div class="df-caption">ROC curve on the test set</div>')
                                _image(value=run["roc"], show_label=False, interactive=False, height=340)

            with gr.Tab("💻 Code"):
                gr.Markdown("The command that trained the voice model and the source of the main parts, read "
                            "from the installed `deepfake` package.")
                view = gr.Radio(list(views), value=first_view, label="Show", elem_classes="df-views")
                code = gr.Code(views[first_view][1], language=views[first_view][0], interactive=False)
                view.change(lambda name: gr.update(value=views[name][1], language=views[name][0]), view, code)

            with gr.Tab("ℹ️ About"):
                gr.Markdown(ABOUT)
    return demo


def launch(demo: gr.Blocks, **kwargs):
    if GRADIO_MAJOR >= 6:
        kwargs.update(theme=THEME, css=CSS)
    return demo.launch(**kwargs)
