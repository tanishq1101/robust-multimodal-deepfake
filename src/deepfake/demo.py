"""Gradio demo UI. Launch with app/app.py.

Tabs:
  Video      upload a clip or pick a labelled sample: verdict, face and voice evidence, timeline,
             Grad-CAM on faces and on the spectrogram, downloadable JSON report
  Audio      upload or record speech: verdict, score per 4 s window, spectrogram Grad-CAM
  Photo      upload a photo: verdict, detected face, Grad-CAM on the face
  Benchmark  every labelled demo clip analysed in one go, against its ground truth
  Models     what each loaded model is, how it was trained and how well it did
  Code       usage, the key source files (without comments) and the training commands, one block each
  Method     pipeline, how to read a result, limitations, sources

Styling lives in deepfake.demo_theme. Works with Gradio 5 and 6: keyword arguments a version does
not know are dropped.
"""

from __future__ import annotations

import html
import inspect
import json
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from deepfake import model_card
from deepfake.analyze import SAMPLES_DIR, summary
from deepfake.analyzer import BORDERLINE, FAKE, INCONCLUSIVE, REAL, VIDEO_RULES, Analyzer
from deepfake.code_view import snippets
from deepfake.demo_theme import COLORS, CSS, HEAD, LOGO, THEME
from deepfake.face_model import FaceSwapDetector

TITLE = "Robust Multimodal Deepfake Analysis"
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
VERDICT_COLORS = {FAKE: COLORS["fake"], REAL: COLORS["real"], INCONCLUSIVE: COLORS["unsure"]}
VERDICT_LABELS = {FAKE: "Likely manipulated", REAL: "No manipulation found", INCONCLUSIVE: "Could not be judged"}
KIND_NOUN = {"video": "video", "audio": "recording", "image": "photo"}
TAB_KIND = {"video": "Video", "audio": "Audio", "image": "Photo"}


def _kw(component, **kwargs) -> dict:
    """Keep only the keyword arguments this Gradio version's component accepts."""
    params = inspect.signature(component).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        params = inspect.signature(component.__init__).parameters if inspect.isclass(component) else params
    return {k: v for k, v in kwargs.items() if k in params}


def _image(**kwargs):
    return gr.Image(**_kw(gr.Image, buttons=["fullscreen", "download"], show_download_button=True, **kwargs))


def _pct(p) -> str:
    return "n/a" if p is None else f"{100 * p:.1f}%"


def _esc(x) -> str:
    return html.escape(str(x))


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
    return (f'<div class="wm-card wm-empty">{LOGO}<b>No analysis yet</b>'
            f'<p>Upload {what}, or pick one of the labelled samples on the left, then press Analyse.</p>'
            '<div class="wm-steps"><div><em>Step 1</em>Choose a file or a sample</div>'
            '<div><em>Step 2</em>Press Analyse</div><div><em>Step 3</em>Read the verdict, then the evidence</div>'
            '</div></div>')


def strength(p: float | None, t: float) -> tuple[str, str]:
    """How far the score sits from the threshold, as a (label, tag colour class)."""
    if p is None:
        return "No score", "n"
    d = abs(p - t)
    if d >= 0.35:
        return "Strong evidence", "r" if p >= t else "g"
    if d >= BORDERLINE:
        return "Moderate evidence", "r" if p >= t else "g"
    return "Borderline: close to the threshold", "y"


def reasons(res: dict) -> list[str]:
    """Plain sentences that explain the verdict, one per line."""
    p, t, noun = res["p_fake"], res["threshold"], KIND_NOUN[res["kind"]]
    if p is None:
        return ["Nothing in this file could be judged by the loaded models. See the notes in the evidence card."]
    side = "at or above" if res["verdict"] == FAKE else "below"
    out = [f"The {noun} is rated {_pct(p)} likely fake, {side} the {_pct(t)} decision threshold."]
    by = res["decided_by"]
    face, voice = res["face"] or {}, res["voice"] or {}
    if by == "face + voice":
        same = "fake" if res["verdict"] == FAKE else "real"
        out.append(f"Face and voice both point to {same}; the higher of the two scores is reported.")
    elif "borderline" in by:
        strong, weak = by.split()[0], by.split("(")[1].split()[0]
        out.append(f"The {weak} score is borderline (within {BORDERLINE:.0%} of the threshold), so it does not "
                   f"overrule the confident {strong} score.")
    elif "(face + voice)" in by:
        part = by.split()[0]
        out.append(f"The {part} is clearly suspicious while the other part looks real. A {noun} counts as fake "
                   f"if its face or its voice is fake.")
    elif by == "fusion model":
        out.append("The trained face + voice fusion model produced the score.")
    else:
        out.append(f"Decided by the {by} model alone.")
    if face.get("p_fake") is not None and res["kind"] == "video":
        out.append(f"Face: {_pct(face['p_fake'])} over {face['faces_found']} of {face['frames_sampled']} frames.")
    if voice.get("p_fake") is not None:
        n = voice.get("speech_windows", len(voice.get("windows", [])))
        out.append(f"Voice: {_pct(voice['p_fake'])} over {n} speech window{'s' if n != 1 else ''} of 4 s.")
    return out


def _meter(p: float | None, t: float) -> str:
    cells = []
    for i in range(20):
        lo = i / 20
        lit = p is not None and lo < p
        cells.append(f'<span class="{("lit-r" if lo >= t else "lit-g") if lit else ""}"></span>')
    mark = "" if p is None else f'<div class="wm-mark" style="left:{100 * p:.1f}%" title="score"></div>'
    return (f'<div class="wm-meter"><div class="wm-track">{"".join(cells)}'
            f'<div class="wm-thr" style="left:{100 * t:.1f}%" title="threshold"></div>{mark}</div>'
            f'<div class="wm-scale"><span>0% real</span><span>dashed line: threshold {_pct(t)}</span>'
            f'<span>100% fake</span></div></div>')


def verdict_html(res: dict | None, kind: str = "video", truth: dict | None = None) -> str:
    if res is None:
        return empty_html(kind)
    v, p, t = res["verdict"], res["p_fake"], res["threshold"]
    color = VERDICT_COLORS[v]
    label, cls = strength(p, t)
    tags = [f'<span class="wm-tag {cls}">{label}</span>']
    face, voice = (res["face"] or {}).get("p_fake"), (res["voice"] or {}).get("p_fake")
    if face is not None and voice is not None:
        agree = (face >= t) == (voice >= t)
        tags.append(f'<span class="wm-tag {"g" if agree else "y"}">'
                    f'{"Face and voice agree" if agree else "Face and voice disagree"}</span>')
    tags.append(f'<span class="wm-tag n">Decided by: {_esc(res["decided_by"])}</span>')
    truth_html = ""
    if truth is not None:
        ok = v == truth["label"].upper()
        truth_html = (f'<div class="wm-truth {"ok" if ok else "bad"}"><b>Labelled sample: truly '
                      f'{truth["label"].upper()} ({_esc(truth["category"])}). '
                      f'{"The verdict is correct." if ok else "The verdict is wrong for this one."}</b>'
                      f'{_esc(truth["description"])}</div>')
    why = "".join(f"<li>{r}</li>" for r in reasons(res))
    return (f'<div class="wm-card"><div class="wm-kicker"><i>V</i>Verdict</div>'
            f'<div class="wm-verdict"><div><div class="wm-word" style="color:{color}">{v}</div>'
            f'<div class="wm-label">{VERDICT_LABELS[v]}</div></div>'
            f'<div class="wm-num" style="color:{color}">{_pct(p)}<small>chance of fake</small></div></div>'
            f'<div class="wm-tags">{"".join(tags)}</div>{_meter(p, t)}<ul class="wm-why">{why}</ul>{truth_html}</div>')


def _branch_row(name: str, sub: str, branch: dict | None, loaded: bool, t: float, missing: str) -> str:
    if not loaded:
        return (f'<div class="wm-row"><span><strong>{name}</strong><small>model not loaded</small></span>'
                f'<span class="wm-na">no model for this part</span><span class="wm-val wm-na">–</span></div>')
    if branch is None or branch["p_fake"] is None:
        return (f'<div class="wm-row"><span><strong>{name}</strong><small>{_esc(sub)}</small></span>'
                f'<span class="wm-na">{missing}</span><span class="wm-val wm-na">n/a</span></div>')
    p = branch["p_fake"]
    color = COLORS["fake"] if p >= t else COLORS["real"]
    return (f'<div class="wm-row"><span><strong>{name}</strong><small>{_esc(sub)}</small></span>'
            f'<div class="wm-bar"><span style="width:{100 * p:.1f}%;background:{color}"></span>'
            f'<i style="left:{100 * t:.1f}%"></i></div><span class="wm-val" style="color:{color}">{_pct(p)}</span></div>')


def evidence_html(res: dict | None, caps: dict) -> str:
    if res is None:
        return ""
    t, info, kind = res["threshold"], res["info"], res["kind"]
    rows = []
    if kind in ("video", "image"):
        face = res["face"]
        sub = (f"{face['faces_found']} of {face['frames_sampled']} frames had a face" if face and kind == "video"
               else "most prominent face, plus its mirror image")
        rows.append(_branch_row("Face", sub, face, bool(caps["face_model"]), t, "no face found"))
    if kind in ("video", "audio"):
        voice = res["voice"]
        if voice and voice["windows"]:
            sub = f"{voice.get('speech_windows', len(voice['windows']))} of {len(voice['windows'])} windows contain speech"
        else:
            sub = "speech"
        rows.append(_branch_row("Voice", sub, voice, bool(caps["voice_model"]), t,
                                "no audio track" if kind == "video" else "no usable audio"))
    facts = [("File", _esc(info.get("file", "")))]
    if info.get("duration"):
        facts.append(("Length", f'{info["duration"]:.1f} s'))
    if info.get("resolution"):
        facts.append(("Resolution", info["resolution"]))
    if info.get("audio_seconds"):
        facts.append(("Audio analysed", f'{info["audio_seconds"]:.1f} s'))
    facts.append(("Threshold", _pct(t)))
    dl = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in facts)
    notes = ""
    if res["warnings"]:
        notes = ('<div class="wm-notes"><b>Notes</b><ol>'
                 + "".join(f"<li>{_esc(w)}</li>" for w in res["warnings"]) + "</ol></div>")
    return (f'<div class="wm-card"><div class="wm-kicker"><i>E</i>Evidence by model</div>{"".join(rows)}'
            f'<dl class="wm-dl">{dl}</dl>{notes}</div>')


def write_report(res: dict, source: str | Path) -> str:
    """The result as a JSON file the user can download; returns its path."""
    stem = Path(source).stem[:40] or "result"
    f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", prefix=f"{stem}_deepfake_report_")
    with f:
        json.dump(summary(res), f, indent=2)
    return f.name


# --------------------------------------------------------------------------- handlers

def make_handler(analyzer: Analyzer, kind: str, samples: list[dict]):
    """analyse(file, threshold[, rule]) -> updates for the result widgets of one tab.

    Output order: verdict, evidence, [timeline], [face gallery], [annotated photo], [spectrogram], json, report file.
    """
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
        out += [summary(res), gr.update(value=write_report(res, path), visible=True)]
        return out
    return analyse


def table_html(df: pd.DataFrame, numeric: tuple[str, ...] = (), html_cols: tuple[str, ...] = ()) -> str:
    """A DataFrame as a clean HTML table; `html_cols` are inserted unescaped."""
    head = "".join(f'<th class="{"num" if c in numeric else ""}">{_esc(c)}</th>' for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            text = str(v) if c in html_cols else _esc("" if v is None else v)
            cells.append(f'<td class="{"num" if c in numeric else ""}">{text}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (f'<div class="wm-table-wrap"><table class="wm-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _chip(verdict: str) -> str:
    return f'<span class="wm-chip {verdict}">{verdict}</span>'


def run_all_samples(analyzer: Analyzer, samples: list[dict], threshold: float, progress=gr.Progress()):
    """Every labelled sample -> (results table HTML, summary tiles HTML)."""
    rows, ok_by_kind = [], {}
    for s in progress.tqdm(samples, desc="Analysing samples"):
        res = analyzer.analyse(s["abs"], threshold=float(threshold), with_explanation=False)
        ok = res["verdict"] == s["label"].upper()
        ok_by_kind.setdefault(s["kind"], []).append(ok)
        rows.append({"Type": TAB_KIND[s["kind"]], "Sample": Path(s["path"]).name, "Truth": _chip(s["label"].upper()),
                     "What it is": s["category"],
                     "Face": _pct(res["face"]["p_fake"]) if res["face"] else "–",
                     "Voice": _pct(res["voice"]["p_fake"]) if res["voice"] else "–",
                     "Score": _pct(res["p_fake"]), "Verdict": _chip(res["verdict"]),
                     "Correct": '<span class="wm-yes">✓ yes</span>' if ok else '<span class="wm-no">✗ no</span>'})
    df = pd.DataFrame(rows)
    n_ok = sum(sum(v) for v in ok_by_kind.values())
    tiles = [("All samples", f"{n_ok} / {len(df)}", f"correct at threshold {float(threshold):.2f}")]
    for kind, oks in ok_by_kind.items():
        tiles.append((TAB_KIND[kind], f"{sum(oks)} / {len(oks)}", "correct"))
    tiles_html = '<div class="wm-tiles">' + "".join(
        f'<div class="wm-tile"><span>{k}</span><b>{v}</b><small>{s}</small></div>' for k, v, s in tiles) + "</div>"
    if df.empty:
        return '<div class="wm-plain wm-note">No labelled samples found in demo_samples/.</div>', tiles_html
    table = table_html(df, numeric=("Face", "Voice", "Score"), html_cols=("Truth", "Verdict", "Correct"))
    return table, tiles_html


# --------------------------------------------------------------------------- header + model pages

def header_html(caps: dict, run: dict | None, n_samples: int) -> str:
    pills = []
    for label, name in (("Face model", caps["face_model"]), ("Voice model", caps["voice_model"])):
        pills.append(f'<span class="wm-pill"><span class="wm-led {"on" if name else "off"}"></span>{label}: '
                     f'{"loaded" if name else "not loaded"}</span>')
    top = (f'<div class="wm-top">{LOGO}<div><h1>{TITLE}</h1><p>Checks videos, voice recordings and photos for '
           f'face swaps and synthetic speech, and shows the evidence behind every verdict. Research prototype: '
           f'scores are model estimates, not forensic proof.</p></div>'
           f'<div class="wm-status">{"".join(pills)}</div></div>')
    m = (run or {}).get("metrics") or {}
    stats = [("Inputs", "3", "video, audio, photo"),
             ("Voice test AUC", f'{m["auc"]:.3f}' if m.get("auc") is not None else "–",
              f'EER {100 * m["eer"]:.1f}% on ASVspoof 2019' if m.get("eer") is not None else "no metrics saved"),
             ("Face model", "DFDC #1" if caps["face_model"] and "DFDC" in caps["face_model"] else
              ("own" if caps["face_model"] else "–"), "Kaggle challenge winner, 2020"),
             ("Labelled samples", str(n_samples), "see the Benchmark tab")]
    strip = '<div class="wm-strip">' + "".join(
        f'<div class="wm-stat"><span>{k}</span><b>{v}</b><small>{s}</small></div>' for k, v, s in stats) + "</div>"
    return top + strip


def _spec(rows: list[tuple[str, str]]) -> str:
    return '<dl class="wm-spec">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>"


def model_cards_html(analyzer: Analyzer, run: dict | None) -> str:
    cards = []
    if analyzer.face_source == "pretrained":
        cards.append(
            '<div class="wm-card"><h3>Face model <span class="wm-badge ext">pretrained, external</span></h3>'
            f'<div class="wm-note">{FaceSwapDetector.name}. Used until this project\'s own face model is trained '
            'on FakeAVCeleb.</div>' + _spec([
                ("Architecture", "EfficientNet-B7 (Noisy Student), one output, 380 × 380 input"),
                ("Trained on", "DFDC training set: over 100,000 clips of 3,426 consenting paid actors"),
                ("Pedigree", "Winner of the Kaggle Deepfake Detection Challenge (2020); one of the 7 networks "
                             "of the winning ensemble"),
                ("Input here", "YuNet face box plus a third of margin per side, 16 frames per video"),
                ("Test-time", "Each face is scored together with its mirror image; the two are averaged"),
                ("Licence", "MIT, github.com/selimsef/dfdc_deepfake_challenge"),
                ("Blind spots", "Fully AI-generated faces (GAN, diffusion), lip-sync-only edits, tiny or heavily "
                                "compressed faces")]) + "</div>")
    elif analyzer.face_source == "project":
        cards.append(f'<div class="wm-card"><h3>Face model <span class="wm-badge own">this project</span></h3>'
                     f'<div class="wm-note">{_esc(analyzer.face_name)}</div></div>')
    if run is not None:
        m = run["metrics"] or {}
        perf = (f'AUC {m["auc"]:.3f}, EER {100 * m["eer"]:.1f}% on {m["n"]:,} held-out clips '
                f'(official evaluation partition, unseen attacks A07 to A19)') if m else "no test metrics saved"
        cards.append(
            '<div class="wm-card"><h3>Voice model <span class="wm-badge own">this project</span></h3>'
            f'<div class="wm-note">{_esc(run["branch"])}, run <code>{_esc(run["name"])}</code></div>' + _spec([
                ("Trained on", _esc(run["dataset"])),
                ("Test result", perf),
                ("Input here", "16 kHz mono, 128-band log-mel, 4 s windows every 2 s over the first 30 s; the "
                               "score is the mean over windows that contain speech"),
                ("Out of domain", "AUC 0.59 on 95 clips from the VITS paper: all 10 genuine recordings score "
                                  "above 0.9 (reports/voice_generalization.md)"),
                ("Blind spots", "Phone microphones, rooms, music and compression shift its scores; attack A17 "
                                "is mostly missed")]) + "</div>")
    return f'<div class="wm-models">{"".join(cards)}</div>'


DECISION_RULE = f"""<div class="wm-card"><h3>How a video verdict is formed</h3>
<ul class="wm-list">
<li>A video counts as fake if its face <b>or</b> its voice is fake, so the more suspicious score normally decides.</li>
<li>When the two disagree and the suspicious score is borderline (within {BORDERLINE:.0%} of the threshold) while the
other is clearly real, the borderline score does not flip the verdict. Both scores are always shown.</li>
<li>Face scores average each face with its mirror image, which steadies scores on faces turned to one side.</li>
<li>The voice score ignores 4-second windows that are mostly silence, so a silent stretch cannot drive it.</li>
<li>On the 11 labelled samples this moved the result from 9 to 10 correct. The fixed error is the genuine DFDC
video: face 1.3%, voice 51.5%. That is one sample, chosen with the rule in view, so read it as a
sanity check, not a measured gain.</li>
<li>Still wrong: the genuine LJ Speech recording (voice 92.2%). No inference-side change separates genuine from
synthetic speech outside the training data; that needs retraining on more varied real speech.</li>
</ul></div>"""


def metric_tiles(run: dict) -> str:
    m = run["metrics"]
    if not m:
        return '<div class="wm-plain wm-note">No test metrics saved for this run.</div>'
    eer_acc = m.get("at_eer_threshold", {}).get("accuracy")
    tiles = [("Test AUC", f'{m["auc"]:.3f}', "ranking quality, 1.0 is perfect"),
             ("EER", f'{100 * m["eer"]:.1f}%', "equal error rate, lower is better"),
             ("Accuracy", f'{100 * m["accuracy"]:.1f}%', "at threshold 0.5"),
             ("Precision", f'{m["precision"]:.3f}', "of clips flagged fake"),
             ("Recall", f'{m["recall"]:.3f}', "share of fakes caught"),
             ("F1", f'{m["f1"]:.3f}', "at threshold 0.5")]
    if eer_acc is not None:
        tiles.append(("Accuracy at EER point", f"{100 * eer_acc:.1f}%", f'threshold {m.get("eer_threshold", 0):.4f}'))
    tiles.append(("Test clips", f'{m["n"]:,}', f'best epoch {m.get("best_epoch", "–")}'))
    return '<div class="wm-tiles">' + "".join(
        f'<div class="wm-tile"><span>{k}</span><b>{v}</b><small>{s}</small></div>' for k, v, s in tiles) + "</div>"


def confusion_html(run: dict) -> str:
    m = run["metrics"]
    if not m or "confusion_matrix" not in m:
        return ""
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    return ('<div class="wm-plain"><div class="wm-kicker">Confusion matrix, test set, threshold 0.5</div>'
            '<div class="wm-cm"><div class="h"></div><div class="h">Predicted real</div><div class="h">Predicted fake</div>'
            f'<div class="h">Actually real</div><div class="ok">{tn:,}</div><div class="bad">{fp:,}</div>'
            f'<div class="h">Actually fake</div><div class="bad">{fn:,}</div><div class="ok">{tp:,}</div></div>'
            f'<p class="wm-note">{fp:,} false alarms on {tn + fp:,} real clips. {fn:,} missed fakes out of '
            f'{fn + tp:,}.</p></div>')


def category_html(run: dict) -> str | None:
    cats = model_card.category_table(run)
    if cats is None:
        return None
    cats = cats.copy()
    bars = []
    for v in cats["Mean P(fake)"].astype(float):
        color = COLORS["fake"] if v >= 0.5 else COLORS["real"]
        bars.append(f'<span class="wm-mini"><span style="width:{100 * v:.0f}%;background:{color}"></span></span>{v:.3f}')
    cats["Mean P(fake)"] = bars
    return table_html(cats, numeric=("Clips", "AUC vs. real", "EER vs. real"), html_cols=("Mean P(fake)",))


ABOUT_PIPELINE = [
    ("Video", "16 evenly spaced frames, the YuNet face detector, then the face model on every face and its "
              "mirror image. Frame scores become one face score; if many frames are confidently fake, only those count."),
    ("Video soundtrack", "The first 30 s at 16 kHz, a 128-band log-mel spectrogram, and the voice model on "
                         "4-second windows every 2 s. Windows that are mostly silence are left out of the mean."),
    ("Video verdict", "Fake if the face or the voice is fake, unless one score is borderline and the other "
                      "clearly real. A trained fusion model, once available, replaces this rule."),
    ("Audio file", "The voice model on the same 4-second speech windows; the score is their mean."),
    ("Photo", "The most prominent face is cropped, scored with its mirror image, and boxed in the output."),
    ("Evidence", "Grad-CAM heatmaps on the faces and on the spectrogram, and a timeline of every frame and window score."),
]
ABOUT_READING = [
    "<b>Score</b> is the model's probability that the input is manipulated.",
    "<b>Threshold</b> turns the score into a verdict. Lower it to catch more fakes at the cost of more false alarms.",
    "<b>Strong, moderate, borderline</b> say how far the score sits from the threshold.",
    "<b>INCONCLUSIVE</b> means nothing could be judged, for example no face in a photo or a silent clip.",
    "<b>Labelled samples</b> show their ground truth next to the verdict, including the ones the models get wrong.",
    "<b>Heatmaps</b> show where the model looked. They are not proof of manipulation.",
]
ABOUT_LIMITS = [
    "The voice model was trained on ASVspoof 2019 LA studio recordings. On genuine speech from other sources it "
    "says fake almost every time (all 10 out-of-domain genuine recordings scored above 0.9).",
    "The face model detects DFDC-style face swaps. It is not built for fully AI-generated faces (GAN, diffusion) "
    "or lip-sync-only edits.",
    "This project's own face and fusion models are pending FakeAVCeleb access; the face model shown is the "
    "external DFDC winner, credited in the Models tab.",
    "The benchmark has 11 samples. It shows how the app behaves; it is too small to measure accuracy.",
]
ABOUT_SOURCES = [
    "Demo clips: DFDC preview set (Dolhansky et al., 2019; consenting paid actors).",
    "Demo audio: the VITS paper's samples (Kim et al., 2021; LJ Speech, public domain).",
    "Voice training data: ASVspoof 2019 LA (Wang et al., 2020).",
    "Details: demo_samples/README.md, DATASETS.md and reports/report.md in the repository.",
]


def about_html() -> str:
    pipe = "".join(f"<li><b>{k}.</b> {v}</li>" for k, v in ABOUT_PIPELINE)
    return (f'<div class="wm-grid2"><div class="wm-card"><h3>How it works</h3><ol class="wm-pipe">{pipe}</ol></div>'
            f'<div class="wm-card"><h3>Reading a result</h3><ul class="wm-list">'
            + "".join(f"<li>{x}</li>" for x in ABOUT_READING) + '</ul></div>'
            '<div class="wm-card"><h3>Limitations</h3><ul class="wm-list warn">'
            + "".join(f"<li>{_esc(x)}</li>" for x in ABOUT_LIMITS) + '</ul></div>'
            '<div class="wm-card"><h3>Sources</h3><ul class="wm-list">'
            + "".join(f"<li>{_esc(x)}</li>" for x in ABOUT_SOURCES) + "</ul></div></div>")


# --------------------------------------------------------------------------- layout

def _examples(component, samples: list[dict], kind: str):
    items = [s for s in samples if s["kind"] == kind]
    if not items:
        return
    labels = [f'{s["label"].upper()} · {s["category"]}' for s in items]
    gr.Examples(**_kw(gr.Examples, examples=[[s["abs"]] for s in items], inputs=[component],
                      label="Labelled samples (click one to load it)", example_labels=labels,
                      examples_per_page=8, cache_examples=False))


def _analysis_tab(analyzer: Analyzer, kind: str, samples: list[dict], threshold_default: float):
    """Builds one input/result tab and wires its buttons."""
    with gr.Row(equal_height=False, elem_classes="wm-stack"):
        with gr.Column(scale=5, min_width=300, elem_classes="wm-input-card"):
            gr.HTML('<div class="wm-kicker"><i>1</i>Choose a file</div>')
            if kind == "video":
                inp = gr.Video(label="Video", sources=["upload"], height=280)
            elif kind == "audio":
                inp = gr.Audio(label="Audio file or microphone recording", sources=["upload", "microphone"],
                               type="filepath")
            else:
                inp = gr.Image(label="Photo", sources=["upload", "clipboard"], type="filepath", height=280)
            _examples(inp, samples, kind)
            gr.HTML('<div class="wm-kicker"><i>2</i>Decision settings</div>')
            with gr.Accordion("Threshold and rule", open=False):
                threshold = gr.Slider(0.05, 0.95, value=threshold_default, step=0.01, label="Threshold",
                                      info="A score at or above this counts as FAKE.")
                inputs = [inp, threshold]
                if kind == "video" and analyzer.face_source and analyzer.voice_source:
                    rule = gr.Radio([(label.capitalize(), key) for key, label in VIDEO_RULES.items()],
                                    value="max", label="The video verdict uses",
                                    info="Both scores are always shown. 'Face only' ignores the voice model, "
                                         "which over-flags real speech recorded outside its training data.")
                    inputs.append(rule)
            gr.HTML('<div class="wm-kicker"><i>3</i>Run</div>')
            with gr.Row():
                run_btn = gr.Button("Analyse", variant="primary", scale=3)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
        with gr.Column(scale=7, min_width=300):
            verdict = gr.HTML(empty_html(kind))
            evidence = gr.HTML()

    outputs = [verdict, evidence]
    gr.HTML('<div class="wm-h">Evidence in detail</div><p class="wm-lead">Each view appears after an analysis.</p>')
    with gr.Tabs():
        if kind != "image":
            with gr.Tab("Timeline"):
                gr.HTML('<p class="wm-note">Each dot is a sampled frame (face) or a 4-second window (voice). '
                        'Faded voice segments are mostly silence and are left out of the voice score. Points above '
                        'the dashed line count as fake.</p>')
                timeline = _image(show_label=False, visible=False, interactive=False)
                outputs.append(timeline)
        if kind != "audio":
            with gr.Tab("Face heatmaps"):
                gr.HTML('<p class="wm-note">Grad-CAM. Red marks the face regions that most influenced the decision.</p>')
                with gr.Row(equal_height=False, elem_classes="wm-stack"):
                    if kind == "image":
                        annotated = _image(label="Detected face", visible=False, interactive=False, height=340)
                    faces = gr.Gallery(**_kw(gr.Gallery, label="Grad-CAM on the face",
                                             columns=4 if kind == "video" else 1,
                                             height=300 if kind == "video" else 340, visible=False,
                                             object_fit="contain", allow_preview=True))
                outputs.append(faces)
                if kind == "image":
                    outputs.append(annotated)
        if kind != "image":
            with gr.Tab("Voice heatmap"):
                gr.HTML('<p class="wm-note">Time runs left to right, pitch rises upwards. Red marks what drove the '
                        'voice score in the most decisive speech window.</p>')
                spectrogram = _image(show_label=False, visible=False, interactive=False)
                outputs.append(spectrogram)
        with gr.Tab("Raw result"):
            report = gr.File(label="Download the full result (JSON)", visible=False, interactive=False)
            raw = gr.JSON(label="Result")
    outputs += [raw, report]

    run_btn.click(make_handler(analyzer, kind, samples), inputs, outputs)
    n_media = len(outputs) - 4
    resets = ([None, empty_html(kind), ""] + [gr.update(value=None, visible=False)] * n_media
              + [None, gr.update(value=None, visible=False)])
    clear_btn.click(lambda: resets, None, [inp, *outputs])


def _code_tab(run: dict | None):
    group = None
    for i, s in enumerate(snippets(run)):
        if s["group"] != group:
            group = s["group"]
            gr.HTML(f'<div class="wm-group">{_esc(group)}</div>')
        where = f' <code>{_esc(s["where"])}</code>' if s.get("where") else ""
        gr.HTML(f'<div class="wm-code-head"><i>{i + 1}</i><div><b>{_esc(s["title"])}</b>'
                f'<p>{_esc(s["text"])}{where}</p></div></div>')
        gr.Code(**_kw(gr.Code, value=s["code"], language=s["language"], interactive=False, show_label=False,
                      label=s["title"], lines=1, max_lines=40))


def create_demo(analyzer: Analyzer) -> gr.Blocks:
    caps = analyzer.capabilities()
    project = analyzer.project
    run = model_card.load_run(project.checkpoint, project.run_config) if project else None
    samples = load_samples()

    blocks_kw = {"title": TITLE}
    if GRADIO_MAJOR < 6:                                # Gradio 6 takes theme, css and head in launch()
        blocks_kw.update(theme=THEME, css=CSS, head=HEAD)
    with gr.Blocks(**_kw(gr.Blocks, **blocks_kw)) as demo:
        gr.HTML(header_html(caps, run, len(samples)))
        with gr.Tabs(elem_classes="wm-tabs"):
            with gr.Tab("Video"):
                _analysis_tab(analyzer, "video", samples, analyzer.threshold)
            with gr.Tab("Audio"):
                _analysis_tab(analyzer, "audio", samples, analyzer.threshold)
            with gr.Tab("Photo"):
                _analysis_tab(analyzer, "image", samples, analyzer.threshold)

            with gr.Tab("Benchmark"):
                gr.HTML('<div class="wm-h">Labelled samples against their ground truth</div>'
                        '<p class="wm-lead">Every demo clip analysed in one go. On a CPU this takes about a minute. '
                        'Sources and licences: demo_samples/README.md.</p>')
                with gr.Row(equal_height=True):
                    s_thr = gr.Slider(0.05, 0.95, value=analyzer.threshold, step=0.01, label="Threshold", scale=3)
                    s_btn = gr.Button("Analyse all samples", variant="primary", scale=1)
                s_score = gr.HTML()
                s_table = gr.HTML('<div class="wm-plain wm-note">Press <b>Analyse all samples</b> to fill this table.</div>')
                s_btn.click(lambda thr, progress=gr.Progress(): run_all_samples(analyzer, samples, thr, progress),
                            [s_thr], [s_table, s_score])

            with gr.Tab("Models"):
                gr.HTML(model_cards_html(analyzer, run))
                gr.HTML(DECISION_RULE)
                if run is not None:
                    desc = (run["data"] or {}).get("description") or f"trained on {run['dataset']}"
                    gr.HTML(f'<div class="wm-h">Voice model in detail</div><p class="wm-lead">Run '
                            f'<code>{_esc(run["name"])}</code>. {_esc(desc)}.</p>')
                    gr.HTML(metric_tiles(run))
                    with gr.Row(equal_height=False, elem_classes="wm-stack"):
                        with gr.Column(scale=1, min_width=300):
                            gr.HTML('<div class="wm-h">Data</div>'
                                    + table_html(model_card.split_table(run), numeric=("Clips", "Fake", "Real")))
                            gr.HTML(confusion_html(run))
                        with gr.Column(scale=1, min_width=300):
                            gr.HTML('<div class="wm-h">Training settings</div>' + table_html(model_card.config_table(run)))
                    cats = category_html(run)
                    if cats is not None:
                        sub = " (ASVspoof A07 to A19 and bona fide speech)" if "asvspoof" in run["dataset"].lower() else ""
                        gr.HTML(f'<div class="wm-h">Results per attack system{sub}</div>{cats}')
                    with gr.Row(equal_height=False, elem_classes="wm-stack"):
                        if run["curves"]:
                            with gr.Column(scale=1, min_width=300):
                                gr.HTML('<div class="wm-h">Training curves</div>')
                                _image(value=run["curves"], show_label=False, interactive=False, height=340)
                        if run["roc"]:
                            with gr.Column(scale=1, min_width=300):
                                gr.HTML('<div class="wm-h">ROC curve on the test set</div>')
                                _image(value=run["roc"], show_label=False, interactive=False, height=340)

            with gr.Tab("Code"):
                gr.HTML('<p class="wm-lead">How to run the analyser, the source of its key parts and the commands '
                        'that trained the voice model. Each block stands alone; comments and docstrings are left out '
                        'so only the code is shown.</p>')
                _code_tab(run)

            with gr.Tab("Method"):
                gr.HTML(about_html())
    return demo


def launch(demo: gr.Blocks, **kwargs):
    if GRADIO_MAJOR >= 6:
        kwargs.update(theme=THEME, css=CSS, head=HEAD)
    return demo.launch(**kwargs)
