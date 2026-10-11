"""Gradio demo UI (Phase 7). Launch with app/app.py."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from deepfake.inference import DeepfakeDetector

TITLE = "Robust Multimodal Deepfake Analysis"
INTRO = """
Upload a short video (with audio). The system checks the **face** and the **voice**, combines
the evidence, and shows Grad-CAM heatmaps of the regions that drove the decision.

*Research prototype for a college project. Scores are model estimates, not forensic proof.*
"""


def _pct(p) -> str:
    return "n/a" if p is None else f"{100 * p:.1f}%"


def format_details(res: dict) -> str:
    """Markdown summary of a DeepfakeDetector.analyse() result."""
    s, info = res["scores"], res["info"]
    lines = [f"### Verdict: **{res['verdict']}** (P(fake) = {_pct(res['p_fake'])}, threshold {res['threshold']:.2f})", ""]
    rows = [("Overall (fusion)", "overall"), ("Face model", "visual"), ("Voice model", "audio")]
    lines += ["| Evidence | P(fake) |", "|---|---|"]
    lines += [f"| {name} | {_pct(s[key])} |" for name, key in rows if key in s]
    lines += ["", f"- Faces detected in {info['faces_detected']} / {info['frames_sampled']} sampled frames",
              f"- Audio: {info['audio_seconds']} s analysed" if info["has_audio"] else "- Audio: none found",
              f"- Model: `{res['task']}` (run `{Path(res['checkpoint']).parent.name}`)"]
    if res.get("explained_class"):
        lines.append(f"- Heatmaps show evidence for: **{res['explained_class'].upper()}**")
    for w in res["warnings"]:
        lines.append(f"- ⚠️ {w}")
    return "\n".join(lines)


def make_handler(detector: DeepfakeDetector):
    def analyse(video_path: str | None, target: str):
        if not video_path:
            raise gr.Error("Please upload a video first.")
        res = detector.analyse(video_path, with_explanation=True, target=target)
        label = ({"INCONCLUSIVE": 1.0} if res["p_fake"] is None else
                 {"FAKE": res["p_fake"], "REAL": 1 - res["p_fake"]})
        return label, format_details(res), res.get("face_overlays", []), res.get("spectrogram")
    return analyse


def create_demo(detector: DeepfakeDetector) -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n{INTRO}")
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Video", sources=["upload"])
                target = gr.Radio(["predicted", "fake", "real"], value="predicted",
                                  label="Heatmaps show evidence for")
                run = gr.Button("Analyse", variant="primary")
            with gr.Column():
                verdict = gr.Label(label="Verdict", num_top_classes=2)
                details = gr.Markdown()
        faces = gr.Gallery(label="Face evidence (Grad-CAM)", columns=6)
        spectrogram = gr.Image(label="Voice evidence (Grad-CAM on the spectrogram)")
        run.click(make_handler(detector), inputs=[video, target], outputs=[verdict, details, faces, spectrogram])
    return demo
