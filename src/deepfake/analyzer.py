"""One entry point for the demo app and the CLI: analyse a video, an audio file or an image.

Which model judges what:
  face   this project's face or fusion model if one is trained (checkpoint task `visual` or
         `fusion`); otherwise the pretrained DFDC face-swap detector (deepfake.face_model)
  voice  this project's voice model (task `audio`, or the audio branch of a fusion model)

  video  face model on up to 16 evenly spaced frames + voice model on the soundtrack. With a
         trained fusion checkpoint its fused score decides; otherwise the late-fusion `max` rule
         (fake if the face OR the voice is fake), the same rule used by deepfake.models.fusion
  audio  voice model on overlapping 4 s windows across the first 30 s; P(fake) is the mean
  image  face model on the most prominent face

Every result has the same shape (see Analyzer.analyse), so the UI and the JSON output do not
depend on the input type.

Usage:
  python -m deepfake.analyze clip.mp4 [--out result/]     # any video, audio or image file
  python -m deepfake.analyze --samples                    # every clip in demo_samples/
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deepfake.datasets.multimodal import standardise_mel  # noqa: E402
from deepfake.explainability.gradcam import explain, overlay  # noqa: E402
from deepfake.face_model import FaceSwapDetector, confident_mean  # noqa: E402
from deepfake.inference import AUDIO_EXTS, DeepfakeDetector, figure_to_array, find_best_checkpoint  # noqa: E402
from deepfake.preprocessing.audio import load_audio, log_mel  # noqa: E402
from deepfake.preprocessing.frames import sample_frames  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
MAX_AUDIO_SECONDS = 30.0
WINDOW_HOP_SECONDS = 2.0
FAKE, REAL, INCONCLUSIVE = "FAKE", "REAL", "INCONCLUSIVE"
VIDEO_RULES = {"max": "face + voice (max rule)", "face": "face only", "voice": "voice only"}
# Measured by scripts/eval_voice_generalization.py (reports/voice_generalization.md)
VOICE_CAVEAT = ("The voice model was trained on ASVspoof 2019 studio speech and over-flags genuine speech "
                "recorded elsewhere (all 10 real LJ Speech / VCTK test recordings scored above 0.9). Treat a "
                "FAKE that rests on the voice alone with caution.")
COLORS = {"face": "#7c3aed", "voice": "#0891b2", "threshold": "#64748b"}


def media_kind(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    return "video"


def _verdict(p: float | None, threshold: float) -> str:
    return INCONCLUSIVE if p is None else (FAKE if p >= threshold else REAL)


class Analyzer:
    def __init__(self, project: DeepfakeDetector | None = None, face: FaceSwapDetector | None = None,
                 threshold: float = 0.5, num_frames: int = 16):
        if project is None and face is None:
            raise ValueError("no model available: run `python scripts/download_model.py`")
        self.project, self.face_model, self.threshold, self.num_frames = project, face, threshold, num_frames
        task = project.task if project else None
        self.voice_source = "project" if task in ("audio", "fusion") else None
        self.face_source = "project" if task in ("visual", "fusion") else ("pretrained" if face else None)

    @classmethod
    def from_defaults(cls, checkpoint: str | Path | None = None, device: str | None = None,
                      threshold: float = 0.5, face: bool = True) -> "Analyzer":
        """Best project checkpoint (or `checkpoint`) + the pretrained face model if downloaded."""
        ckpt = checkpoint or find_best_checkpoint()
        project = DeepfakeDetector(ckpt, device, threshold) if ckpt else None
        face_model = None
        if face and not (project and project.task in ("visual", "fusion")):
            face_model = FaceSwapDetector.load(device)
        return cls(project, face_model, threshold)

    # ------------------------------------------------------------------ descriptions
    @property
    def face_name(self) -> str | None:
        if self.face_source == "project":
            return f"project face model ({self.project.checkpoint.parent.name})"
        return FaceSwapDetector.name if self.face_source else None

    @property
    def voice_name(self) -> str | None:
        return f"project voice model ({self.project.checkpoint.parent.name})" if self.voice_source else None

    def capabilities(self) -> dict:
        return {"video": bool(self.face_source or self.voice_source), "audio": bool(self.voice_source),
                "image": bool(self.face_source), "face_model": self.face_name, "voice_model": self.voice_name}

    # ------------------------------------------------------------------ branches
    def _face_from_frames(self, frames: list[np.ndarray], threshold: float,
                          explain_faces: int) -> dict:
        """Pretrained face model on RGB frames -> face branch dict."""
        found = self.face_model.faces(frames)
        branch = {"model": self.face_name, "frames_sampled": len(frames), "faces_found": len(found),
                  "p_fake": None, "frames": [], "overlays": []}
        if not found:
            return branch
        scores = self.face_model.score_canvases([f["canvas"] for f in found])
        n = max(len(frames) - 1, 1)
        for f, p in zip(found, scores):
            # frames are spread evenly, so frame i of N sits at i/N of the clip
            branch["frames"].append({"index": f["frame"], "position": f["frame"] / n, "p_fake": float(p)})
        branch["p_fake"] = confident_mean(scores) if len(scores) > 1 else float(scores[0])
        target = "fake" if branch["p_fake"] >= threshold else "real"
        order = np.argsort(scores)[::-1] if target == "fake" else np.argsort(scores)
        for k in sorted(order[:explain_faces], key=lambda k: found[k]["frame"]):
            cam = self.face_model.gradcam(found[k]["canvas"], target)
            branch["overlays"].append((overlay(found[k]["canvas"], cam),
                                       f"frame {found[k]['frame'] + 1}/{len(frames)} · P(fake) {scores[k]:.0%}"))
        branch["explained_class"] = target
        return branch

    def _voice(self, path: Path, threshold: float, with_explanation: bool) -> dict:
        """Project voice model on overlapping 4 s windows -> voice branch dict."""
        det = self.project
        a = det.cfg["audio"]
        sr, hop = a["sample_rate"], a["hop_length"]
        win = int(a["window_seconds"] * sr / hop) + 1
        wav = load_audio(path, sr, MAX_AUDIO_SECONDS)
        branch = {"model": self.voice_name, "seconds": round(len(wav) / sr, 2), "p_fake": None, "windows": []}
        if len(wav) < sr // 2:                                   # under half a second: nothing to judge
            return branch
        mel = log_mel(wav, sr, a["n_fft"], a["win_length"], hop, a["n_mels"]).astype(np.float32)
        step = int(WINDOW_HOP_SECONDS * sr / hop)
        starts = list(range(0, max(mel.shape[1] - win, 0) + 1, step))
        if mel.shape[1] > win and starts[-1] + win < mel.shape[1]:
            starts.append(mel.shape[1] - win)                    # cover the end of the clip too
        windows = []
        for s in starts:
            m = mel[:, s:s + win]
            if m.shape[1] < win:
                m = np.pad(m, ((0, 0), (0, win - m.shape[1])), constant_values=m.min())
            windows.append(standardise_mel(m))
        x = torch.from_numpy(np.stack(windows))[:, None].to(det.device)
        model = det.model.audio if det.task == "fusion" else det.model
        with torch.no_grad():
            logits = model(x)["logit"].flatten()
        scores = torch.sigmoid(logits).cpu().numpy()
        for s, p in zip(starts, scores):
            branch["windows"].append({"start": round(s * hop / sr, 2),
                                      "end": round(min(s + win, mel.shape[1]) * hop / sr, 2), "p_fake": float(p)})
        branch["p_fake"] = float(scores.mean())
        if with_explanation:
            k = int(np.argmax(scores) if branch["p_fake"] >= threshold else np.argmin(scores))
            batch = {"frames": torch.zeros(1, 1, 3, 8, 8, device=det.device), "mel": x[k:k + 1]}
            target = "fake" if branch["p_fake"] >= threshold else "real"
            exp = explain(model, "audio", batch, target=target)
            branch["explained_class"] = target
            branch["spectrogram"] = _spectrogram_figure(x[k, 0].cpu().numpy(), exp["audio_cam"],
                                                        branch["windows"][k], target)
        return branch

    def _project_video(self, path: Path, threshold: float, with_explanation: bool) -> dict:
        """Face (and fused) scores from this project's visual or fusion checkpoint."""
        res = self.project.analyse(path, with_explanation=with_explanation, threshold=threshold)
        info = res["info"]
        branch = {"model": self.face_name, "frames_sampled": info["frames_sampled"],
                  "faces_found": info["faces_detected"], "p_fake": res["scores"].get("visual"),
                  "frames": [], "overlays": [(img, f"face {i + 1}") for i, img in enumerate(res.get("face_overlays", []))]}
        return {"branch": branch, "fused": res["scores"].get("overall"), "raw": res}

    # ------------------------------------------------------------------ public API
    def analyse(self, path: str | Path, threshold: float | None = None, with_explanation: bool = True,
                explain_faces: int = 4, video_rule: str = "max") -> dict:
        """Returns:
          kind          "video" | "audio" | "image"
          verdict       FAKE | REAL | INCONCLUSIVE, p_fake (None if inconclusive), threshold
          decided_by    what produced p_fake ("face", "voice", "face + voice (max rule)", "fusion model")
          face, voice   branch dicts or None: model, p_fake, per-frame / per-window scores, evidence images
          info          file name, duration, frames, faces found, audio seconds
          warnings      human-readable caveats
        video_rule: "max" (fake if face OR voice is fake), "face" or "voice": which branches decide a
        video when no trained fusion model is loaded. Both branch scores are reported either way.
          timeline      RGB array: per-frame and per-window scores over time (video and audio)
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        threshold = self.threshold if threshold is None else float(threshold)
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if video_rule not in VIDEO_RULES:
            raise ValueError(f"video_rule must be one of {list(VIDEO_RULES)}")
        kind = media_kind(path)
        face = voice = None
        fused = None
        warnings: list[str] = []
        info = {"file": path.name}

        if kind == "image" and not self.face_source:
            warnings.append("No face model is loaded, so images cannot be judged.")
        elif kind == "image":
            face = self._analyse_image(path, threshold, with_explanation)
            if face["p_fake"] is None:
                warnings.append("No face found. This detector judges face swaps, so it cannot rate this image.")
        else:
            if kind == "video":
                info.update(_video_info(path))
                if self.face_source == "project":
                    pv = self._project_video(path, threshold, with_explanation)
                    face, fused = pv["branch"], pv["fused"]
                elif self.face_source == "pretrained":
                    frames = sample_frames(path, self.num_frames)
                    face = self._face_from_frames(frames, threshold,
                                                  explain_faces if with_explanation else 0)
                if face is not None and face["p_fake"] is None:
                    warnings.append("No face found in the sampled frames: the face model could not judge the video.")
                elif face is not None and face["faces_found"] < face["frames_sampled"] / 2:
                    warnings.append(f"A face was found in only {face['faces_found']} of {face['frames_sampled']} "
                                    "frames, so the face score rests on few frames.")
            if self.voice_source:
                voice = self._voice(path, threshold, with_explanation)
                info["audio_seconds"] = voice["seconds"]
                if voice["p_fake"] is None:
                    warnings.append("No usable audio (silent or shorter than 0.5 s)." if kind == "audio"
                                    else "No audio track: the voice model could not judge the video.")
                elif voice["seconds"] >= MAX_AUDIO_SECONDS:
                    warnings.append(f"Only the first {MAX_AUDIO_SECONDS:.0f} s of audio were analysed.")
            elif kind == "audio":
                warnings.append("No voice model is loaded, so audio files cannot be judged.")

        if kind == "video" and fused is None and video_rule != "max":
            p, decided_by = _combine(face if video_rule == "face" else None,
                                     voice if video_rule == "voice" else None, None)
        else:
            p, decided_by = _combine(face, voice, fused)
        voice_p = voice["p_fake"] if voice else None
        face_p = face["p_fake"] if face else None
        if (self.voice_source and voice_p is not None and voice_p >= threshold and p is not None
                and p >= threshold and decided_by != "face" and (face_p is None or face_p < threshold)):
            warnings.append(VOICE_CAVEAT)
        if p is None and not any("cannot" in w or "could not" in w for w in warnings):
            warnings.append("Insufficient evidence: nothing in this file could be judged.")
        result = {"kind": kind, "verdict": _verdict(p, threshold), "p_fake": p, "threshold": threshold,
                  "decided_by": decided_by, "face": face, "voice": voice, "info": info, "warnings": warnings}
        if with_explanation and kind != "image":
            result["timeline"] = timeline_figure(result)
        return result

    def _analyse_image(self, path: Path, threshold: float, with_explanation: bool) -> dict:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read image: {path.name}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        branch = {"model": self.face_name, "frames_sampled": 1, "faces_found": 0, "p_fake": None,
                  "frames": [], "overlays": []}
        if self.face_source == "project":
            p = self.project.score_image(rgb)
            if p is not None:
                branch.update(p_fake=p, faces_found=1)
            return branch
        out = self.face_model.score_image(rgb, explain=with_explanation, threshold=threshold)
        if out is None:
            return branch
        branch.update(p_fake=out["p_fake"], faces_found=1, box=out["box"], face_crop=out["canvas"])
        if with_explanation:
            branch["overlays"] = [(out["overlay"], f"Grad-CAM: evidence for {out['explained_class'].upper()}")]
            branch["explained_class"] = out["explained_class"]
            x, y, w, h = out["box"]
            boxed = rgb.copy()
            cv2.rectangle(boxed, (x, y), (x + w, y + h), (124, 58, 237), max(2, rgb.shape[1] // 300))
            branch["annotated"] = boxed
        return branch


def _combine(face: dict | None, voice: dict | None, fused: float | None) -> tuple[float | None, str]:
    if fused is not None:
        return fused, "fusion model"
    pf = face["p_fake"] if face else None
    pv = voice["p_fake"] if voice else None
    if pf is not None and pv is not None:
        return max(pf, pv), "face + voice (max rule)"
    if pf is not None:
        return pf, "face"
    if pv is not None:
        return pv, "voice"
    return None, "nothing"


def _video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path.name}. Supported: mp4, mov, avi, mkv, webm.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": round(fps, 2), "duration": round(n / fps, 2) if fps else None, "resolution": f"{w}×{h}"}


# --------------------------------------------------------------------------- figures

def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=8)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)


def _spectrogram_figure(mel: np.ndarray, cam: np.ndarray, window: dict, target: str) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=110)
    t0, t1 = window["start"], window["start"] + mel.shape[1] / 100
    ax.imshow(mel, origin="lower", aspect="auto", cmap="gray", extent=(t0, t1, 0, mel.shape[0]))
    ax.imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.42, vmin=0, vmax=1,
              extent=(t0, t1, 0, mel.shape[0]))
    ax.set_xlabel("time (s)", fontsize=8, color="#475569")
    ax.set_ylabel("mel band (low → high pitch)", fontsize=8, color="#475569")
    ax.set_title(f"Spectrogram of the {window['start']:.0f}–{window['end']:.0f} s window, "
                 f"heatmap = evidence for {target.upper()}", fontsize=9, color="#0f172a", loc="left")
    ax.tick_params(colors="#475569", labelsize=8)
    fig.tight_layout()
    return figure_to_array(fig)


def timeline_figure(res: dict) -> np.ndarray | None:
    """Per-frame face scores and per-window voice scores over the clip, with the threshold."""
    face, voice = res["face"], res["voice"]
    has_face = bool(face and face["frames"])
    has_voice = bool(voice and voice["windows"])
    if not (has_face or has_voice):
        return None
    duration = res["info"].get("duration") or (voice["seconds"] if voice else 1.0) or 1.0
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=110)
    if has_voice:
        for w in voice["windows"]:
            ax.plot([w["start"], w["end"]], [w["p_fake"]] * 2, color=COLORS["voice"], linewidth=5,
                    alpha=0.55, solid_capstyle="butt")
        mids = [(w["start"] + w["end"]) / 2 for w in voice["windows"]]
        ax.plot(mids, [w["p_fake"] for w in voice["windows"]], "o", color=COLORS["voice"], markersize=4,
                label=f"voice, 4 s windows (mean {voice['p_fake']:.0%})")
    if has_face:
        xs = [f["position"] * duration for f in face["frames"]]
        ax.plot(xs, [f["p_fake"] for f in face["frames"]], "-o", color=COLORS["face"], markersize=4,
                linewidth=1.6, label=f"face, per frame (video score {face['p_fake']:.0%})")
    ax.axhline(res["threshold"], color=COLORS["threshold"], linestyle="--", linewidth=1,
               label=f"threshold {res['threshold']:.0%}")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, max(duration, 0.1))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("time in the clip (s)", fontsize=8, color="#475569")
    ax.set_ylabel("P(fake)", fontsize=8, color="#475569")
    _style(ax)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return figure_to_array(fig)
