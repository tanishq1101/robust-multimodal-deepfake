"""Single-file inference: raw video (or audio) -> verdict, per-modality scores, Grad-CAM.

Runs the same preprocessing as training (16 uniform frames, YuNet face crops, 16 kHz log-mel,
centre 4 s window of the first 10 s) in memory, so nothing is written to disk.

Usage:
  python -m deepfake.inference --checkpoint <ckpts>/fusion_intermediate_fakeavceleb/best.pt clip.mp4
  python -m deepfake.inference clip.mp4            # picks the best available checkpoint
  python -m deepfake.inference clip.mp4 --out result/   # also saves the explanation figure

Python:
  det = DeepfakeDetector("best.pt")
  result = det.analyse("clip.mp4")      # dict, see DeepfakeDetector.analyse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deepfake.config import REPO_ROOT, get_path, load_config  # noqa: E402
from deepfake.datasets.multimodal import face_to_tensor, standardise_mel  # noqa: E402
from deepfake.explainability.gradcam import denormalise, explain, overlay, render_explanation  # noqa: E402
from deepfake.preprocessing.audio import crop_or_pad, load_audio, log_mel  # noqa: E402
from deepfake.preprocessing.faces import FaceDetector, extract_faces  # noqa: E402
from deepfake.preprocessing.frames import sample_frames  # noqa: E402

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
# preferred order when no checkpoint is given (first existing wins)
CHECKPOINT_PREFERENCE = ["fusion_intermediate_*_robust", "fusion_intermediate_*", "fusion_late_*",
                         "visual_*_robust", "visual_*", "audio_*_robust", "audio_*"]


def load_model(checkpoint: str | Path, device) -> tuple[torch.nn.Module, dict]:
    """Rebuild a model from any training checkpoint (visual / audio / fusion)."""
    from deepfake.training.tasks import TASKS

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rc = ckpt["run_config"]
    model = TASKS[rc["task"]].build_model(rc, False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), rc


def find_best_checkpoint(cfg: dict | None = None, roots: list[Path] | None = None) -> Path | None:
    """Best available checkpoint: <data_root>/checkpoints first, then the published models/ folder."""
    if roots is None:
        cfg = cfg or load_config()
        roots = [get_path(cfg, "checkpoints"), REPO_ROOT / "models"]
    for pattern in CHECKPOINT_PREFERENCE:
        for root in roots:
            hits = sorted(Path(root).glob(f"{pattern}/best.pt"))
            if hits:
                return hits[0]
    return None


def figure_to_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


class DeepfakeDetector:
    def __init__(self, checkpoint: str | Path, device: str | None = None, threshold: float = 0.5,
                 use_face_detector: bool = True, cfg: dict | None = None):
        from deepfake.training.engine import get_device

        self.cfg = cfg or load_config()
        self.device = torch.device(device) if device else get_device()
        self.checkpoint = Path(checkpoint)
        self.model, self.run_config = load_model(self.checkpoint, self.device)
        if "preprocessing" in self.run_config:
            self.cfg = {**self.cfg, **self.run_config["preprocessing"]}
        self.task = self.run_config["task"]
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.face_detector = FaceDetector(self.cfg["video"]["detector_score"]) if use_face_detector else None

    # ------------------------------------------------------------------ preprocessing
    def preprocess(self, path: str | Path) -> tuple[dict, dict]:
        """Returns (batch of size 1 on the model's device, info dict)."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        v, a = self.cfg["video"], self.cfg["audio"]
        T, S = v["num_frames"], v["face_size"]
        mel_frames = int(a["window_seconds"] * a["sample_rate"] / a["hop_length"]) + 1
        info = {"file": path.name, "frames_sampled": 0, "faces_detected": 0, "audio_seconds": 0.0}

        frames = torch.zeros(T, 3, S, S)
        mask = torch.zeros(T, dtype=torch.bool)
        has_video = False
        if path.suffix.lower() not in AUDIO_EXTS:
            raw = sample_frames(path, T)
            if raw:
                faces, detected = extract_faces(raw, self.face_detector, S, v["face_margin"])
                info["frames_sampled"], info["faces_detected"] = len(faces), int(detected.sum())
                for i, f in enumerate(faces):
                    frames[i], mask[i] = face_to_tensor(f), True
                # without a face detector we trust the centre crops; otherwise require >= 1 face
                has_video = bool(detected.any()) or self.face_detector is None

        mel = torch.zeros(1, a["n_mels"], mel_frames)
        wav = load_audio(path, a["sample_rate"], a["max_seconds"])
        has_audio = len(wav) > 0
        if has_audio:
            info["audio_seconds"] = round(len(wav) / a["sample_rate"], 2)
            m = log_mel(wav, a["sample_rate"], a["n_fft"], a["win_length"], a["hop_length"], a["n_mels"])
            m = crop_or_pad(m.astype(np.float32), mel_frames)
            mel = torch.from_numpy(standardise_mel(m))[None]

        batch = {"frames": frames[None], "frame_mask": mask[None], "mel": mel[None],
                 "has_video": torch.tensor([has_video]), "has_audio": torch.tensor([has_audio])}
        info.update(has_face=has_video, has_audio=has_audio)
        return {k: t.to(self.device) for k, t in batch.items()}, info

    @torch.no_grad()
    def score_image(self, rgb: np.ndarray) -> float | None:
        """Face P(fake) of one RGB image with a visual or fusion checkpoint; None if no face is found."""
        if self.task not in ("visual", "fusion"):
            raise ValueError(f"a {self.task} model cannot score images")
        v, a = self.cfg["video"], self.cfg["audio"]
        S = v["face_size"]
        faces, detected = extract_faces([rgb], self.face_detector, S, v["face_margin"])
        if self.face_detector is not None and not detected.any():
            return None
        frames = torch.zeros(1, v["num_frames"], 3, S, S)
        mask = torch.zeros(1, v["num_frames"], dtype=torch.bool)
        frames[0, 0], mask[0, 0] = face_to_tensor(faces[0]), True
        frames, mask = frames.to(self.device), mask.to(self.device)
        if self.task == "visual":
            return torch.sigmoid(self.model(frames, mask)["logit"]).item()
        mel = torch.zeros(1, 1, a["n_mels"], int(a["window_seconds"] * a["sample_rate"] / a["hop_length"]) + 1,
                          device=self.device)
        out = self.model(frames, mask, mel, torch.tensor([True], device=self.device),
                         torch.tensor([False], device=self.device))
        return torch.sigmoid(out["visual_logit"]).item()

    # ------------------------------------------------------------------ analysis
    @torch.no_grad()
    def _scores(self, batch: dict) -> dict:
        m = self.model
        if self.task == "visual":
            return {"visual": torch.sigmoid(m(batch["frames"], batch["frame_mask"])["logit"]).item()}
        if self.task == "audio":
            return {"audio": torch.sigmoid(m(batch["mel"])["logit"]).item()}
        out = m(batch["frames"], batch["frame_mask"], batch["mel"], batch["has_video"], batch["has_audio"])
        return {"overall": torch.sigmoid(out["logit"]).item(),
                "visual": torch.sigmoid(out["visual_logit"]).item() if batch["has_video"].item() else None,
                "audio": torch.sigmoid(out["audio_logit"]).item() if batch["has_audio"].item() else None}

    def analyse(self, path: str | Path, with_explanation: bool = True, target: str = "predicted",
                threshold: float | None = None) -> dict:
        """Returns a dict:
          p_fake, verdict ("FAKE"/"REAL"), threshold, task, checkpoint,
          scores        {"overall"?, "visual"?, "audio"?} branch probabilities (None if modality missing)
          info          frames sampled, faces detected, audio seconds, has_face, has_audio
          warnings      list of human-readable caveats
          face_overlays list of RGB arrays (Grad-CAM on face frames)   [with_explanation]
          spectrogram   RGB array (spectrogram + Grad-CAM)             [with_explanation]
          figure        RGB array of the full explanation figure        [with_explanation]
        """
        threshold = self.threshold if threshold is None else threshold
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        batch, info = self.preprocess(path)
        warnings = []
        if self.task in ("visual", "fusion") and not info["has_face"]:
            warnings.append("No face detected: the visual evidence is unreliable." if self.task == "visual"
                            else ("No face detected: decision based on audio only." if info["has_audio"]
                                  else "No face detected."))
        if self.task in ("audio", "fusion") and not info["has_audio"]:
            warnings.append(("No audio track: decision based on the face only." if info["has_face"]
                             else "No audio track.") if self.task == "fusion"
                            else "No audio track: the audio model cannot judge this file.")
        usable = ((self.task in ("visual", "fusion") and info["has_face"])
                  or (self.task in ("audio", "fusion") and info["has_audio"]))
        if not usable:
            keys = ["overall", "visual", "audio"] if self.task == "fusion" else [self.task]
            warnings.append("Insufficient evidence: this model cannot judge the file.")
            return {"p_fake": None, "verdict": "INCONCLUSIVE", "threshold": threshold,
                    "task": self.task, "checkpoint": str(self.checkpoint),
                    "scores": dict.fromkeys(keys), "info": info, "warnings": warnings}
        scores = self._scores(batch)
        p = scores.get("overall", scores.get("visual", scores.get("audio")))
        result = {"p_fake": p, "verdict": "FAKE" if p >= threshold else "REAL",
                  "threshold": threshold, "task": self.task, "checkpoint": str(self.checkpoint),
                  "scores": scores, "info": info, "warnings": warnings}

        if with_explanation and (info["has_face"] or info["has_audio"]):
            explained_target = ("fake" if p >= threshold else "real") if target == "predicted" else target
            exp = explain(self.model, self.task, batch, target=explained_target)
            result["explained_class"] = exp["target"]
            if exp["visual_cam"] is not None:
                imgs = denormalise(batch["frames"][0])
                valid = np.flatnonzero(batch["frame_mask"][0].cpu().numpy())
                pick = valid[np.linspace(0, len(valid) - 1, min(6, len(valid))).round().astype(int)]
                result["face_overlays"] = [overlay(imgs[t], exp["visual_cam"][t]) for t in pick]
            if exp["audio_cam"] is not None:
                fig, ax = plt.subplots(figsize=(9, 2.6))
                ax.imshow(batch["mel"][0, 0].cpu().numpy(), origin="lower", aspect="auto", cmap="gray")
                ax.imshow(exp["audio_cam"], origin="lower", aspect="auto", cmap="jet", alpha=0.45, vmin=0, vmax=1)
                ax.set(xlabel="time (10 ms frames)", ylabel="mel bin", title="voice: spectrogram + Grad-CAM")
                fig.tight_layout()
                result["spectrogram"] = figure_to_array(fig)
            cpu_batch = {k: t.cpu() for k, t in batch.items()}
            result["figure"] = figure_to_array(render_explanation(cpu_batch, exp, title=info["file"]))
        return result


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="video (.mp4, .avi, .mov, ...) or audio (.wav, .mp3, ...) file")
    p.add_argument("--checkpoint", help="default: best available checkpoint in <data_root>/checkpoints")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--target", choices=["predicted", "fake", "real"], default="predicted")
    p.add_argument("--out", help="folder to save explanation.png + result.json")
    p.add_argument("--device")
    args = p.parse_args(argv)
    ckpt = args.checkpoint or find_best_checkpoint()
    if not ckpt:
        p.error("no checkpoint given and none found in <data_root>/checkpoints")
    det = DeepfakeDetector(ckpt, args.device, args.threshold)
    res = det.analyse(args.file, with_explanation=bool(args.out), target=args.target)
    summary = {k: v for k, v in res.items() if k not in ("face_overlays", "spectrogram", "figure")}
    print(json.dumps(summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(summary, indent=2))
        if "figure" in res:
            plt.imsave(out / "explanation.png", res["figure"])
        print(f"Saved to {out}")
    return res


if __name__ == "__main__":
    main()
