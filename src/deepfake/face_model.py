"""Pretrained face-swap detector for images and video frames.

Until this project's own face model is trained (FakeAVCeleb access pending), the demo uses the
model that won the Deepfake Detection Challenge (DFDC, 2020): an EfficientNet-B7 (Noisy Student)
fine-tuned on ~100k DFDC videos by Selim Seferbekov, MIT licence,
github.com/selimsef/dfdc_deepfake_challenge. One of its seven ensemble members is used here
(`final_111_..._0_36`); the weights are downloaded by scripts/download_model.py.

Preprocessing follows the original code: the face box is enlarged by a third of its width and
height on each side, resized so its longer side is 380 px, centred on a black 380 x 380 canvas and
ImageNet-normalised. Faces are found with this project's YuNet detector instead of MTCNN.

Usage:
  model = FaceSwapDetector.load()                 # None if the weights are missing
  out = model.score_image(rgb)                    # {"p_fake", "box", "crop", "cam"} or None (no face)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from deepfake.config import REPO_ROOT
from deepfake.datasets.multimodal import IMAGENET_MEAN, IMAGENET_STD
from deepfake.explainability.gradcam import overlay
from deepfake.preprocessing.faces import FaceDetector

INPUT_SIZE = 380
WEIGHTS_DIR = REPO_ROOT / "models" / "face_dfdc_effb7"
WEIGHTS = WEIGHTS_DIR / "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36"
WEIGHTS_URL = ("https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/"
               "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36")
WEIGHTS_SHA256 = "9db77ab9318863e2f8ab287c8eb83c2232584b82dc2fb41f1d614ddd7900cccb"


class DFDCClassifier(nn.Module):
    """Same layout as DeepFakeClassifier in the DFDC solution, so its state dict loads strictly."""

    def __init__(self):
        super().__init__()
        import timm

        self.encoder = timm.create_model("tf_efficientnet_b7.ns_jft_in1k", pretrained=False, drop_path_rate=0.2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(2560, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.forward_features(x)
        return self.fc(self.dropout(self.avg_pool(x).flatten(1)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state_dict(path: Path) -> dict:
    """State dict of the third-party checkpoint, loaded without unpickling arbitrary objects.

    `weights_only=True` stops a pickle from running code. The file also stores numpy scalars
    (training metrics), so numpy's scalar and dtype reconstructors are allowed: they only
    rebuild plain numbers.
    """
    try:
        from numpy._core.multiarray import scalar      # numpy 2
    except ImportError:
        from numpy.core.multiarray import scalar       # numpy 1
    dtypes = [type(np.dtype(t)) for t in ("float32", "float64", "int64", "int32", "bool")]
    # the file was pickled under numpy 1, so the scalar is referenced by its old module path
    allowed = [(scalar, "numpy.core.multiarray.scalar"), scalar, np.dtype, *dtypes]
    with torch.serialization.safe_globals(allowed):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt.get("state_dict", ckpt)


def crop_face(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """DFDC-style crop: the box plus a third of its size on every side, clipped to the image."""
    x, y, w, h = box
    pw, ph = w // 3, h // 3
    return rgb[max(y - ph, 0):y + h + ph, max(x - pw, 0):x + w + pw]


def to_canvas(face: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Resize so the longer side is `size`, then centre on a black square canvas (uint8 RGB)."""
    h, w = face.shape[:2]
    scale = size / max(h, w)
    if scale != 1:
        face = cv2.resize(face, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    h, w = face.shape[:2]
    y0, x0 = (size - h) // 2, (size - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = face
    return canvas


def confident_mean(p: np.ndarray, t: float = 0.8) -> float:
    """Video score from per-frame scores, after the DFDC solution's `confident_strategy`.

    If a large share of frames is confidently fake, average only those (a face swap is often
    visible in some frames only); if nearly all are confidently real, average those; otherwise
    the plain mean. The original thresholds assume 32 frames and are scaled to len(p) here.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    fakes = np.count_nonzero(p > t)
    if fakes > n / 2.5 and fakes > 11 * n / 32:
        return float(p[p > t].mean())
    if np.count_nonzero(p < 0.2) > 0.9 * n:
        return float(p[p < 0.2].mean())
    return float(p.mean())


class FaceSwapDetector:
    name = "DFDC-winning EfficientNet-B7 (Seferbekov, 2020)"

    def __init__(self, weights: str | Path = WEIGHTS, device: str | torch.device | None = None,
                 face_detector: FaceDetector | None = None, score_threshold: float = 0.6):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights = Path(weights)
        model = DFDCClassifier()
        state = load_state_dict(self.weights)
        model.load_state_dict({re.sub(r"^module\.", "", k): v for k, v in state.items()}, strict=True)
        self.model = model.to(self.device).eval()
        self.face_detector = face_detector or FaceDetector(score_threshold)
        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)

    @classmethod
    def load(cls, device=None, weights: str | Path = WEIGHTS) -> "FaceSwapDetector | None":
        """The detector if its weights are present, else None (the app then hides face analysis)."""
        return cls(weights, device) if Path(weights).is_file() else None

    # ------------------------------------------------------------------ scoring
    def _tensor(self, canvases: list[np.ndarray]) -> torch.Tensor:
        x = torch.from_numpy(np.stack(canvases)).permute(0, 3, 1, 2).float() / 255.0
        return ((x - self._mean) / self._std).to(self.device)

    @torch.no_grad()
    def score_canvases(self, canvases: list[np.ndarray], batch_size: int = 4) -> np.ndarray:
        """P(fake) for each prepared 380 x 380 face canvas."""
        out = []
        for i in range(0, len(canvases), batch_size):
            out.append(torch.sigmoid(self.model(self._tensor(canvases[i:i + batch_size]))).flatten().cpu())
        return torch.cat(out).numpy() if out else np.zeros(0)

    def gradcam(self, canvas: np.ndarray, target: str = "fake") -> np.ndarray:
        """Grad-CAM over the last EfficientNet feature map (12 x 12), upsampled, in [0, 1]."""
        acts = {}
        layer = self.model.encoder.bn2

        def keep(_module, _inputs, output):          # must return None, or it replaces the output
            output.retain_grad()
            acts["a"] = output

        handle = layer.register_forward_hook(keep)
        try:
            with torch.enable_grad():
                self.model.zero_grad(set_to_none=True)
                x = self._tensor([canvas]).requires_grad_(True)
                logit = self.model(x)
                ((1.0 if target == "fake" else -1.0) * logit.sum()).backward()
        finally:
            handle.remove()
        a = acts["a"]
        cam = F.relu((a.grad.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=canvas.shape[:2], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        self.model.zero_grad(set_to_none=True)
        return cam / cam.max() if cam.max() > 0 else cam

    def faces(self, frames: list[np.ndarray]) -> list[dict]:
        """Detect and crop one face per frame: [{"frame": i, "box", "canvas"}] for frames with a face."""
        found = []
        for i, rgb in enumerate(frames):
            box = self.face_detector.detect(rgb)
            if box is not None and box[2] >= 24 and box[3] >= 24:
                found.append({"frame": i, "box": box, "canvas": to_canvas(crop_face(rgb, box))})
        return found

    def score_image(self, rgb: np.ndarray, explain: bool = True, threshold: float = 0.5) -> dict | None:
        """One RGB image -> {"p_fake", "box", "canvas", "overlay"} or None if no face is found."""
        found = self.faces([rgb])
        if not found:
            return None
        face = found[0]
        p = float(self.score_canvases([face["canvas"]])[0])
        out = {"p_fake": p, "box": face["box"], "canvas": face["canvas"]}
        if explain:
            target = "fake" if p >= threshold else "real"
            out["overlay"] = overlay(face["canvas"], self.gradcam(face["canvas"], target))
            out["explained_class"] = target
        return out
