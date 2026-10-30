"""Face detection and cropping.

Uses OpenCV's YuNet detector (a small ONNX model downloaded once, ~230 KB). If the model
cannot be downloaded it falls back to OpenCV's bundled Haar cascade.

Crops are stored per video as one compressed .npz of JPEG bytes — far fewer files and
far less space on Google Drive than thousands of PNGs.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import cv2
import numpy as np

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
# opencv_zoo stores the model in Git LFS: the media URL serves it directly, the /raw/ URL redirects
YUNET_URLS = (
    f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/{YUNET_FILE}",
    f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{YUNET_FILE}",
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
CACHE_DIR = Path.home() / ".cache" / "deepfake"


def yunet_model() -> Path:
    """Path of the cached YuNet model, downloading and checksum-verifying it on first use."""
    model = CACHE_DIR / YUNET_FILE
    if model.is_file() and hashlib.sha256(model.read_bytes()).hexdigest() == YUNET_SHA256:
        return model
    model.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in YUNET_URLS:
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
        except OSError as e:
            errors.append(f"{url}: {e}")
            continue
        if hashlib.sha256(data).hexdigest() == YUNET_SHA256:
            model.write_bytes(data)
            return model
        errors.append(f"{url}: checksum mismatch")
    raise RuntimeError("could not download the YuNet face detector: " + "; ".join(errors))


class FaceDetector:
    """Returns the most prominent face box (x, y, w, h) in an RGB frame, or None."""

    def __init__(self, score_threshold: float = 0.7):
        self.backend = "yunet"
        try:
            self._yunet = cv2.FaceDetectorYN.create(str(yunet_model()), "", (320, 320), score_threshold)
        except Exception as e:  # offline or old OpenCV
            # OpenCV 5 no longer ships the Haar cascades, so the fallback only exists on OpenCV 4
            if not hasattr(cv2, "CascadeClassifier"):
                raise RuntimeError(f"No face detector available: {e}. Connect to the internet once so the "
                                   f"YuNet model can be cached in {CACHE_DIR}, or install opencv<5.") from e
            print(f"YuNet unavailable ({e}); falling back to Haar cascade")
            self.backend = "haar"
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect(self, rgb: np.ndarray) -> tuple[int, int, int, int] | None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self.backend == "yunet":
            h, w = bgr.shape[:2]
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(bgr)
            if faces is None or len(faces) == 0:
                return None
            # largest box weighted by confidence
            best = max(faces, key=lambda f: f[2] * f[3] * f[-1])
            return tuple(int(v) for v in best[:4])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


def crop_square(rgb: np.ndarray, box: tuple[int, int, int, int] | None, size: int,
                margin: float) -> np.ndarray:
    """Square crop around `box` enlarged by `margin`; centre crop if box is None."""
    h, w = rgb.shape[:2]
    if box is None:
        side = min(h, w)
        cx, cy = w / 2, h / 2
    else:
        x, y, bw, bh = box
        side = max(bw, bh) * margin
        cx, cy = x + bw / 2, y + bh / 2
    half = side / 2
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return cv2.resize(rgb[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def extract_faces(frames: list[np.ndarray], detector: FaceDetector | None, size: int = 224,
                  margin: float = 1.3) -> tuple[np.ndarray, np.ndarray]:
    """Crop one face per frame. Returns (faces [N,size,size,3] uint8, detected [N] bool).

    Frames where detection fails reuse the previous detected box; if no face was ever
    found the centre of the frame is used. `detected` records which frames truly had a face.
    """
    crops, detected, last_box = [], [], None
    for f in frames:
        box = detector.detect(f) if detector else None
        detected.append(box is not None)
        if box is not None:
            last_box = box
        crops.append(crop_square(f, box or last_box, size, margin))
    if not crops:
        return np.zeros((0, size, size, 3), np.uint8), np.zeros(0, bool)
    return np.stack(crops), np.array(detected)


def save_faces(path: str | Path, faces: np.ndarray, detected: np.ndarray, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blobs = [
        cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        for f in faces
    ]
    offsets = np.cumsum([0] + [len(b) for b in blobs])
    data = np.concatenate(blobs) if blobs else np.zeros(0, np.uint8)
    buf = io.BytesIO()
    np.savez(buf, jpeg=data, offsets=offsets, detected=detected)
    path.write_bytes(buf.getvalue())


def load_faces(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of save_faces: returns (faces [N,H,W,3] RGB uint8, detected [N] bool)."""
    with np.load(path) as z:
        data, offsets, detected = z["jpeg"], z["offsets"], z["detected"]
    faces = [
        cv2.cvtColor(cv2.imdecode(data[offsets[i]:offsets[i + 1]], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        for i in range(len(offsets) - 1)
    ]
    return (np.stack(faces) if faces else np.zeros((0, 1, 1, 3), np.uint8)), detected
