"""Grad-CAM explanations for the visual, audio and fusion models (Phase 6).

Grad-CAM (Selvaraju et al., 2017): weight the feature maps of a late convolutional layer by the
average gradient of the model's output w.r.t. them, sum, ReLU, upsample. Bright regions are the
parts of the input that pushed the score towards the explained class.

Implemented with plain PyTorch hooks (not the pytorch-grad-cam package) because the fusion model
takes several inputs: one backward pass of the FUSED logit gives a heatmap for every face frame
and one for the spectrogram at the same time.

  layers        visual: visual backbone `bn2` (EfficientNet, 7x7 per frame)
                audio:  audio backbone `layer4` (ResNet-18, 4x13 over the 4 s spectrogram)
  target        "predicted" (default): evidence for the class the model chose
                "fake": evidence for fake · "real": evidence for real

Usage:
  exp = explain(model, "fusion", batch)        # batch = dataset item collated to size 1
  fig = render_explanation(batch, exp, title="...")
"""

from __future__ import annotations

from contextlib import contextmanager

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from deepfake.datasets.multimodal import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def target_layer(backbone: nn.Module) -> nn.Module:
    """Last convolutional feature layer of a timm EfficientNet or torchvision ResNet."""
    for name in ("bn2", "layer4", "conv_head"):
        if hasattr(backbone, name):
            return getattr(backbone, name)
    raise ValueError(f"no known Grad-CAM layer in {type(backbone).__name__}")


def cam_layers(model: nn.Module, task: str) -> dict[str, nn.Module]:
    if task == "visual":
        return {"visual": target_layer(model.backbone)}
    if task == "audio":
        return {"audio": target_layer(model.backbone)}
    return {"visual": target_layer(model.visual.backbone), "audio": target_layer(model.audio.backbone)}


@contextmanager
def _capture(layers: dict[str, nn.Module]):
    """Record forward activations of the given layers and keep their gradients."""
    acts: dict[str, torch.Tensor] = {}
    handles = []
    for key, layer in layers.items():
        def hook(_m, _i, out, key=key):
            out.retain_grad()
            acts[key] = out
        handles.append(layer.register_forward_hook(hook))
    try:
        yield acts
    finally:
        for h in handles:
            h.remove()


def _cam(act: torch.Tensor) -> torch.Tensor:
    """act [N, C, h, w] with .grad -> CAM [N, h, w] (unnormalised, >= 0)."""
    weights = act.grad.mean(dim=(2, 3), keepdim=True)
    return F.relu((weights * act).sum(1))


def _normalise(cam: np.ndarray) -> np.ndarray:
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def forward_logit(model: nn.Module, task: str, batch: dict) -> torch.Tensor:
    if task == "visual":
        return model(batch["frames"], batch["frame_mask"])["logit"]
    if task == "audio":
        return model(batch["mel"])["logit"]
    return model(batch["frames"], batch["frame_mask"], batch["mel"], batch["has_video"],
                 batch["has_audio"])["logit"]


def explain(model: nn.Module, task: str, batch: dict, target: str = "predicted") -> dict:
    """Grad-CAM for ONE sample (batch size 1, on the model's device).

    Returns {"prob": P(fake), "target": "fake"|"real",
             "visual_cam": [T, H, W] in [0, 1] (one shared scale for all frames) or None,
             "audio_cam": [n_mels, F] in [0, 1] or None}
    """
    if batch["frames"].shape[0] != 1:
        raise ValueError("explain() works on a single sample (batch size 1)")
    if target not in {"predicted", "fake", "real"}:
        raise ValueError(f"Unknown explanation target: {target}")
    # Frozen parameters still need a differentiable input for activation gradients.
    batch = {k: (v.detach().requires_grad_(True) if k in {"frames", "mel"} else v)
             for k, v in batch.items()}
    model.eval()
    layers = cam_layers(model, task)
    with torch.enable_grad(), _capture(layers) as acts:
        model.zero_grad(set_to_none=True)
        logit = forward_logit(model, task, batch)
        prob = torch.sigmoid(logit).item()
        explained = ("fake" if prob >= 0.5 else "real") if target == "predicted" else target
        sign = 1.0 if explained == "fake" else -1.0
        (sign * logit.sum()).backward()

    out = {"prob": prob, "target": explained, "visual_cam": None, "audio_cam": None}
    if "visual" in acts and bool(batch["has_video"].item() if task == "fusion" else True):
        H, W = batch["frames"].shape[-2:]
        valid_cam = _cam(acts["visual"])
        mask = batch["frame_mask"][0].bool()
        cam = valid_cam.new_zeros(len(mask), *valid_cam.shape[-2:])
        cam[mask] = valid_cam
        cam = cam[None]                                       # [1, T, h, w]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)[0]
        cam = cam * batch["frame_mask"][0, :, None, None].float()
        out["visual_cam"] = _normalise(cam.detach().cpu().numpy())
    if "audio" in acts and bool(batch["has_audio"].item() if task == "fusion" else True):
        n_mels, n_frames = batch["mel"].shape[-2:]
        cam = _cam(acts["audio"])[None]                        # [1, 1, h, w]
        cam = F.interpolate(cam, size=(n_mels, n_frames), mode="bilinear", align_corners=False)[0, 0]
        out["audio_cam"] = _normalise(cam.detach().cpu().numpy())
    model.zero_grad(set_to_none=True)
    return out


# --------------------------------------------------------------------------- rendering

def denormalise(frames: torch.Tensor) -> np.ndarray:
    """Dataset frames [T, 3, H, W] (ImageNet-normalised) -> uint8 RGB [T, H, W, 3]."""
    x = frames.detach().cpu().numpy().transpose(0, 2, 3, 1) * IMAGENET_STD + IMAGENET_MEAN
    return (x.clip(0, 1) * 255).astype(np.uint8)


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] heatmap (JET colour map) onto an RGB uint8 image."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (img * (1 - alpha) + heat * alpha).astype(np.uint8)


def render_explanation(batch: dict, exp: dict, title: str = "", n_frames: int = 6) -> plt.Figure:
    """Figure: face frames with Grad-CAM overlay (top) and spectrogram + CAM (bottom)."""
    has_v, has_a = exp["visual_cam"] is not None, exp["audio_cam"] is not None
    rows = int(has_v) * 2 + int(has_a)
    fig = plt.figure(figsize=(2.2 * n_frames, 2.3 * max(rows, 1) + 0.6))
    gs = fig.add_gridspec(max(rows, 1), n_frames)
    r = 0
    if has_v:
        valid = np.flatnonzero(batch["frame_mask"][0].cpu().numpy())
        pick = valid[np.linspace(0, len(valid) - 1, min(n_frames, len(valid))).round().astype(int)]
        imgs = denormalise(batch["frames"][0])
        for j, t in enumerate(pick):
            ax = fig.add_subplot(gs[r, j]); ax.imshow(imgs[t]); ax.axis("off")
            ax.set_title(f"frame {t}", fontsize=8)
            ax = fig.add_subplot(gs[r + 1, j]); ax.imshow(overlay(imgs[t], exp["visual_cam"][t])); ax.axis("off")
        r += 2
    if has_a:
        ax = fig.add_subplot(gs[r, :])
        mel = batch["mel"][0, 0].detach().cpu().numpy()
        ax.imshow(mel, origin="lower", aspect="auto", cmap="gray")
        ax.imshow(exp["audio_cam"], origin="lower", aspect="auto", cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set(xlabel="time (10 ms frames)", ylabel="mel bin", title="spectrogram + Grad-CAM")
    head = f"P(fake) = {exp['prob']:.3f} · heatmap shows evidence for: {exp['target'].upper()}"
    fig.suptitle(f"{title}\n{head}" if title else head, fontsize=10)
    fig.tight_layout()
    return fig
