"""Visual, audio and fusion models."""

from deepfake.models.audio import AudioModel
from deepfake.models.fusion import MultimodalModel, build_model
from deepfake.models.visual import VisualModel

__all__ = ["AudioModel", "MultimodalModel", "VisualModel", "build_model"]
