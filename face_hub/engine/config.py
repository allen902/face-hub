"""
FaceHub default configuration constants.

The library never reads any config file. Callers should:
  1. Read their own config file (JSON / YAML / TOML etc.)
  2. Merge with DEFAULT_SETTINGS as fallback
  3. Pass final values to each component's __init__

Note: dict.update() is a *shallow* merge.  If DEFAULT_SETTINGS ever gains
nested structures, callers must switch to a recursive deep-merge.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal, TypedDict


class FaceHubSettings(TypedDict, total=False):
    """Typed schema for FaceHub configuration.

    All keys are optional (total=False) so callers can supply a partial dict
    and fill the rest from DEFAULT_SETTINGS.
    """
    device: Literal["auto", "cuda", "cpu"]
    confidence: float
    tolerance: float
    cam_width: int
    cam_height: int
    cam_fps: int
    proc_fps: int
    det_size: int
    track_smooth: int
    min_face_size: int
    quality_filter: bool


DEFAULT_SETTINGS: FaceHubSettings = {
    "device": "auto",              # auto → CUDA → DirectML → CPU fallback
    "confidence": 0.50,            # detection confidence (RetinaFace: ≥0.45)
    "tolerance": 0.45,             # cosine similarity: 0.40=strict, 0.45=recommended, 0.50=loose
    "cam_width": 640,
    "cam_height": 360,
    "cam_fps": 30,
    "proc_fps": 30,                # ML processing FPS cap (0=unlimited)
    "det_size": 640,               # 320=fast, 480=balanced, 640=accurate
    "track_smooth": 5,             # 3=fast, 5=recommended, 8=stable
    "min_face_size": 80,
    "quality_filter": True,
}


def get_default_settings() -> FaceHubSettings:
    """Return a deep copy of default settings (safe to mutate)."""
    return deepcopy(DEFAULT_SETTINGS)
