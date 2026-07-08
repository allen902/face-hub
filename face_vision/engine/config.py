"""
FaceVision default configuration constants.

The library never reads any config file. Callers should:
  1. Read their own config file (JSON / YAML / TOML etc.)
  2. Merge with DEFAULT_SETTINGS as fallback
  3. Pass final values to each component's __init__
"""

from copy import deepcopy


DEFAULT_SETTINGS = {
    "device": "cuda",              # cpu / cuda (cuda auto-detects CUDA→DML→CPU)
    "confidence": 0.50,            # detection confidence (RetinaFace: ≥0.45)
    "tolerance": 0.45,             # cosine similarity: 0.40=strict, 0.45=recommended, 0.50=loose
    "cam_width": 640,
    "cam_height": 360,
    "cam_fps": 30,
    "proc_fps": 30,                # ML processing FPS cap (0=unlimited)
    "det_size": 640,               # 320=fast, 480=balanced, 640=accurate
    "track_smooth": 5,             # 3=fast, 5=recommended, 8=stable
    "min_face_size": 60,
    "quality_filter": True,
}


def get_default_settings() -> dict:
    """Return a deep copy of default settings (safe to mutate)."""
    return deepcopy(DEFAULT_SETTINGS)
