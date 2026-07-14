"""
FaceHub — Real-time Face Recognition Library

Usage:
    from face_hub import (
        FaceHubPipeline, FaceDetector, FaceRecognizer,
        CameraThread, FaceTracker, FaceDatabase,
    )
    from face_hub.types import UNKNOWN_SENTINEL, PipelineResult, TrackedFace
"""

__version__ = "1.1.0"
__author__ = "AllenDeng"

# When installed as a package, prefer the metadata version (pyproject.toml).
# Falls back to the hardcoded value above during local development.
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("face-hub")
except Exception:  # noqa: BLE001 — fallback for local dev without install
    pass

# ── Core components (from engine subpackage) ─────────────────
from face_hub.engine.camera import CameraThread
from face_hub.engine.face_detector import FaceDetector
from face_hub.engine.face_recognizer import FaceRecognizer
from face_hub.engine.face_tracker import FaceTracker
from face_hub.engine.face_database import FaceDatabase

# ── Pipeline (from face_hub/ main package) ─────────────────────
from face_hub.pipeline import FaceHubPipeline
from face_hub.detector_protocol import DetectorProtocol

# ── Types ──────────────────────────────────────────────────────
from face_hub.types import (
    UNKNOWN_SENTINEL,
    BBox,
    DetectionResult,
    DetectionWithEmbedding,
    TrackedFace,
    PipelineResult,
)

# ── Exceptions ─────────────────────────────────────────────────
from face_hub.exceptions import (
    FaceHubError,
    ModelLoadError,
    DependencyError,
    InferenceError,
    CameraError,
    DatabaseError,
    SerializationError,
    RecognitionError,
)

# ── Config ─────────────────────────────────────────────────────
from face_hub.engine.config import DEFAULT_SETTINGS, FaceHubSettings, get_default_settings

__all__ = [
    # Core
    "CameraThread",
    "FaceDetector",
    "FaceRecognizer",
    "FaceTracker",
    "FaceDatabase",
    "FaceHubPipeline",
    "DetectorProtocol",
    # Types
    "UNKNOWN_SENTINEL",
    "BBox",
    "DetectionResult",
    "DetectionWithEmbedding",
    "TrackedFace",
    "PipelineResult",
    # Exceptions
    "FaceHubError",
    "ModelLoadError",
    "DependencyError",
    "InferenceError",
    "CameraError",
    "DatabaseError",
    "SerializationError",
    "RecognitionError",
    # Config
    "DEFAULT_SETTINGS",
    "FaceHubSettings",
    "get_default_settings",
]
