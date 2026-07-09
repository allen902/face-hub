"""
FaceHub — Real-time Face Recognition Library

Usage:
    from face_hub import (
        FaceHubPipeline, FaceDetector, FaceRecognizer,
        CameraThread, FaceTracker, FaceDatabase,
    )
    from face_hub.types import UNKNOWN_SENTINEL, PipelineResult, TrackedFace
"""

__version__ = "1.0.0"
__author__ = "AllenDeng"

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
    InferenceError,
    CameraError,
    DatabaseError,
    RecognitionError,
)

# ── Config ─────────────────────────────────────────────────────
from face_hub.engine.config import DEFAULT_SETTINGS, get_default_settings

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
    "InferenceError",
    "CameraError",
    "DatabaseError",
    "RecognitionError",
    # Config
    "DEFAULT_SETTINGS",
    "get_default_settings",
]
