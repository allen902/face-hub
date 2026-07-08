"""
FaceVision — Real-time Face Recognition Library

Usage:
    from face_vision import (
        FaceVisionPipeline, FaceDetector, FaceRecognizer,
        CameraThread, FaceTracker, FaceDatabase,
    )
    from face_vision.types import UNKNOWN_SENTINEL, PipelineResult, TrackedFace
"""

__version__ = "1.0.0"
__author__ = "AllenDeng"

# ── Core components (from engine subpackage) ─────────────────
from face_vision.engine.camera import CameraThread
from face_vision.engine.face_detector import FaceDetector
from face_vision.engine.face_recognizer import FaceRecognizer
from face_vision.engine.face_tracker import FaceTracker
from face_vision.engine.face_database import FaceDatabase

# ── Pipeline (from face_vision/ main package) ───────────────────
from face_vision.pipeline import FaceVisionPipeline
from face_vision.detector_protocol import DetectorProtocol

# ── Types ──────────────────────────────────────────────────────
from face_vision.types import (
    UNKNOWN_SENTINEL,
    BBox,
    DetectionResult,
    DetectionWithEmbedding,
    TrackedFace,
    PipelineResult,
)

# ── Exceptions ─────────────────────────────────────────────────
from face_vision.exceptions import (
    FaceVisionError,
    ModelLoadError,
    InferenceError,
    CameraError,
    DatabaseError,
    RecognitionError,
)

# ── Config ─────────────────────────────────────────────────────
from face_vision.engine.config import DEFAULT_SETTINGS, get_default_settings

__all__ = [
    # Core
    "CameraThread",
    "FaceDetector",
    "FaceRecognizer",
    "FaceTracker",
    "FaceDatabase",
    "FaceVisionPipeline",
    "DetectorProtocol",
    # Types
    "UNKNOWN_SENTINEL",
    "BBox",
    "DetectionResult",
    "DetectionWithEmbedding",
    "TrackedFace",
    "PipelineResult",
    # Exceptions
    "FaceVisionError",
    "ModelLoadError",
    "InferenceError",
    "CameraError",
    "DatabaseError",
    "RecognitionError",
    # Config
    "DEFAULT_SETTINGS",
    "get_default_settings",
]
