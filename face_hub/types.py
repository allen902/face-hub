"""
FaceHub type definitions.
All public API return types are dataclasses defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ── Global constants ──────────────────────────────────────────────

UNKNOWN_SENTINEL = "unknown"
"""
Stable sentinel for "not recognized as any registered person".
Import via: from face_hub.types import UNKNOWN_SENTINEL
"""


# ── Geometry ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class BBox:
    """Immutable bounding box (hashable)."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_tuple(self) -> tuple:
        """Convert to (x1, y1, x2, y2) for OpenCV functions."""
        return (self.x1, self.y1, self.x2, self.y2)


# ── Detection results ─────────────────────────────────────────────

@dataclass
class DetectionResult:
    """
    Returned by FaceDetector.detect().
    """
    bbox: BBox
    confidence: float          # detection confidence 0.0~1.0

    @classmethod
    def from_tuple(cls, t: tuple) -> "DetectionResult":
        return cls(bbox=BBox(x1=t[0], y1=t[1], x2=t[2], y2=t[3]), confidence=t[4])


@dataclass
class DetectionWithEmbedding:
    """
    Returned by FaceDetector.detect_with_embeddings().
    embedding is None if feature extraction failed (rare).
    """
    bbox: BBox
    confidence: float                               # detection confidence 0.0~1.0
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    quality_pass: bool = True                       # quality filter passed

    @classmethod
    def from_tuple(cls, t: tuple) -> "DetectionWithEmbedding":
        return cls(
            bbox=BBox(x1=t[0], y1=t[1], x2=t[2], y2=t[3]),
            confidence=t[4],
            embedding=t[5],
            quality_pass=t[6],
        )

    @property
    def has_embedding(self) -> bool:
        return self.embedding is not None


# ── Tracking results ──────────────────────────────────────────────

@dataclass
class TrackedFace:
    """
    Returned by FaceTracker.update().
    """
    track_id: int
    bbox: BBox
    name: str                       # recognized name, or UNKNOWN_SENTINEL
    confidence: float               # cosine similarity 0.0~1.0
    det_confidence: float           # detection confidence
    is_confirmed: bool              # identity confirmed via majority vote
    quality_pass: bool

    @property
    def is_known(self) -> bool:
        return self.name != UNKNOWN_SENTINEL


# ── Pipeline result ───────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Returned by FaceHubPipeline.process_frame().
    """
    frame: np.ndarray = field(repr=False)
    raw_detections: List[DetectionWithEmbedding] = field(default_factory=list)
    tracked_faces: List[TrackedFace] = field(default_factory=list)
    fps: float = 0.0

    @property
    def known_faces(self) -> List[TrackedFace]:
        return [t for t in self.tracked_faces if t.is_known]

    @property
    def unknown_count(self) -> int:
        return sum(1 for t in self.tracked_faces if not t.is_known)

    @property
    def total_faces(self) -> int:
        return len(self.tracked_faces)
