"""
FaceHub type definitions.
All public API return types are dataclasses defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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


# ── Photo classification results ────────────────────────────────

@dataclass
class PhotoFace:
    """
    One face found in one photo.

    label is either a registered person's name, an anonymous cluster
    label ("person_001", ...), or UNKNOWN_SENTINEL if the face could
    not be assigned.
    """
    photo_id: str
    bbox: BBox
    det_confidence: float             # detection confidence 0.0~1.0
    label: str                        # person name / cluster label / UNKNOWN_SENTINEL
    similarity: float                 # cosine similarity to the matched person / cluster centroid


@dataclass
class PhotoGroup:
    """
    A group of photos that contain the same person.

    photo_ids are unique and kept in insertion order; a photo containing
    several different people appears in several groups.
    """
    label: str                        # person name or cluster label
    photo_ids: List[str] = field(default_factory=list)
    face_count: int = 0

    @property
    def photo_count(self) -> int:
        return len(self.photo_ids)


@dataclass
class PhotoClassificationResult:
    """
    Returned by PhotoClassifier.classify_photos().
    """
    groups: Dict[str, PhotoGroup] = field(default_factory=dict)
    faces: List[PhotoFace] = field(default_factory=list)
    no_face_photos: List[str] = field(default_factory=list)      # photos with no usable face
    unreadable_photos: List[str] = field(default_factory=list)   # photos that failed to decode
    total_photos: int = 0

    @property
    def labels(self) -> List[str]:
        """All group labels in first-appearance order."""
        return list(self.groups.keys())

    @property
    def total_faces(self) -> int:
        return len(self.faces)

    def photos_of(self, label: str) -> List[str]:
        """Photo ids assigned to a person / cluster (empty if unknown label)."""
        group = self.groups.get(label)
        return list(group.photo_ids) if group else []

    def labels_of(self, photo_id: str) -> List[str]:
        """All person / cluster labels found in one photo."""
        seen: List[str] = []
        for face in self.faces:
            if face.photo_id == photo_id and face.label not in seen:
                seen.append(face.label)
        return seen

    def summary(self) -> Dict[str, int]:
        """Label → photo count, plus bookkeeping entries."""
        out = {label: g.photo_count for label, g in self.groups.items()}
        if self.no_face_photos:
            out["__no_face__"] = len(self.no_face_photos)
        if self.unreadable_photos:
            out["__unreadable__"] = len(self.unreadable_photos)
        return out


@dataclass
class ExportResult:
    """
    Returned by export_to_folders().

    exported maps each folder label to the file paths written inside it.
    A multi-person photo is exported into every group it belongs to, so
    total_files may exceed the number of source photos.
    """
    exported: Dict[str, List[str]] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)      # photo ids that are not existing files
    errors: Dict[str, str] = field(default_factory=dict)  # photo id → error message

    @property
    def total_files(self) -> int:
        return sum(len(paths) for paths in self.exported.values())

    @property
    def labels(self) -> List[str]:
        return list(self.exported.keys())
