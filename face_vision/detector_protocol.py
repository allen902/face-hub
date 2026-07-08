"""
Detector protocol — abstract interface for face detection + embedding.

Built-in implementation: face_vision.engine.face_detector.FaceDetector (insightface).
Users can implement this protocol to plug in custom models (YOLO, MediaPipe,
commercial SDK, etc.) without modifying the rest of the pipeline.
"""

from __future__ import annotations

from typing import Protocol, List, runtime_checkable
import numpy as np

from face_vision.types import DetectionResult, DetectionWithEmbedding


@runtime_checkable
class DetectorProtocol(Protocol):
    """
    Protocol for face detection and embedding extraction.

    Any object implementing these three methods can be passed to
    FaceVisionPipeline as the detector.

    Minimal implementation: just implement detect_with_embeddings().
    detect() and reload_model() have default no-op fallbacks.
    """

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect all faces in a BGR frame.

        Args:
            frame: BGR image as (H, W, 3) numpy array.

        Returns:
            List of DetectionResult, sorted by confidence descending.
        """
        ...

    def detect_with_embeddings(
        self, frame: np.ndarray
    ) -> List[DetectionWithEmbedding]:
        """
        Detect faces and extract feature embeddings in one pass.

        This is the primary method called by the pipeline. Implementations
        should return embeddings that are L2-normalized 512-dim float32 vectors
        (or set the confidence threshold in FaceRecognizer accordingly).

        Args:
            frame: BGR image as (H, W, 3) numpy array.

        Returns:
            List of DetectionWithEmbedding, sorted by confidence descending.
        """
        ...

    def reload_model(self, det_size: int = None) -> None:
        """
        Reload the underlying model with new parameters (optional).

        Called when the user changes detection resolution at runtime.
        Default no-op is acceptable if the implementation doesn't support
        hot-reload.
        """
        pass
