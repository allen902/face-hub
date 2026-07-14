"""
Detector protocol — abstract interface for face detection + embedding.

Built-in implementation: face_hub.engine.face_detector.FaceDetector (insightface).
Users can implement this protocol to plug in custom models (YOLO, MediaPipe,
commercial SDK, etc.) without modifying the rest of the pipeline.
"""

from __future__ import annotations

from typing import Protocol, List, runtime_checkable
import numpy as np

from face_hub.types import DetectionResult, DetectionWithEmbedding


@runtime_checkable
class DetectorProtocol(Protocol):
    """
    Detector protocol — abstract interface for face detection + embedding.

    Built-in implementation: face_hub.engine.face_detector.FaceDetector (insightface).
    Users can implement this protocol to plug in custom models (YOLO, MediaPipe,
    commercial SDK, etc.) without modifying the rest of the pipeline.
    """

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect all faces in a BGR frame.
        
        Default fallback: Calls detect_with_embeddings and downgrades the results.
        """
        # 真正可运行的默认回退逻辑，避免返回 None 导致崩溃
        try:
            results = self.detect_with_embeddings(frame)
            # 自动将 DetectionWithEmbedding 转换为基础的 DetectionResult
            return [
                DetectionResult(
                    bbox=r.bbox, 
                    confidence=r.confidence, 
                    landmarks=getattr(r, 'landmarks', None)
                ) 
                for r in results
            ]
        except NotImplementedError:
            return []

    def detect_with_embeddings(
        self, frame: np.ndarray
    ) -> List[DetectionWithEmbedding]:
        """
        Detect faces and extract feature embeddings in one pass.

        This is the primary method called by the pipeline. Implementations
        should return embeddings that are L2-normalized 512-dim float32 vectors.
        """
        # 显式抛出异常，强制要求实现类必须重写此方法
        raise NotImplementedError("Subclasses must implement detect_with_embeddings")

    def reload_model(self, det_size: int | None = None) -> None:
        """
        Reload the underlying model with new parameters (optional).

        Called when the user changes detection resolution at runtime.
        Default no-op is acceptable if the implementation doesn't support hot-reload.
        """
        pass
