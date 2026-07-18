"""Test the DetectorProtocol default method implementations."""

import numpy as np

from face_hub.detector_protocol import DetectorProtocol
from face_hub.types import BBox, DetectionResult, DetectionWithEmbedding


class _StubDetector(DetectorProtocol):
    """Minimal implementation: only the abstract embedding method."""

    def detect_with_embeddings(self, frame: np.ndarray):
        return [
            DetectionWithEmbedding(
                bbox=BBox(x1=1, y1=2, x2=3, y2=4),
                confidence=0.9,
                embedding=None,
            )
        ]


class TestDetectorProtocol:
    def test_default_detect_downgrades_embeddings(self):
        """The default detect() must not crash and must drop embeddings."""
        detector = _StubDetector()
        results = detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))
        assert len(results) == 1
        assert isinstance(results[0], DetectionResult)
        assert results[0].bbox == BBox(x1=1, y1=2, x2=3, y2=4)
        assert results[0].confidence == 0.9

    def test_default_reload_model_is_noop(self):
        detector = _StubDetector()
        assert detector.reload_model(det_size=320) is None
