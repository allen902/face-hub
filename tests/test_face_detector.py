"""
Test FaceDetector model loading and device fallback.
SKIP in CI (requires insightface model download + GPU).
Run locally only: pytest tests/test_face_detector.py
"""
import pytest
import numpy as np
from face_hub import FaceDetector


class TestFaceDetectorConstruction:
    def test_cpu_init(self):
        """Detector should load on CPU without error."""
        d = FaceDetector(device="cpu", det_size=320)
        assert d.device == "cpu"
        assert d.app is not None
        assert d.det_size == 320

    def test_auto_falls_back_to_cpu(self):
        """device='auto' should at minimum succeed (even if no GPU)."""
        d = FaceDetector(device="auto", det_size=320)
        assert d.app is not None

    def test_detect_on_dummy_frame(self, sample_frame):
        """Detection should return list (may be empty on random noise)."""
        d = FaceDetector(device="cpu", det_size=320)
        results = d.detect(sample_frame)
        assert isinstance(results, list)

    def test_detect_with_embeddings_dummy(self, sample_frame):
        """Embedding extraction should return list."""
        d = FaceDetector(device="cpu", det_size=320)
        results = d.detect_with_embeddings(sample_frame)
        assert isinstance(results, list)
