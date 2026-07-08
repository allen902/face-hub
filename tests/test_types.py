"""Test dataclass types and UNKNOWN_SENTINEL."""

import numpy as np
from face_vision.types import (
    UNKNOWN_SENTINEL, BBox, DetectionResult,
    DetectionWithEmbedding, TrackedFace, PipelineResult,
)


class TestBBox:
    def test_create(self):
        b = BBox(x1=10, y1=20, x2=100, y2=200)
        assert b.x1 == 10
        assert b.width == 90
        assert b.height == 180

    def test_to_tuple(self):
        b = BBox(1, 2, 3, 4)
        assert b.to_tuple() == (1, 2, 3, 4)

    def test_frozen(self):
        b = BBox(1, 2, 3, 4)
        try:
            b.x1 = 99
            assert False, "Should raise FrozenInstanceError"
        except Exception:
            pass

    def test_hashable(self):
        a = BBox(1, 2, 3, 4)
        b = BBox(1, 2, 3, 4)
        c = BBox(5, 6, 7, 8)
        assert hash(a) == hash(b)
        assert hash(a) != hash(c)
        d = {a: "test"}
        assert d[b] == "test"


class TestDetectionResult:
    def test_from_tuple(self):
        r = DetectionResult.from_tuple((10, 20, 100, 200, 0.95))
        assert r.bbox.x1 == 10
        assert r.confidence == 0.95


class TestDetectionWithEmbedding:
    def test_from_tuple(self, sample_encoding):
        r = DetectionWithEmbedding.from_tuple(
            (10, 20, 100, 200, 0.95, sample_encoding, True)
        )
        assert r.bbox.width == 90
        assert r.confidence == 0.95
        assert r.has_embedding is True
        assert r.quality_pass is True

    def test_no_embedding(self):
        r = DetectionWithEmbedding.from_tuple((0, 0, 50, 50, 0.8, None, False))
        assert r.has_embedding is False


class TestTrackedFace:
    def test_is_known(self):
        known = TrackedFace(
            track_id=1, bbox=BBox(0, 0, 10, 10),
            name="Alice", confidence=0.9, det_confidence=0.95,
            is_confirmed=True, quality_pass=True,
        )
        assert known.is_known is True

    def test_is_unknown(self):
        unknown = TrackedFace(
            track_id=2, bbox=BBox(0, 0, 10, 10),
            name=UNKNOWN_SENTINEL, confidence=0.0, det_confidence=0.8,
            is_confirmed=False, quality_pass=True,
        )
        assert unknown.is_known is False


class TestPipelineResult:
    def test_known_faces_filter(self):
        faces = [
            TrackedFace(1, BBox(0,0,10,10), "Alice", 0.9, 0.95, True, True),
            TrackedFace(2, BBox(10,10,20,20), UNKNOWN_SENTINEL, 0.0, 0.8, False, True),
        ]
        r = PipelineResult(frame=np.zeros((10,10,3)), tracked_faces=faces)
        assert len(r.known_faces) == 1
        assert r.unknown_count == 1
        assert r.total_faces == 2


class TestUnknownSentinel:
    def test_value(self):
        assert UNKNOWN_SENTINEL == "unknown"
