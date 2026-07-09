"""Test FaceTracker IoU matching, majority vote, identity confirmation."""

import numpy as np
from face_hub import FaceTracker, UNKNOWN_SENTINEL, BBox, DetectionWithEmbedding


def make_detection(x1, y1, x2, y2, embedding=None, quality=True):
    return DetectionWithEmbedding(
        bbox=BBox(x1, y1, x2, y2),
        confidence=0.95,
        embedding=embedding if embedding is not None else np.random.randn(512).astype(np.float32),
        quality_pass=quality,
    )


class TestFaceTracker:
    def test_init(self):
        t = FaceTracker(smooth_frames=5)
        assert t.track_count == 0

    def test_first_detection_creates_track(self):
        t = FaceTracker(smooth_frames=3)
        dets = [make_detection(10, 10, 100, 100)]
        results = t.update(dets)
        assert len(results) == 1
        assert results[0].name == UNKNOWN_SENTINEL
        assert results[0].is_confirmed is False

    def test_iou_matching_same_person(self):
        t = FaceTracker(smooth_frames=3, iou_threshold=0.20)
        dets1 = [make_detection(10, 10, 100, 100)]
        t.update(dets1)

        # Slightly moved bbox — should match same track
        dets2 = [make_detection(12, 8, 102, 98)]
        results = t.update(dets2)
        assert len(results) == 1
        assert t.track_count == 1  # still one track, not two

    def test_no_iou_creates_new_track(self):
        t = FaceTracker(smooth_frames=3, iou_threshold=0.20)
        dets1 = [make_detection(10, 10, 100, 100)]
        t.update(dets1)

        # Far away detection — new track
        dets2 = [make_detection(300, 300, 400, 400)]
        results = t.update(dets2)
        assert len(results) == 2  # both tracks alive (old not stale yet)

    def test_stale_track_removal(self):
        t = FaceTracker(smooth_frames=3, max_missed=2)
        dets = [make_detection(10, 10, 100, 100)]
        t.update(dets)
        assert t.track_count == 1

        # Two empty updates → track becomes stale
        t.update([])
        assert t.track_count == 1  # 1 miss
        t.update([])
        assert t.track_count == 0  # 2 misses = stale

    def test_reset(self):
        t = FaceTracker()
        t.update([make_detection(0, 0, 10, 10)])
        assert t.track_count == 1
        t.reset()
        assert t.track_count == 0
