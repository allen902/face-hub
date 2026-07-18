"""Test FaceHubPipeline.det_interval (frame-skip detection) with stub components.

No camera, no insightface, no model download required.
"""

import numpy as np
import pytest

from face_hub import (
    FaceHubPipeline, FaceRecognizer, FaceTracker, FaceDatabase, PipelineResult,
)


class MockCamera:
    """Minimal camera stub for pipeline tests."""
    running = False
    def start(self):
        self.running = True
    def stop(self):
        self.running = False
    def get_frame(self, timeout=None, copy=True):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class CountingDetector:
    """Detector stub that counts how often full inference ran."""
    def __init__(self):
        self.calls = 0

    def detect_with_embeddings(self, frame):
        self.calls += 1
        return []


def _make_pipeline(temp_db_paths, detector, det_interval):
    db_path, enc_path = temp_db_paths
    db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
    pipeline = FaceHubPipeline(
        MockCamera(), detector, FaceRecognizer(), FaceTracker(), db,
        det_interval=det_interval,
    )
    pipeline.start()
    return pipeline


class TestDetInterval:
    def test_default_detects_every_frame(self, temp_db_paths):
        detector = CountingDetector()
        pipeline = _make_pipeline(temp_db_paths, detector, det_interval=1)
        try:
            for _ in range(5):
                result = pipeline.process_frame()
                assert isinstance(result, PipelineResult)
        finally:
            pipeline.stop()
        assert detector.calls == 5

    def test_interval_skips_detection(self, temp_db_paths):
        detector = CountingDetector()
        pipeline = _make_pipeline(temp_db_paths, detector, det_interval=3)
        try:
            for _ in range(7):
                pipeline.process_frame()
        finally:
            pipeline.stop()
        # Detection runs on frames 1, 4, 7 → 3 calls for 7 frames
        assert detector.calls == 3

    def test_skipped_frames_reuse_tracked_faces(self, temp_db_paths):
        detector = CountingDetector()
        pipeline = _make_pipeline(temp_db_paths, detector, det_interval=2)
        try:
            r1 = pipeline.process_frame()
            r2 = pipeline.process_frame()
            assert r2.tracked_faces is r1.tracked_faces
            assert r2.raw_detections == []
        finally:
            pipeline.stop()

    def test_invalid_interval_rejected(self, temp_db_paths):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        with pytest.raises(ValueError, match="det_interval"):
            FaceHubPipeline(
                MockCamera(), CountingDetector(), FaceRecognizer(),
                FaceTracker(), db, det_interval=0,
            )

    def test_reset_tracker_forces_fresh_detection(self, temp_db_paths):
        detector = CountingDetector()
        pipeline = _make_pipeline(temp_db_paths, detector, det_interval=10)
        try:
            pipeline.process_frame()      # detects (first frame)
            pipeline.process_frame()      # skipped
            pipeline.reset_tracker()
            pipeline.process_frame()      # must detect again, not reuse
        finally:
            pipeline.stop()
        assert detector.calls == 2
