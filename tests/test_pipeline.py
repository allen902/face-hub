"""Test FaceHubPipeline integration (using mock camera)."""

import numpy as np
from face_hub import (
    FaceHubPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, PipelineResult,
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


class TestPipeline:
    def test_init(self, temp_db_paths):
        db_path, enc_path = temp_db_paths
        camera = MockCamera()
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        # Use CPU to avoid GPU dependency in tests
        detector = FaceDetector(device="cpu", det_size=320)
        recognizer = FaceRecognizer(tolerance=0.45)
        tracker = FaceTracker(smooth_frames=3)

        pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
        assert pipeline.is_running is False

    def test_start_stop(self, temp_db_paths):
        db_path, enc_path = temp_db_paths
        camera = MockCamera()
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        detector = FaceDetector(device="cpu", det_size=320)
        recognizer = FaceRecognizer()
        tracker = FaceTracker()

        pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
        pipeline.start()
        assert pipeline.is_running is True
        pipeline.stop()
        assert pipeline.is_running is False

    def test_process_frame_with_explicit_frame(self, temp_db_paths, sample_frame):
        db_path, enc_path = temp_db_paths
        camera = MockCamera()
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        detector = FaceDetector(device="cpu", det_size=320)
        recognizer = FaceRecognizer()
        tracker = FaceTracker()

        pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
        # process_frame with explicit frame (no camera needed)
        result = pipeline.process_frame(frame=sample_frame)
        assert isinstance(result, PipelineResult)
        assert result.fps >= 0.0

    def test_detect_only(self, temp_db_paths, sample_frame):
        db_path, enc_path = temp_db_paths
        camera = MockCamera()
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        detector = FaceDetector(device="cpu", det_size=320)
        recognizer = FaceRecognizer()
        tracker = FaceTracker()

        pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
        results = pipeline.detect_only(sample_frame)
        assert isinstance(results, list)

    def test_update_database_cache(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        camera = MockCamera()
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        detector = FaceDetector(device="cpu", det_size=320)
        recognizer = FaceRecognizer()
        tracker = FaceTracker()

        pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
        db.add_person("Test", "/tmp/test.jpg", sample_encoding)
        rebuilt = pipeline.update_database_cache()
        assert rebuilt is True
        assert "Test" in recognizer.cached_names
