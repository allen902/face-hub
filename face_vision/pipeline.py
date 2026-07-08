from __future__ import annotations

import time
import threading
import logging
from typing import Optional

import numpy as np

from typing import List
from face_vision.types import PipelineResult, DetectionResult, DetectionWithEmbedding
from face_vision.exceptions import FaceVisionError
from face_vision.detector_protocol import DetectorProtocol

from face_vision.engine.face_recognizer import FaceRecognizer
from face_vision.engine.face_tracker import FaceTracker
from face_vision.engine.face_database import FaceDatabase
from face_vision.engine.camera import CameraThread

logger = logging.getLogger("face_vision.pipeline")


class FaceVisionPipeline:
    """
    Full face recognition pipeline.

    Usage:
        pipeline = FaceVisionPipeline(camera, detector, recognizer, tracker, db)
        pipeline.start()

        while True:
            result = pipeline.process_frame()
            if result is None:
                continue
            for face in result.known_faces:
                print(f"{face.name} ({face.confidence:.2%})")

        pipeline.stop()
    """

    def __init__(
        self,
        camera: CameraThread,
        detector: DetectorProtocol,  # ← accepts any object satisfying the protocol
        recognizer: FaceRecognizer,
        tracker: FaceTracker,
        db: FaceDatabase,
    ):
        self.camera = camera
        self.detector = detector
        self.recognizer = recognizer
        self.tracker = tracker
        self.db = db

        self._running = False
        self._lock = threading.Lock()
        self._frame_count = 0
        self._debug_frame_count = 0     # independent counter for debug logging
        self._fps_timer = time.time()
        self._current_fps = 0.0

    # ── Lifecycle ────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """Start pipeline (starts camera if needed)."""
        if self._running:
            return
        if not self.camera.running:
            self.camera.start()
        self._running = True
        self._fps_timer = time.time()
        self._frame_count = 0
        logger.info("Pipeline started")

    def stop(self):
        """Stop pipeline (stops camera)."""
        self._running = False
        if self.camera.running:
            self.camera.stop()
        logger.info("Pipeline stopped")

    def reset_tracker(self):
        """Reset face tracker (e.g. after settings change)."""
        self.tracker.reset()
        logger.info("Tracker reset")

    # ── Database cache ───────────────────────────────────────────

    def update_database_cache(self) -> bool:
        """
        Sync recognizer encoding cache with database.
        Returns True if cache was rebuilt.
        """
        known_encodings, known_names = self.db.get_encodings_and_names()
        return self.recognizer.update_cache(
            known_encodings, known_names, self.db.version
        )

    # ── Core processing ──────────────────────────────────────────

    def process_frame(
        self, frame: Optional[np.ndarray] = None
    ) -> Optional[PipelineResult]:
        """
        Process one frame through the full pipeline.

        Args:
            frame: BGR numpy array. If None, fetches from camera.

        Returns:
            PipelineResult, or None if no frame available.

        Thread-safe via internal lock.
        """
        with self._lock:
            return self._process_frame_impl(frame)

    def _process_frame_impl(self, frame) -> Optional[PipelineResult]:
        """Internal — caller must hold self._lock."""

        # 1. Acquire frame
        if frame is None:
            frame = self.camera.get_frame()
            if frame is None:
                return None

        try:
            # 2. Sync encoding cache
            known_encodings, known_names = self.db.get_encodings_and_names()
            self.recognizer.update_cache(
                known_encodings, known_names, self.db.version
            )

            # 3. Detect + extract embeddings
            detections = self.detector.detect_with_embeddings(frame)

            # 4. Track + recognize
            tracked_faces = self.tracker.update(
                detections,
                recognizer=self.recognizer,
            )

            # 5. FPS counter
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = now

            # 6. Periodic debug log (independent counter, not reset by FPS)
            self._debug_frame_count += 1
            if self._debug_frame_count % 30 == 0:
                n_valid = sum(1 for d in detections if d.quality_pass)
                logger.debug(
                    "Frame #%d: %d faces, %d quality-pass, %d tracks",
                    self._frame_count, len(detections), n_valid,
                    self.tracker.track_count,
                )

            return PipelineResult(
                frame=frame,
                raw_detections=detections,
                tracked_faces=tracked_faces,
                fps=self._current_fps,
            )

        except FaceVisionError:
            raise
        except Exception as e:
            raise FaceVisionError(f"Pipeline processing failed: {e}") from e

    # ── Convenience methods ──────────────────────────────────────

    def detect_only(self, frame: np.ndarray) -> "List[DetectionResult]":
        """Run detection only (no tracking/recognition)."""
        with self._lock:
            return self.detector.detect(frame)

    def extract_embeddings(self, frame: np.ndarray) -> "List[DetectionWithEmbedding]":
        """Run detection + embedding extraction (no tracking)."""
        with self._lock:
            return self.detector.detect_with_embeddings(frame)
