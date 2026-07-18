from __future__ import annotations

import time
import threading
import logging
from typing import Optional, List

import numpy as np

from face_hub.types import PipelineResult, DetectionResult, DetectionWithEmbedding
from face_hub.exceptions import FaceHubError
from face_hub.detector_protocol import DetectorProtocol

from face_hub.engine.face_recognizer import FaceRecognizer
from face_hub.engine.face_tracker import FaceTracker
from face_hub.engine.face_database import FaceDatabase
from face_hub.engine.camera import CameraThread

logger = logging.getLogger("face_hub.pipeline")


class FaceHubPipeline:
    """
    Full face recognition pipeline.

    Usage:
        pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
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
        det_interval: int = 1,
    ):
        """
        Args:
            det_interval: Run the expensive detection/embedding inference
                once every N frames and reuse the previous tracked result in
                between (bboxes stay frozen for N-1 frames). 1 = detect every
                frame (default, original behavior). 2-3 roughly doubles/triples
                pipeline throughput on CPU with little visible lag at 30 fps.
        """
        if not isinstance(det_interval, int) or det_interval < 1:
            raise ValueError(
                f"det_interval must be a positive int, got {det_interval}"
            )
        self.camera = camera
        self.detector = detector
        self.recognizer = recognizer
        self.tracker = tracker
        self.db = db
        self.det_interval = det_interval

        self._running = False
        self._lock = threading.Lock()
        self._frame_count = 0          # frames since last FPS tick (reset each second)
        self._total_frame_count = 0    # monotonically increasing total
        self._fps_timer = time.time()
        self._current_fps = 0.0
        self._last_db_version = -1     # skip redundant cache syncs
        self._frames_since_detect = 0  # frames since the last full detection
        self._last_tracked_faces = None  # reused on skipped (non-detection) frames

    # ── Lifecycle ────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start pipeline (starts camera if needed)."""
        with self._lock:
            if self._running:
                return
            if not self.camera.running:
                self.camera.start()
            self._running = True
            self._fps_timer = time.time()
            self._frame_count = 0
            logger.info("Pipeline started")

    def stop(self) -> None:
        """Stop pipeline (stops camera)."""
        with self._lock:
            self._running = False
            if self.camera.running:
                self.camera.stop()
            logger.info("Pipeline stopped")

    def reset_tracker(self) -> None:
        """Reset face tracker (e.g. after settings change)."""
        with self._lock:
            self.tracker.reset()
            self._last_tracked_faces = None
            logger.info("Tracker reset")

    # ── Database cache ───────────────────────────────────────────

    def update_database_cache(self) -> bool:
        """
        Sync recognizer encoding cache with database.
        Returns True if cache was rebuilt.
        """
        with self._lock:
            version = self.db.version
            known_encodings, known_names = self.db.get_encodings_and_names()
            rebuilt = self.recognizer.update_cache(
                known_encodings, known_names, version
            )
            if rebuilt:
                self._last_db_version = version
            return rebuilt

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
            Note: result.frame is a shared read-only buffer when it comes
            from the camera — call frame.copy() before drawing on it.
            On frames where detection is skipped (see det_interval),
            raw_detections is empty and tracked_faces repeats the result
            of the last detected frame.

        Thread-safe via internal lock.
        """
        with self._lock:
            if not self._running:
                return None
            return self._process_frame_impl(frame)

    def _process_frame_impl(self, frame: Optional[np.ndarray]) -> Optional[PipelineResult]:
        """Internal — caller must hold self._lock."""

        # 1. Acquire frame.
        # copy=False is safe here: the capture thread appends a brand-new
        # array every frame and never mutates a published one, so the frame
        # we hold cannot be overwritten. This avoids a full-frame memcpy per
        # call — do NOT mutate PipelineResult.frame (call .copy() to draw).
        if frame is None:
            frame = self.camera.get_frame(copy=False)
            if frame is None:
                return None

        try:
            # 2. Sync encoding cache (only when database version changes)
            current_version = self.db.version
            if current_version != self._last_db_version:
                known_encodings, known_names = self.db.get_encodings_and_names()
                self.recognizer.update_cache(
                    known_encodings, known_names, current_version
                )
                self._last_db_version = current_version

            # 3. Detect + track. Full inference runs every det_interval
            # frames; in between, the previous tracked result is reused so
            # the expensive model call is amortised (bboxes stay frozen for
            # at most det_interval-1 frames).
            self._frames_since_detect += 1
            if (self._frames_since_detect >= self.det_interval
                    or self._last_tracked_faces is None):
                detections = self.detector.detect_with_embeddings(frame)

                # 4. Track + recognize (batched inside the tracker)
                tracked_faces = self.tracker.update(
                    detections,
                    recognizer=self.recognizer,
                )
                self._last_tracked_faces = tracked_faces
                self._frames_since_detect = 0
            else:
                detections = []
                tracked_faces = self._last_tracked_faces

            # 5. FPS counter
            self._frame_count += 1
            self._total_frame_count += 1
            now = time.time()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = now

            # 6. Periodic debug log
            if self._total_frame_count % 30 == 0:
                n_valid = sum(1 for d in detections if d.quality_pass)
                logger.debug(
                    "Frame #%d: %d faces, %d quality-pass, %d tracks",
                    self._total_frame_count, len(detections), n_valid,
                    self.tracker.track_count,
                )

            return PipelineResult(
                frame=frame,
                raw_detections=detections,
                tracked_faces=tracked_faces,
                fps=self._current_fps,
            )

        except FaceHubError:
            raise
        except Exception as e:
            raise FaceHubError(f"Pipeline processing failed: {e}") from e

    # ── Convenience methods ──────────────────────────────────────

    def detect_only(self, frame: np.ndarray) -> List[DetectionResult]:
        """Run detection only (no tracking/recognition)."""
        with self._lock:
            return self.detector.detect(frame)

    def extract_embeddings(self, frame: np.ndarray) -> List[DetectionWithEmbedding]:
        """Run detection + embedding extraction (no tracking)."""
        with self._lock:
            return self.detector.detect_with_embeddings(frame)
