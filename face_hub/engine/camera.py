"""
Camera capture thread module.
Runs camera capture in a dedicated thread, provides thread-safe latest-frame access.

Platform backends:
  - Windows:  DShow (cv2.CAP_DSHOW)
  - macOS:    AVFoundation (cv2.CAP_AVFOUNDATION)
  - Linux:    V4L2 (cv2.CAP_V4L2)
"""

import cv2
import threading
import time
import sys
import logging
from collections import deque

logger = logging.getLogger("face_hub.camera")


class CameraThread:
    """Camera capture thread — pure acquisition, no processing.

    Usage::

        with CameraThread(camera_id=0) as cam:
            frame = cam.get_frame()
    """

    def __init__(self, camera_id=0, width=640, height=360, fps=30):
        if not isinstance(camera_id, int) or camera_id < 0:
            raise ValueError(f"camera_id must be a non-negative int, got {camera_id}")
        if not isinstance(width, int) or width <= 0:
            raise ValueError(f"width must be a positive int, got {width}")
        if not isinstance(height, int) or height <= 0:
            raise ValueError(f"height must be a positive int, got {height}")
        if not isinstance(fps, (int, float)) or fps <= 0:
            raise ValueError(f"fps must be a positive number, got {fps}")

        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.cap = None
        self.running = False
        self._frame_buffer = deque(maxlen=1)
        self.thread = None
        self._actual_fps = 0.0
        self._fps_lock = threading.Lock()
        # Condition protects _frame_buffer and lets multiple consumers
        # all receive the same new-frame notification without "stealing".
        self._frame_cond = threading.Condition(threading.Lock())

    # ── Context manager ─────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    # ── Platform-aware backend selection ──────────────────────

    @staticmethod
    def _get_backend():
        """Return the optimal OpenCV backend for the current platform."""
        if sys.platform == "win32":
            return cv2.CAP_DSHOW
        elif sys.platform == "darwin":
            # AVFoundation is default on macOS, but explicit avoids warnings
            return cv2.CAP_AVFOUNDATION
        else:
            # Linux: V4L2 is the standard
            return cv2.CAP_V4L2

    # ── Public API ────────────────────────────────────────────

    @property
    def actual_fps(self):
        with self._fps_lock:
            return self._actual_fps

    def start(self):
        """Start camera capture."""
        if self.running:
            return

        self.cap = cv2.VideoCapture(self.camera_id, self._get_backend())
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            from face_hub.exceptions import CameraError
            raise CameraError(
                f"Cannot open camera (ID={self.camera_id})",
                camera_id=self.camera_id,
            )

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info("Actual resolution: %dx%d", int(actual_w), int(actual_h))

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop camera capture."""
        self.running = False
        # Wake any threads blocked in get_frame() so they can see
        # running=False and return None. Also drop the buffered frame so a
        # later restart never serves a stale frame from the previous session.
        with self._frame_cond:
            self._frame_buffer.clear()
            self._frame_cond.notify_all()
        if self.thread:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                # The capture thread is still blocked in cap.read().
                # Releasing the handle now would race the in-flight read
                # (potential crash in OpenCV) — defer release to GC instead.
                logger.warning(
                    "Capture thread did not exit within 2s; "
                    "deferring camera release to avoid a use-after-free"
                )
                return
            self.thread = None
        if self.cap:
            self.cap.release()
        self.cap = None

    def get_frame(self, timeout=0.05, copy=True):
        """
        Thread-safe latest frame retrieval.

        Multiple threads can call this concurrently — each receives the
        same frame (as a independent copy) without "stealing" it from
        other consumers.

        Args:
            timeout: Max seconds to wait for a new frame.
            copy: If True (default), returns a safe copy. Set False for
                  zero-copy mode — caller must NOT mutate the returned frame.

        Returns:
            Frame as numpy array, or None if timeout expired.
        """
        with self._frame_cond:
            # Wait until a frame is available or the camera stops.
            while len(self._frame_buffer) == 0 and self.running:
                if not self._frame_cond.wait(timeout=timeout):
                    # Timed out — no frame arrived within the deadline.
                    return None

            if not self.running or len(self._frame_buffer) == 0:
                return None

            frame = self._frame_buffer[0]
            return frame.copy() if copy else frame

    def _loop(self):
        """Main capture loop."""
        interval = 1.0 / max(self.target_fps, 1)
        last_time = time.time()
        fps_counter = 0
        fps_timer = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)

            with self._frame_cond:
                self._frame_buffer.append(frame)
                # Wake all waiting consumers so each gets a copy.
                self._frame_cond.notify_all()

            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                with self._fps_lock:
                    self._actual_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            elapsed = now - last_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
            last_time = time.time()

    @staticmethod
    def list_cameras(max_test=5):
        """List available camera indices (suppresses backend warnings)."""
        try:
            cv_log_level = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LEVEL_ERROR)
        except Exception:
            cv_log_level = None

        available = []
        for i in range(max_test):
            try:
                cap = cv2.VideoCapture(i, CameraThread._get_backend())
                if cap.isOpened():
                    available.append(i)
                    cap.release()
            except Exception:
                pass

        if cv_log_level is not None:
            try:
                cv2.utils.logging.setLogLevel(cv_log_level)
            except Exception:
                pass
        return available
