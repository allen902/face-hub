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
import os
import sys
import logging
from collections import deque

# Suppress DShow hardware transform warnings on Windows
if sys.platform == "win32":
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

logger = logging.getLogger("face_vision.camera")


class CameraThread:
    """Camera capture thread — pure acquisition, no processing."""

    def __init__(self, camera_id=0, width=640, height=360, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.cap = None
        self.running = False
        self._frame_buffer = deque(maxlen=1)
        self.lock = threading.Lock()
        self.thread = None
        self._actual_fps = 0.0
        self._fps_lock = threading.Lock()
        self._frame_available = threading.Event()

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
            from face_vision.exceptions import CameraError
            raise CameraError(f"Cannot open camera (ID={self.camera_id})")

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info("Actual resolution: %dx%d", int(actual_w), int(actual_h))

        self.running = True
        self._frame_available.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop camera capture."""
        self.running = False
        self._frame_available.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        self.cap = None

    def get_frame(self, timeout=0.05, copy=True):
        """
        Thread-safe latest frame retrieval.

        Args:
            timeout: Max seconds to wait for a new frame.
            copy: If True (default), returns a safe copy. Set False for
                  zero-copy mode — caller must NOT mutate the returned frame.

        Returns:
            Frame as numpy array, or None if timeout expired.
        """
        if self._frame_available.wait(timeout):
            with self.lock:
                if len(self._frame_buffer) > 0:
                    frame = self._frame_buffer[0]
                    self._frame_available.clear()
                    return frame.copy() if copy else frame
        return None

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

            with self.lock:
                self._frame_buffer.append(frame)
                self._frame_available.set()

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
