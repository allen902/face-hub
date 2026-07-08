"""
Face detection module.
Uses insightface RetinaFace for accurate face detection.
Supports GPU acceleration (DirectML/CUDA) with automatic CPU fallback.
Built-in face quality assessment (blur detection) and model warm-up.
"""

import sys
from typing import List
import threading
import logging

import cv2
import numpy as np

from face_vision.types import DetectionResult, DetectionWithEmbedding, BBox
from face_vision.exceptions import ModelLoadError, InferenceError

logger = logging.getLogger("face_vision.detector")


class FaceDetector:
    """Face detector based on insightface RetinaFace."""

    def __init__(self, confidence=0.50, device="auto", det_size=640,
                 quality_filter=True, min_face_size=80):
        if device == "auto":
            device = "cuda"
        self.confidence = confidence
        self.device = device
        self.det_size = det_size
        self.quality_filter = quality_filter
        self.min_face_size = min_face_size
        self.app = None
        self._lock = threading.Lock()
        self._inference_error_count = 0
        self._gpu_name = "CPU"
        self._load_model()
        self._warmup()

    def _resolve_providers(self, force_cpu=False):
        """Resolve ONNX Runtime execution providers."""
        import onnxruntime as ort

        available = ort.get_available_providers()

        if self.device == "cuda" and not force_cpu:
            if 'CUDAExecutionProvider' in available:
                return ['CUDAExecutionProvider', 'CPUExecutionProvider'], 0, "GPU(CUDA)"
            elif 'DmlExecutionProvider' in available:
                return ['DmlExecutionProvider', 'CPUExecutionProvider'], 0, "GPU(DirectML)"
            else:
                logger.info("No usable GPU found, using CPU")
                return ['CPUExecutionProvider'], -1, "CPU (no usable GPU)"
        else:
            if force_cpu:
                logger.warning("Inference falling back to CPU mode")
            logger.info("Using CPU")
            return ['CPUExecutionProvider'], -1, "CPU"

    def _create_app(self, det_size):
        """Create and prepare a FaceAnalysis instance."""
        from insightface.app import FaceAnalysis

        providers, ctx_id, gpu_name = self._resolve_providers()
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        return app, gpu_name

    def _load_model(self, fallback_to_cpu=False):
        """Load the model, auto-detecting CUDA > DirectML > CPU."""
        try:
            self.app, gpu_name = self._create_app(self.det_size)
            logger.info("Model loaded (device=%s, det_size=%d)", gpu_name, self.det_size)
        except ImportError as e:
            raise ModelLoadError("insightface is not installed") from e
        except Exception as e:
            if not fallback_to_cpu and self.device == "cuda":
                logger.warning("GPU load failed: %s", e)
                logger.info("Falling back to CPU…")
                self.device = "cpu"
                self._load_model(fallback_to_cpu=True)
            else:
                raise ModelLoadError(f"Failed to load FaceDetector model: {e}") from e

    def _warmup(self):
        """Warm up the model with a dummy inference and validate it works."""
        try:
            dummy = np.random.randint(0, 255, (self.det_size, self.det_size, 3), dtype=np.uint8)
            with self._lock:
                faces = self.app.get(dummy)
            logger.info("Warm-up complete (detected %d fake faces)", len(faces))
        except Exception as e:
            err_msg = str(e)
            logger.warning("Warm-up failed (size=%d): %s", self.det_size, err_msg)

            # DirectML 1.24.x has a Reshape bug on some input sizes (e.g. 480).
            # Retry with a fresh app at 640, which is known to work.
            if self.device == "cuda" and "UnicodeDecodeError" in type(e).__name__:
                try:
                    logger.warning("DirectML incompatible with det_size=%d, retrying with 640", self.det_size)
                    new_app, _ = self._create_app(640)
                    dummy2 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    new_app.get(dummy2)
                    self.app = new_app
                    self.det_size = 640
                    logger.info("Auto-adjusted to det_size=640 (DirectML compat)")
                    return
                except Exception as inner:
                    logger.error("640 retry also failed: %s", inner)
                logger.warning("DirectML inference failed, falling back to CPU…")
                self.device = "cpu"
                self._load_model(fallback_to_cpu=True)
            elif self.device == "cuda":
                logger.warning("Inference anomaly, attempting CPU switch…")
                self.device = "cpu"
                self._load_model(fallback_to_cpu=True)

    # ── Face quality assessment ───────────────────────────────────

    @staticmethod
    def _face_quality(face_roi):
        """
        Assess the quality of a face region.

        Returns:
            (is_good, blur_score) where blur_score is Laplacian variance.
            Higher is sharper: >100 sharp, >50 usable, <50 blurry.
        """
        if face_roi is None or face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
            return False, 0.0

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_good = blur_score >= 50.0
        return is_good, blur_score

    @staticmethod
    def _face_roi(frame, bbox, expand=0.20):
        """Crop a face region with a small expansion margin."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        dw = int((x2 - x1) * expand)
        dh = int((y2 - y1) * expand)
        ex1 = max(0, x1 - dw)
        ey1 = max(0, y1 - dh)
        ex2 = min(w, x2 + dw)
        ey2 = min(h, y2 + dh)

        if ex2 <= ex1 or ey2 <= ey1:
            return np.array([])

        return frame[ey1:ey2, ex1:ex2]

    # ── Core detection API ────────────────────────────────────────

    def _run_inference(self, frame):
        """Run inference with automatic fallback on repeated failures."""
        with self._lock:
            faces = self.app.get(frame)
        return faces

    def _handle_inference_error(self, e, frame):
        """Handle an inference error, falling back to CPU if needed."""
        self._inference_error_count += 1
        err_type = type(e).__name__

        if self.device == "cuda":
            logger.warning("GPU inference error (%s: %s), switching to CPU…", err_type, e)
            try:
                self.device = "cpu"
                self._load_model(fallback_to_cpu=True)
                with self._lock:
                    return self.app.get(frame)
            except Exception as e2:
                raise InferenceError(f"Inference failed on both GPU and CPU: {e2}") from e2
        raise InferenceError(f"Inference error on {self.device}: {e}") from e

    def detect(self, frame) -> List[DetectionResult]:
        """
        Detect all faces in the frame.

        Returns:
            List of DetectionResult, coordinates as integers.
        """
        if self.app is None or frame is None or frame.size == 0:
            return []

        try:
            faces = self._run_inference(frame)
        except Exception as e:
            try:
                faces = self._handle_inference_error(e, frame)
            except Exception:
                return []

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            conf = float(face.det_score) if hasattr(face, 'det_score') else 0.95
            if conf >= self.confidence:
                results.append(DetectionResult(
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=conf,
                ))

        if len(faces) > 0 and len(results) == 0:
            raw_confs = [float(f.det_score) if hasattr(f, 'det_score') else 0.95 for f in faces]
            logger.debug(
                "detect: %d faces all below threshold %.2f (confidences: %s)",
                len(faces), self.confidence, [f"{c:.2f}" for c in raw_confs]
            )
        return results

    def detect_with_embeddings(self, frame) -> List[DetectionWithEmbedding]:
        """
        Detect faces and extract embeddings in one shared inference pass.

        Returns:
            List of DetectionWithEmbedding.
        """
        if self.app is None or frame is None or frame.size == 0:
            return []

        try:
            faces = self._run_inference(frame)
        except Exception as e:
            try:
                faces = self._handle_inference_error(e, frame)
            except Exception:
                return []

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            det_conf = float(face.det_score) if hasattr(face, 'det_score') else 0.95

            if det_conf < self.confidence:
                continue

            emb = face.normed_embedding if hasattr(face, 'normed_embedding') else None

            quality_pass = True
            if self.quality_filter:
                face_w = x2 - x1
                face_h = y2 - y1
                if face_w < self.min_face_size or face_h < self.min_face_size:
                    quality_pass = False
                else:
                    roi = self._face_roi(frame, (x1, y1, x2, y2), expand=0.10)
                    is_clear, _blur = self._face_quality(roi)
                    if not is_clear:
                        quality_pass = False

            results.append(DetectionWithEmbedding(
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=det_conf,
                embedding=emb,
                quality_pass=quality_pass,
            ))

        if len(faces) > 0 and len(results) == 0:
            raw_confs = [float(f.det_score) if hasattr(f, 'det_score') else 0.95 for f in faces]
            logger.debug(
                "detect_with_emb: %d faces all below threshold %.2f (confidences: %s)",
                len(faces), self.confidence, [f"{c:.2f}" for c in raw_confs]
            )
        return results

    def extract_face_roi(self, frame, face_rect):
        """Crop a face region with a small expansion (legacy-compatible interface)."""
        x1, y1, x2, y2, _ = face_rect
        return self._face_roi(frame, (x1, y1, x2, y2), expand=0.20)

    def reload_model(self, det_size=None):
        """Reload the model with new parameters (used for runtime resolution changes)."""
        if det_size is not None:
            self.det_size = det_size
        self._load_model()
        self._warmup()
