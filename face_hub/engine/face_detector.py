"""
Face detection module.
Uses insightface RetinaFace for accurate face detection.
Supports GPU acceleration (DirectML/CUDA) with automatic CPU fallback.
Built-in face quality assessment (blur detection) and model warm-up.
"""

from typing import List, Tuple
import threading
import logging

import cv2
import numpy as np

from face_hub.types import DetectionResult, DetectionWithEmbedding, BBox
from face_hub.exceptions import ModelLoadError, InferenceError

logger = logging.getLogger("face_hub.detector")

# Maximum consecutive inference errors before forcing CPU fallback.
_MAX_INFERENCE_ERRORS = 3


class FaceDetector:
    """Face detector based on insightface RetinaFace."""

    def __init__(self, confidence=0.50, device="auto", det_size=640,
                 quality_filter=True, min_face_size=100, blur_threshold=50.0):
        if not isinstance(confidence, (int, float)) or not (0 < confidence <= 1):
            raise ValueError(f"confidence must be in (0, 1], got {confidence}")
        if not isinstance(det_size, int) or det_size < 160:
            raise ValueError(f"det_size must be an int >= 160, got {det_size}")
        if not isinstance(min_face_size, int) or min_face_size < 0:
            raise ValueError(f"min_face_size must be a non-negative int, got {min_face_size}")
        if not isinstance(blur_threshold, (int, float)) or blur_threshold < 0:
            raise ValueError(f"blur_threshold must be >= 0, got {blur_threshold}")

        self.confidence = confidence
        self.device = device  # kept as-is; resolved to actual device in _load_model
        self.det_size = det_size
        self.quality_filter = quality_filter
        self.min_face_size = min_face_size
        self.blur_threshold = float(blur_threshold)
        self.app = None
        self._lock = threading.Lock()
        self._inference_error_count = 0
        self._gpu_name = "CPU"
        self._load_model()
        self._warmup()

    def _resolve_providers(self, force_cpu=False):
        """Resolve ONNX Runtime execution providers.

        Returns (providers, ctx_id, device_label).
        When self.device is "auto", tries CUDA → DirectML → CPU.
        When self.device is "cuda", requires a GPU or falls back to CPU
        (with a warning).
        When self.device is "cpu", always uses CPU.
        """
        import onnxruntime as ort

        available = ort.get_available_providers()

        if force_cpu:
            logger.warning("Inference falling back to CPU mode")
            return ['CPUExecutionProvider'], -1, "CPU"

        if self.device == "cpu":
            logger.info("Using CPU")
            return ['CPUExecutionProvider'], -1, "CPU"

        # "auto" or "cuda" — try GPU providers in preference order.
        if 'CUDAExecutionProvider' in available:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider'], 0, "GPU(CUDA)"
        elif 'DmlExecutionProvider' in available:
            return ['DmlExecutionProvider', 'CPUExecutionProvider'], 0, "GPU(DirectML)"
        else:
            if self.device == "cuda":
                logger.warning("device='cuda' requested but no GPU provider found, using CPU")
            else:
                logger.info("No usable GPU found, using CPU")
            return ['CPUExecutionProvider'], -1, "CPU"

    def _create_app(self, det_size):
        """Create and prepare a FaceAnalysis instance."""
        from insightface.app import FaceAnalysis

        providers, ctx_id, gpu_name = self._resolve_providers()
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        return app, gpu_name

    def _load_model(self, fallback_to_cpu=False):
        """Load the model, auto-detecting CUDA > DirectML > CPU.

        After a successful load, self.device reflects the *actual* device
        in use (one of "cpu", "cuda"), not the user's original request.
        """
        # Release old model resources before creating a new one to prevent
        # ONNX Runtime from leaking VRAM / memory on reload.
        if self.app is not None:
            try:
                del self.app
            except Exception:
                pass
            self.app = None

        try:
            self.app, device_label = self._create_app(self.det_size)
            self._gpu_name = device_label
            # Sync self.device to the actual provider that was resolved.
            # "auto" becomes "cpu" or "cuda" depending on hardware.
            if device_label.startswith("GPU"):
                self.device = "cuda"
            else:
                self.device = "cpu"
            logger.info("Model loaded (device=%s, det_size=%d)", device_label, self.det_size)
        except ImportError as e:
            # Distinguish insightface vs onnxruntime import failures so the
            # error message tells the user which package is actually missing.
            from face_hub.exceptions import DependencyError
            missing_module = getattr(e, "name", "")
            if "insightface" in str(e) or missing_module == "insightface":
                raise DependencyError(
                    "insightface is not installed",
                    model_name="insightface",
                ) from e
            elif "onnxruntime" in str(e) or "onnx" in missing_module:
                raise DependencyError(
                    "onnxruntime is not installed (required by insightface)",
                    model_name="onnxruntime",
                ) from e
            else:
                raise DependencyError(
                    f"Missing dependency during model load: {e}",
                ) from e
        except Exception as e:
            if not fallback_to_cpu and self.device != "cpu":
                # GPU load failed — "auto" silently falls back, "cuda" warns.
                logger.warning("GPU load failed: %s", e)
                logger.info("Falling back to CPU…")
                self.device = "cpu"
                self._load_model(fallback_to_cpu=True)
            else:
                raise ModelLoadError(
                    f"Failed to load FaceDetector model: {e}",
                    model_name="RetinaFace",
                ) from e

    def _is_directml_reshape_bug(self, exc: Exception) -> bool:
        """Detect DirectML 1.24.x Reshape bug heuristically.

        The bug manifests as either a UnicodeDecodeError (corrupted C++ error
        string leaking into Python) or a reshape/shape-mismatch error from
        ONNX Runtime when using certain input sizes on DirectML.
        """
        err_name = type(exc).__name__
        if err_name == "UnicodeDecodeError":
            return True
        msg = str(exc).lower()
        # DirectML reshape failures surface as shape/reshape errors.
        if "reshape" in msg or "shape" in msg:
            # Only attribute to DirectML if we are actually using it.
            if self._gpu_name.startswith("GPU(DirectML"):
                return True
        return False

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
            if self.device != "cpu" and self._is_directml_reshape_bug(e):
                try:
                    logger.warning("DirectML incompatible with det_size=%d, retrying with 640", self.det_size)
                    # Release old app before creating a new one.
                    old_app = self.app
                    new_app, _ = self._create_app(640)
                    dummy2 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    new_app.get(dummy2)
                    self.app = new_app
                    self.det_size = 640
                    del old_app
                    logger.info("Auto-adjusted to det_size=640 (DirectML compat)")
                    return
                except Exception as inner:
                    logger.error("640 retry also failed: %s", inner)
                logger.warning("DirectML inference failed, falling back to CPU…")
                self.device = "cpu"
                self._load_model(fallback_to_cpu=True)
            elif self.device != "cpu":
                logger.warning("Inference anomaly, attempting CPU switch…")
                self.device = "cpu"
                self._load_model(fallback_to_cpu=True)

    # ── Face quality assessment ───────────────────────────────────

    def _face_quality(self, face_roi):
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
        is_good = blur_score >= self.blur_threshold
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
            # A successful inference resets the consecutive error counter so
            # that transient glitches don't accumulate toward the fallback
            # threshold.
            self._inference_error_count = 0
        return faces

    def _handle_inference_error(self, e, frame):
        """Handle an inference error, falling back to CPU if needed.

        Thread-safe: acquires _lock so that at most one thread performs the
        fallback while others wait and then retry on the new (CPU) model.
        """
        err_type = type(e).__name__

        with self._lock:
            self._inference_error_count += 1
            err_count = self._inference_error_count

            # Another thread already completed the CPU fallback — just retry.
            if self.device == "cpu":
                logger.info("Device already switched to CPU by another thread, retrying…")
                return self.app.get(frame)

            if self.device != "cpu":
                # Only trigger fallback after enough consecutive errors to
                # avoid a single transient GPU glitch causing a switch.
                if err_count < _MAX_INFERENCE_ERRORS:
                    logger.warning(
                        "GPU inference error #%d (%s: %s), will fallback after %d",
                        err_count, err_type, e, _MAX_INFERENCE_ERRORS,
                    )
                    raise InferenceError(
                        f"GPU inference error ({err_type}): {e}",
                        device=self.device,
                    ) from e

                logger.warning("GPU inference error (%s: %s), switching to CPU…", err_type, e)
                try:
                    self.device = "cpu"
                    self._load_model(fallback_to_cpu=True)
                    self._inference_error_count = 0  # Reset after successful fallback
                    return self.app.get(frame)
                except Exception as e2:
                    raise InferenceError(
                        f"Inference failed on both GPU and CPU: {e2}",
                        device=self.device,
                    ) from e2

        raise InferenceError(
            f"Inference error on {self.device}: {e}", device=self.device
        ) from e

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
            # Convert to Python ints so BBox stays JSON-serialisable
            # (face.bbox holds numpy scalars)
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
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
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
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

    def extract_face_roi(self, frame, face_rect: Tuple):
        """Crop a face region with a small expansion (legacy-compatible interface).

        Accepts either a 4-element (x1, y1, x2, y2) or 5-element
        (x1, y1, x2, y2, confidence) tuple/list, matching both BBox.to_tuple()
        and the old (bbox + score) convention.
        """
        if len(face_rect) == 5:
            x1, y1, x2, y2, _ = face_rect
        elif len(face_rect) == 4:
            x1, y1, x2, y2 = face_rect
        else:
            raise ValueError(
                f"face_rect must have 4 or 5 elements, got {len(face_rect)}"
            )
        return self._face_roi(frame, (x1, y1, x2, y2), expand=0.20)

    def reload_model(self, det_size=None):
        """Reload the model with new parameters (used for runtime resolution changes)."""
        if det_size is not None:
            self.det_size = det_size
        self._load_model()
        self._warmup()
