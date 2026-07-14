"""
FaceHub exception hierarchy.
All library exceptions inherit from FaceHubError.

Design principles:
  - Multiple inheritance from built-in exceptions where it helps callers
    use existing ``except OSError`` / ``except ValueError`` patterns.
  - Structured attributes (model_name, camera_id, db_path …) so callers
    can inspect failure context programmatically.
  - Subclasses for distinct failure modes (DependencyError, SerializationError).
  - Full backward compatibility: ``except FaceHubError`` still catches everything.
"""


class FaceHubError(Exception):
    """Base exception for all FaceHub errors."""
    pass


# ── Model loading ─────────────────────────────────────────────────

class ModelLoadError(FaceHubError, RuntimeError):
    """Model loading failed.

    Attributes:
        model_name: Which model failed (e.g. ``"buffalo_l"``, ``"RetinaFace"``).
        model_path: File path that was attempted, if any.
    """

    def __init__(self, message, *, model_name=None, model_path=None):
        super().__init__(message)
        self.model_name = model_name
        self.model_path = model_path


class DependencyError(ModelLoadError):
    """A required Python package is not installed.

    Raised when insightface, onnxruntime, or another runtime dependency
    is missing.  Subclass of ModelLoadError so existing
    ``except ModelLoadError`` catches it.
    """
    pass


# ── Inference ─────────────────────────────────────────────────────

class InferenceError(FaceHubError, RuntimeError):
    """ML inference runtime error (GPU crash + CPU fallback also failed).

    Attributes:
        device: Device that failed (``"cuda"``, ``"cpu"``, …).
    """

    def __init__(self, message, *, device=None):
        super().__init__(message)
        self.device = device


# ── Camera ────────────────────────────────────────────────────────

class CameraError(FaceHubError):
    """Camera error (not connected / in use / unsupported resolution).

    Attributes:
        camera_id: Camera index or path that failed.
    """

    def __init__(self, message, *, camera_id=None):
        super().__init__(message)
        self.camera_id = camera_id


# ── Database ──────────────────────────────────────────────────────

class DatabaseError(FaceHubError, OSError):
    """Database I/O error (disk full / permission denied / file locked).

    Attributes:
        db_path: Path to the database file that failed.
    """

    def __init__(self, message, *, db_path=None):
        super().__init__(message)
        self.db_path = db_path


class SerializationError(DatabaseError, ValueError):
    """Data format error (JSON parse failure / corrupt .npy / pickle issue).

    Subclass of both DatabaseError and ValueError so callers can catch at
    either level: ``except DatabaseError`` or ``except ValueError``.
    """
    pass


# ── Recognition ───────────────────────────────────────────────────

class RecognitionError(FaceHubError):
    """Recognition matching error (encoding dimension mismatch / empty cache)."""
    pass
