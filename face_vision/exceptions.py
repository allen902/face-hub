"""
FaceVision exception hierarchy.
All library exceptions inherit from FaceVisionError.
"""


class FaceVisionError(Exception):
    """Base exception for all FaceVision errors."""
    pass


class ModelLoadError(FaceVisionError):
    """Model loading failed (insightface not installed / no ONNX provider / corrupt model)."""
    pass


class InferenceError(FaceVisionError):
    """ML inference runtime error (GPU crash + CPU fallback also failed)."""
    pass


class CameraError(FaceVisionError):
    """Camera error (not connected / in use / unsupported resolution)."""
    pass


class DatabaseError(FaceVisionError):
    """Database read/write error (JSON parse / pickle corrupt / disk full / permission)."""
    pass


class RecognitionError(FaceVisionError):
    """Recognition matching error (encoding dimension mismatch / empty cache)."""
    pass
