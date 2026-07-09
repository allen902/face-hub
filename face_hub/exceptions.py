"""
FaceHub exception hierarchy.
All library exceptions inherit from FaceHubError.
"""


class FaceHubError(Exception):
    """Base exception for all FaceHub errors."""
    pass


class ModelLoadError(FaceHubError):
    """Model loading failed (insightface not installed / no ONNX provider / corrupt model)."""
    pass


class InferenceError(FaceHubError):
    """ML inference runtime error (GPU crash + CPU fallback also failed)."""
    pass


class CameraError(FaceHubError):
    """Camera error (not connected / in use / unsupported resolution)."""
    pass


class DatabaseError(FaceHubError):
    """Database read/write error (JSON parse / pickle corrupt / disk full / permission)."""
    pass


class RecognitionError(FaceHubError):
    """Recognition matching error (encoding dimension mismatch / empty cache)."""
    pass
