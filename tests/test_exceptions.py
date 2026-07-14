"""Test exception hierarchy."""

import pytest
from face_hub.exceptions import (
    FaceHubError, ModelLoadError, DependencyError,
    InferenceError, CameraError, DatabaseError,
    SerializationError, RecognitionError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        for cls in [ModelLoadError, InferenceError, CameraError,
                     DatabaseError, RecognitionError,
                     DependencyError, SerializationError]:
            assert issubclass(cls, FaceHubError), f"{cls.__name__} not a FaceHubError"

    def test_catch_by_base(self):
        for cls in [ModelLoadError, DependencyError, InferenceError,
                     CameraError, DatabaseError, SerializationError,
                     RecognitionError]:
            with pytest.raises(FaceHubError):
                raise cls("test")

    def test_cause_chain(self):
        try:
            try:
                raise RuntimeError("root cause")
            except RuntimeError as e:
                raise ModelLoadError("wrapper") from e
        except ModelLoadError as ex:
            assert isinstance(ex.__cause__, RuntimeError)
            assert "root cause" in str(ex.__cause__)

    def test_multiple_inheritance(self):
        """Exceptions that inherit from built-ins can be caught either way."""
        assert issubclass(ModelLoadError, RuntimeError)
        assert issubclass(InferenceError, RuntimeError)
        assert issubclass(DatabaseError, OSError)
        assert issubclass(SerializationError, DatabaseError)
        assert issubclass(SerializationError, ValueError)

        # DependencyError is a ModelLoadError
        assert issubclass(DependencyError, ModelLoadError)
        assert issubclass(DependencyError, RuntimeError)

        # SerializationError can be caught as DatabaseError, ValueError, or OSError
        with pytest.raises(DatabaseError):
            raise SerializationError("bad json")
        with pytest.raises(ValueError):
            raise SerializationError("bad json")
        with pytest.raises(OSError):
            raise SerializationError("bad json")

    def test_structured_context(self):
        """Exceptions carry typed context attributes."""
        e = ModelLoadError("fail", model_name="buffalo_l", model_path="/x.onnx")
        assert e.model_name == "buffalo_l"
        assert e.model_path == "/x.onnx"

        e = InferenceError("fail", device="cuda")
        assert e.device == "cuda"

        e = CameraError("fail", camera_id=0)
        assert e.camera_id == 0

        e = DatabaseError("fail", db_path="/tmp/db.json")
        assert e.db_path == "/tmp/db.json"

        e = SerializationError("fail", db_path="/tmp/db.json")
        assert e.db_path == "/tmp/db.json"

    def test_context_defaults_to_none(self):
        """Context attributes default to None when not provided."""
        assert ModelLoadError("x").model_name is None
        assert InferenceError("x").device is None
        assert CameraError("x").camera_id is None
        assert DatabaseError("x").db_path is None
