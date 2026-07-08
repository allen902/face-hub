"""Test exception hierarchy."""

from face_vision.exceptions import (
    FaceVisionError, ModelLoadError, InferenceError,
    CameraError, DatabaseError, RecognitionError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        for cls in [ModelLoadError, InferenceError, CameraError,
                     DatabaseError, RecognitionError]:
            assert issubclass(cls, FaceVisionError)

    def test_catch_by_base(self):
        try:
            raise ModelLoadError("test")
        except FaceVisionError:
            pass  # should catch
        else:
            assert False

    def test_cause_chain(self):
        try:
            try:
                raise RuntimeError("root cause")
            except RuntimeError as e:
                raise ModelLoadError("wrapper") from e
        except ModelLoadError as ex:
            assert isinstance(ex.__cause__, RuntimeError)
            assert "root cause" in str(ex.__cause__)
