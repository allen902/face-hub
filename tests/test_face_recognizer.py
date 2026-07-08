"""Test FaceRecognizer cosine similarity matching and cache."""

import numpy as np
from face_vision import FaceRecognizer, UNKNOWN_SENTINEL


def make_encoding(seed=42):
    rng = np.random.RandomState(seed)
    vec = rng.randn(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestFaceRecognizer:
    def test_init_default_tolerance(self):
        r = FaceRecognizer()
        assert r.tolerance == 0.45

    def test_recognize_empty_cache(self, sample_encoding):
        r = FaceRecognizer()
        name, conf = r.recognize(sample_encoding)
        assert name == UNKNOWN_SENTINEL
        assert conf == 0.0

    def test_recognize_self_match(self):
        r = FaceRecognizer(tolerance=0.40)
        enc = make_encoding(42)
        r.update_cache([enc], ["Alice"], db_version=1)
        name, conf = r.recognize(enc)  # same vector
        assert name == "Alice"
        assert conf > 0.99  # cosine of identical vectors ~1.0

    def test_recognize_below_tolerance(self):
        r = FaceRecognizer(tolerance=0.90)  # very strict
        enc1 = make_encoding(42)
        enc2 = make_encoding(99)  # different seed → different direction
        r.update_cache([enc1], ["Alice"], db_version=1)
        name, conf = r.recognize(enc2)
        assert name == UNKNOWN_SENTINEL

    def test_cache_version_skip(self):
        r = FaceRecognizer()
        enc = make_encoding(1)
        assert r.update_cache([enc], ["Alice"], db_version=1) is True
        assert r.update_cache([enc], ["Alice"], db_version=1) is False  # same version, no-op

    def test_cache_rebuild_on_version_change(self):
        r = FaceRecognizer()
        enc1 = make_encoding(1)
        enc2 = make_encoding(2)
        r.update_cache([enc1], ["Alice"], db_version=1)
        assert r.cached_names == ["Alice"]
        r.update_cache([enc2], ["Bob"], db_version=2)
        assert r.cached_names == ["Bob"]

    def test_recognize_with_explicit_encodings(self):
        r = FaceRecognizer(tolerance=0.40)
        enc = make_encoding(7)
        name, conf = r.recognize(enc, known_encodings=[enc], known_names=["Test"])
        assert name == "Test"

    def test_recognize_none_encoding(self):
        r = FaceRecognizer()
        name, conf = r.recognize(None)
        assert name == UNKNOWN_SENTINEL
