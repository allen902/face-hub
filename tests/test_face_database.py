"""Test FaceDatabase CRUD and persistence."""

import numpy as np
import pytest
from pathlib import Path
from face_hub import FaceDatabase


def _img(db_path, name):
    """Helper: build an image path inside the DB's safe directory."""
    return str(Path(db_path).parent / name)


class TestFaceDatabase:
    def test_init_empty(self, temp_db_paths):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        assert len(db.get_names()) == 0
        assert db.version > 0

    def test_add_person(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        ok, msg = db.add_person("Alice", _img(db_path, "alice.jpg"), sample_encoding)
        assert ok is True
        assert "Alice" in db.get_names()
        assert len(db.get_encodings_and_names()[0]) == 1

    def test_add_duplicate(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        db.add_person("Alice", _img(db_path, "a.jpg"), sample_encoding)
        ok, msg = db.add_person("Alice", _img(db_path, "b.jpg"), sample_encoding)
        assert ok is False

    def test_add_person_validates_name(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        with pytest.raises(ValueError, match="non-empty string"):
            db.add_person("", _img(db_path, "a.jpg"), sample_encoding)
        with pytest.raises(ValueError, match="non-empty string"):
            db.add_person(None, _img(db_path, "a.jpg"), sample_encoding)

    def test_add_person_validates_encoding_shape(self, temp_db_paths):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        with pytest.raises(ValueError, match="shape"):
            db.add_person("Alice", _img(db_path, "a.jpg"), np.zeros(256))
        with pytest.raises(ValueError, match="shape"):
            db.add_person("Alice", _img(db_path, "a.jpg"), "not_an_array")

    def test_add_person_validates_path_traversal(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        with pytest.raises(ValueError, match="outside"):
            db.add_person("Eve", "/tmp/evil.jpg", sample_encoding)
        with pytest.raises(ValueError, match="outside"):
            db.add_person("Eve", "../../../etc/passwd", sample_encoding)

    def test_remove_person(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        db.add_person("Alice", _img(db_path, "a.jpg"), sample_encoding)
        ok, msg = db.remove_person("Alice")
        assert ok is True
        assert len(db.get_names()) == 0

    def test_remove_nonexistent(self, temp_db_paths):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        ok, msg = db.remove_person("Nobody")
        assert ok is False

    def test_remove_persons_batch(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        db.add_person("Alice", _img(db_path, "a.jpg"), sample_encoding)
        db.add_person("Bob", _img(db_path, "b.jpg"), sample_encoding)
        db.add_person("Charlie", _img(db_path, "c.jpg"), sample_encoding)
        removed, not_found = db.remove_persons(["Alice", "Bob", "Nobody"])
        assert removed == ["Alice", "Bob"]
        assert not_found == ["Nobody"]
        assert db.get_names() == ["Charlie"]

    def test_remove_persons_deduplicates(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        db.add_person("Alice", _img(db_path, "a.jpg"), sample_encoding)
        db.add_person("Bob", _img(db_path, "b.jpg"), sample_encoding)
        # Duplicate name in the input should not cause index corruption
        removed, not_found = db.remove_persons(["Alice", "Alice"])
        assert removed == ["Alice", "Alice"]
        # Bob must NOT be accidentally deleted
        assert db.get_names() == ["Bob"]

    def test_version_bumps_on_change(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        v1 = db.version
        db.add_person("Alice", _img(db_path, "a.jpg"), sample_encoding)
        v2 = db.version
        assert v2 > v1
        db.remove_person("Alice")
        v3 = db.version
        assert v3 > v2

    def test_persistence(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db1 = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        db1.add_person("Alice", _img(db_path, "a.jpg"), sample_encoding)

        db2 = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        assert "Alice" in db2.get_names()
        encs, names = db2.get_encodings_and_names()
        assert len(encs) == 1
        assert names == ["Alice"]

    def test_clear(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        db.add_person("Alice", _img(db_path, "a.jpg"), sample_encoding)
        db.clear()
        assert len(db.get_names()) == 0

    def test_encoding_saved_as_npy(self, temp_db_paths, sample_encoding):
        """Verify encodings are stored as .npy (not pickle)."""
        import os
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        db.add_person("Alice", _img(db_path, "a.jpg"), sample_encoding)
        assert os.path.exists(enc_path)
        assert enc_path.endswith(".npy")
        # Verify it can be loaded with numpy (no pickle needed)
        data = np.load(enc_path, allow_pickle=False)
        assert data.shape == (1, 512)
