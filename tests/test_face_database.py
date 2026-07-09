"""Test FaceDatabase CRUD and persistence."""

import numpy as np
from face_hub import FaceDatabase


class TestFaceDatabase:
    def test_init_empty(self, temp_db_paths):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        assert len(db.get_names()) == 0
        assert db.version > 0

    def test_add_person(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        ok, msg = db.add_person("Alice", "/tmp/alice.jpg", sample_encoding)
        assert ok is True
        assert "Alice" in db.get_names()
        assert len(db.get_encodings_and_names()[0]) == 1

    def test_add_duplicate(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        db.add_person("Alice", "/tmp/a.jpg", sample_encoding)
        ok, msg = db.add_person("Alice", "/tmp/b.jpg", sample_encoding)
        assert ok is False

    def test_remove_person(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        db.add_person("Alice", "/tmp/a.jpg", sample_encoding)
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
        db.add_person("Alice", "/tmp/a.jpg", sample_encoding)
        db.add_person("Bob", "/tmp/b.jpg", sample_encoding)
        db.add_person("Charlie", "/tmp/c.jpg", sample_encoding)
        removed, not_found = db.remove_persons(["Alice", "Bob", "Nobody"])
        assert removed == ["Alice", "Bob"]
        assert not_found == ["Nobody"]
        assert db.get_names() == ["Charlie"]

    def test_version_bumps_on_change(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        v1 = db.version
        db.add_person("Alice", "/tmp/a.jpg", sample_encoding)
        v2 = db.version
        assert v2 > v1
        db.remove_person("Alice")
        v3 = db.version
        assert v3 > v2

    def test_persistence(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db1 = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        db1.add_person("Alice", "/tmp/a.jpg", sample_encoding)

        db2 = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        assert "Alice" in db2.get_names()
        encs, names = db2.get_encodings_and_names()
        assert len(encs) == 1
        assert names == ["Alice"]

    def test_clear(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        db.add_person("Alice", "/tmp/a.jpg", sample_encoding)
        db.clear()
        assert len(db.get_names()) == 0
