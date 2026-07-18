"""
Tests for face_hub.engine.photo_classifier.

Uses a stub detector returning synthetic 512-dim embeddings — no model
download, no insightface, no camera required.
"""

import os

import numpy as np
import pytest

from face_hub.types import BBox, DetectionWithEmbedding, UNKNOWN_SENTINEL
from face_hub.engine.face_recognizer import FaceRecognizer
from face_hub.engine.photo_classifier import (
    PhotoClassifier,
    classify_photos,
    export_to_folders,
    _EmbeddingClusterer,
)


# ── Helpers ─────────────────────────────────────────────────────

def _make_embedding(seed_vector: np.ndarray, noise: float = 0.02, rng=None) -> np.ndarray:
    """A normalized embedding close to seed_vector (same "person")."""
    rng = rng or np.random.default_rng(0)
    emb = seed_vector + rng.normal(0, noise, size=seed_vector.shape).astype(np.float32)
    return (emb / np.linalg.norm(emb)).astype(np.float32)


def _person_seed(person_id: int) -> np.ndarray:
    """Deterministic, well-separated base embedding per fake person."""
    rng = np.random.default_rng(1000 + person_id)
    vec = rng.normal(0, 1, size=512).astype(np.float32)
    return vec / np.linalg.norm(vec)


class StubDetector:
    """DetectorProtocol stub: returns pre-scripted detections per call."""

    def __init__(self, scripted):
        # scripted: list of lists — detections returned per successive call
        self._scripted = list(scripted)
        self.calls = 0

    def detect_with_embeddings(self, frame):
        detections = self._scripted[self.calls % len(self._scripted)]
        self.calls += 1
        return detections


def _det(embedding, confidence=0.9, quality_pass=True, bbox=None):
    return DetectionWithEmbedding(
        bbox=bbox or BBox(x1=10, y1=10, x2=90, y2=90),
        confidence=confidence,
        embedding=embedding,
        quality_pass=quality_pass,
    )


_BLANK = np.zeros((100, 100, 3), dtype=np.uint8)


# ── Clusterer unit tests ────────────────────────────────────────

class TestEmbeddingClusterer:
    def test_same_person_one_cluster(self):
        seed = _person_seed(1)
        rng = np.random.default_rng(7)
        clusterer = _EmbeddingClusterer(threshold=0.45)
        idxs = [clusterer.assign(_make_embedding(seed, rng=rng))[0] for _ in range(5)]
        assert len(set(idxs)) == 1

    def test_different_people_separate_clusters(self):
        clusterer = _EmbeddingClusterer(threshold=0.45)
        idx_a, _ = clusterer.assign(_person_seed(1))
        idx_b, _ = clusterer.assign(_person_seed(2))
        assert idx_a != idx_b


# ── Discovery mode ──────────────────────────────────────────────

class TestDiscoveryMode:
    def test_groups_by_person(self):
        seed_a, seed_b = _person_seed(1), _person_seed(2)
        detector = StubDetector([
            [_det(_make_embedding(seed_a))],                 # photo 1 → A
            [_det(_make_embedding(seed_a))],                 # photo 2 → A
            [_det(_make_embedding(seed_b))],                 # photo 3 → B
            [],                                              # photo 4 → no face
        ])
        result = PhotoClassifier(detector).classify_photos(
            [_BLANK] * 4, photo_ids=["p1", "p2", "p3", "p4"]
        )

        assert result.total_photos == 4
        assert len(result.groups) == 2
        assert result.no_face_photos == ["p4"]
        assert sorted(result.groups["person_001"].photo_ids) == ["p1", "p2"]
        assert result.groups["person_002"].photo_ids == ["p3"]
        assert result.labels_of("p1") == ["person_001"]

    def test_multi_person_photo_in_multiple_groups(self):
        seed_a, seed_b = _person_seed(1), _person_seed(2)
        detector = StubDetector([
            [_det(_make_embedding(seed_a)), _det(_make_embedding(seed_b))],
        ])
        result = PhotoClassifier(detector).classify_photos([_BLANK], photo_ids=["group.jpg"])

        assert result.labels_of("group.jpg") == ["person_001", "person_002"]
        assert result.groups["person_001"].photo_ids == ["group.jpg"]
        assert result.groups["person_002"].photo_ids == ["group.jpg"]
        assert result.groups["person_001"].face_count == 1

    def test_unreadable_photo(self, tmp_path):
        bad = tmp_path / "not_an_image.jpg"
        bad.write_bytes(b"this is not a jpeg")
        detector = StubDetector([[]])
        result = PhotoClassifier(detector).classify_photos([bad])

        assert result.unreadable_photos == [str(bad)]
        assert result.no_face_photos == []
        assert result.groups == {}

    def test_low_quality_faces_skipped(self):
        seed = _person_seed(1)
        detector = StubDetector([[_det(_make_embedding(seed), quality_pass=False)]])
        result = PhotoClassifier(detector).classify_photos([_BLANK], photo_ids=["p1"])

        assert result.no_face_photos == ["p1"]

        detector2 = StubDetector([[_det(_make_embedding(seed), quality_pass=False)]])
        result2 = PhotoClassifier(detector2, skip_low_quality=False).classify_photos(
            [_BLANK], photo_ids=["p1"]
        )
        assert result2.groups["person_001"].photo_ids == ["p1"]

    def test_progress_callback(self):
        seed = _person_seed(1)
        detector = StubDetector([[_det(_make_embedding(seed))]] * 3)
        calls = []
        PhotoClassifier(detector).classify_photos(
            [_BLANK] * 3,
            progress_callback=lambda done, total, pid: calls.append((done, total)),
        )
        assert calls == [(1, 3), (2, 3), (3, 3)]


# ── Gallery mode ────────────────────────────────────────────────

class TestGalleryMode:
    def _recognizer_with_alice(self, alice_seed):
        recognizer = FaceRecognizer(tolerance=0.45)
        recognizer.update_cache([alice_seed], ["alice"], db_version=1)
        return recognizer

    def test_known_face_filed_under_name(self):
        alice_seed = _person_seed(1)
        recognizer = self._recognizer_with_alice(alice_seed)
        detector = StubDetector([
            [_det(_make_embedding(alice_seed, noise=0.01))],  # alice
            [_det(_make_embedding(_person_seed(2)))],         # stranger
        ])
        result = PhotoClassifier(detector, recognizer=recognizer).classify_photos(
            [_BLANK] * 2, photo_ids=["a.jpg", "b.jpg"]
        )

        assert result.groups["alice"].photo_ids == ["a.jpg"]
        assert result.groups["person_001"].photo_ids == ["b.jpg"]
        assert result.faces[0].similarity >= 0.45

    def test_empty_recognizer_falls_back_to_clustering(self):
        recognizer = FaceRecognizer()  # no gallery registered
        name, _ = recognizer.recognize(_person_seed(1))
        assert name == UNKNOWN_SENTINEL

        detector = StubDetector([[_det(_person_seed(1))]])
        result = PhotoClassifier(detector, recognizer=recognizer).classify_photos(
            [_BLANK], photo_ids=["p1"]
        )
        assert result.groups["person_001"].photo_ids == ["p1"]


# ── Input validation & helper ───────────────────────────────────

class TestValidation:
    def test_photo_ids_length_mismatch(self):
        detector = StubDetector([[]])
        with pytest.raises(ValueError, match="same length"):
            PhotoClassifier(detector).classify_photos([_BLANK], photo_ids=["a", "b"])

    def test_photo_ids_duplicates(self):
        detector = StubDetector([[]])
        with pytest.raises(ValueError, match="duplicates"):
            PhotoClassifier(detector).classify_photos(
                [_BLANK] * 2, photo_ids=["a", "a"]
            )

    def test_invalid_cluster_threshold(self):
        with pytest.raises(ValueError, match="cluster_threshold"):
            PhotoClassifier(StubDetector([[]]), cluster_threshold=1.5)

    def test_default_photo_ids_for_arrays(self):
        seed = _person_seed(1)
        detector = StubDetector([[_det(_make_embedding(seed))]])
        result = PhotoClassifier(detector).classify_photos([_BLANK])
        assert result.groups["person_001"].photo_ids == ["image_0001"]


class TestOneShotHelper:
    def test_classify_photos_with_provided_detector(self):
        seed = _person_seed(1)
        detector = StubDetector([[_det(_make_embedding(seed))]])
        result = classify_photos([_BLANK], detector=detector)
        assert result.groups["person_001"].photo_ids == ["image_0001"]

    def test_summary_bookkeeping(self):
        seed = _person_seed(1)
        detector = StubDetector([[_det(_make_embedding(seed))], []])
        result = classify_photos([_BLANK, _BLANK], detector=detector)
        summary = result.summary()
        assert summary["person_001"] == 1
        assert summary["__no_face__"] == 1


# ── Folder export ───────────────────────────────────────────────

def _write_photo(path, value=128):
    """Write a real (tiny) JPEG so cv2.imread succeeds."""
    import cv2
    img = np.full((32, 32, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), img)
    return str(path)


def _classified_fixture(tmp_path):
    """
    4 photos: p1=A, p2=A+B, p3=B, p4=no face.
    Returns (result, photo_paths).
    """
    seed_a, seed_b = _person_seed(1), _person_seed(2)
    photos = [
        _write_photo(tmp_path / "p1.jpg", 10),
        _write_photo(tmp_path / "p2.jpg", 20),
        _write_photo(tmp_path / "p3.jpg", 30),
        _write_photo(tmp_path / "p4.jpg", 40),
    ]
    detector = StubDetector([
        [_det(_make_embedding(seed_a))],
        [_det(_make_embedding(seed_a)), _det(_make_embedding(seed_b))],
        [_det(_make_embedding(seed_b))],
        [],
    ])
    result = PhotoClassifier(detector).classify_photos(photos)
    return result, photos


class TestExportToFolders:
    def test_copy_mode_creates_per_person_folders(self, tmp_path):
        result, photos = _classified_fixture(tmp_path)
        out = tmp_path / "sorted"
        export = export_to_folders(result, out)

        # person_001 → p1, p2 / person_002 → p2, p3 / _no_face → p4
        assert sorted(os.listdir(out / "person_001")) == ["p1.jpg", "p2.jpg"]
        assert sorted(os.listdir(out / "person_002")) == ["p2.jpg", "p3.jpg"]
        assert os.listdir(out / "_no_face") == ["p4.jpg"]
        assert export.total_files == 5          # multi-person photo counted twice
        assert export.skipped == []
        assert export.errors == {}
        # copy keeps originals
        for p in photos:
            assert os.path.isfile(p)

    def test_move_mode_removes_originals(self, tmp_path):
        result, photos = _classified_fixture(tmp_path)
        out = tmp_path / "sorted"
        export_to_folders(result, out, mode="move")

        assert not os.path.exists(photos[0])                    # p1 moved
        assert os.path.isfile(out / "person_001" / "p1.jpg")
        assert os.path.isfile(out / "person_002" / "p2.jpg")    # p2's second export

    def test_include_no_face_false(self, tmp_path):
        result, _ = _classified_fixture(tmp_path)
        out = tmp_path / "sorted"
        export_to_folders(result, out, include_no_face=False)
        assert not (out / "_no_face").exists()

    def test_non_file_photo_ids_are_skipped(self):
        seed = _person_seed(1)
        detector = StubDetector([[_det(_make_embedding(seed))]])
        result = PhotoClassifier(detector).classify_photos([_BLANK])  # array input
        export = export_to_folders(result, "unused_out")
        assert export.skipped == ["image_0001"]
        assert export.total_files == 0

    def test_conflict_rename(self, tmp_path):
        result, _ = _classified_fixture(tmp_path)
        out = tmp_path / "sorted"
        # Pre-create a conflicting file
        (out / "person_001").mkdir(parents=True)
        (out / "person_001" / "p1.jpg").write_bytes(b"old")

        export_to_folders(result, out, on_conflict="rename")
        files = sorted(os.listdir(out / "person_001"))
        assert files == ["p1.jpg", "p1_1.jpg", "p2.jpg"]
        assert (out / "person_001" / "p1.jpg").read_bytes() == b"old"  # untouched

    def test_conflict_skip_and_overwrite(self, tmp_path):
        result, _ = _classified_fixture(tmp_path)

        out1 = tmp_path / "s1"
        (out1 / "person_001").mkdir(parents=True)
        (out1 / "person_001" / "p1.jpg").write_bytes(b"old")
        export_to_folders(result, out1, on_conflict="skip")
        assert (out1 / "person_001" / "p1.jpg").read_bytes() == b"old"
        assert not (out1 / "person_001" / "p1_1.jpg").exists()

        out2 = tmp_path / "s2"
        (out2 / "person_001").mkdir(parents=True)
        (out2 / "person_001" / "p1.jpg").write_bytes(b"old")
        export_to_folders(result, out2, on_conflict="overwrite")
        assert (out2 / "person_001" / "p1.jpg").read_bytes() != b"old"

    def test_sanitizes_folder_names(self, tmp_path):
        seed = _person_seed(1)
        detector = StubDetector([[_det(_make_embedding(seed))]])
        result = PhotoClassifier(detector).classify_photos(
            [_write_photo(tmp_path / "x.jpg")], photo_ids=[str(tmp_path / "x.jpg")]
        )
        # Forge an unsafe label
        group = result.groups.pop("person_001")
        group.label = 'weird<>:"/\\|?*name'
        result.groups['weird<>:"/\\|?*name'] = group

        out = tmp_path / "sorted"
        export_to_folders(result, out)
        folders = os.listdir(out)
        assert folders == ["weird_________name"]

    def test_invalid_arguments(self, tmp_path):
        result, _ = _classified_fixture(tmp_path)
        with pytest.raises(ValueError, match="mode"):
            export_to_folders(result, tmp_path / "o", mode="delete")
        with pytest.raises(ValueError, match="on_conflict"):
            export_to_folders(result, tmp_path / "o", on_conflict="explode")
