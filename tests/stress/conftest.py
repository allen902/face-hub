"""
Stress test infrastructure: Fake components, memory helper, baseline loader.

All fake components implement the same interfaces as real components
so they can be dropped into FaceHubPipeline without any code change.
"""
from __future__ import annotations

import gc
import json
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import psutil
import pytest

from face_hub.detector_protocol import DetectorProtocol
from face_hub.types import (
    BBox,
    DetectionResult,
    DetectionWithEmbedding,
    UNKNOWN_SENTINEL,
)

# ═══════════════════════════════════════════════════════════════════
# Baseline helpers
# ═══════════════════════════════════════════════════════════════════

BASELINE_DIR = Path(__file__).parent
BASELINE_PATH = BASELINE_DIR / "baseline.json"


def load_baseline() -> dict:
    """Load performance baseline, return empty dict if missing."""
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_baseline(data: dict) -> None:
    """Persist performance baseline to disk."""
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════
# Memory helper fixture
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MemoryDelta:
    """Memory measurement result."""
    rss_delta_mb: float
    vms_delta_mb: float
    top_tracemalloc: list  # top 10 allocation sites


@pytest.fixture
def memory_helper():
    """Fixture: measure RSS + tracemalloc delta across a test body.

    Usage:
        def test_foobar(memory_helper):
            # ... code under test ...
            delta = memory_helper.snapshot()
            assert delta.rss_delta_mb < 50  # 50 MB threshold
    """
    gc.collect()
    tracemalloc.start()
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    vms_before = proc.memory_info().vms
    snap_before = tracemalloc.take_snapshot()

    result = {"done": False, "delta": None}

    def snapshot() -> MemoryDelta:
        gc.collect()
        rss_after = proc.memory_info().rss
        vms_after = proc.memory_info().vms
        snap_after = tracemalloc.take_snapshot()
        top = snap_after.compare_to(snap_before, "lineno")
        delta = MemoryDelta(
            rss_delta_mb=(rss_after - rss_before) / (1024 * 1024),
            vms_delta_mb=(vms_after - vms_before) / (1024 * 1024),
            top_tracemalloc=[str(t) for t in top[:10]],
        )
        result["delta"] = delta
        result["done"] = True
        return delta

    yield type("MemoryHelper", (), {
        "snapshot": snapshot,
        "rss_mb": lambda: (proc.memory_info().rss - rss_before) / (1024 * 1024),
    })

    if not result["done"]:
        try:
            snapshot()  # ensure we always capture
        except RuntimeError:
            pass  # tracemalloc may have been stopped already
    try:
        tracemalloc.stop()
    except RuntimeError:
        pass


# ═══════════════════════════════════════════════════════════════════
# FakeCamera — controllable frame source
# ═══════════════════════════════════════════════════════════════════

class FakeCamera:
    """Mock camera that returns synthetic frames on demand.

    Supports:
      - Fixed or random frame generation
      - Controllable frame rate
      - Error injection
      - Multi-consumer testing via Condition (matching real CameraThread)
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 360,
        fps: int = 30,
        frame_factory: Optional[Callable[[], np.ndarray]] = None,
    ):
        self.width = width
        self.height = height
        self.target_fps = fps
        self.running = False
        self._frame_factory = frame_factory or self._default_frame
        self._frame_buffer = None
        self._frame_cond = threading.Condition(threading.Lock())
        self._error_on_next: Optional[Exception] = None
        self._frame_count = 0

    @staticmethod
    def _default_frame() -> np.ndarray:
        return np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False
        with self._frame_cond:
            self._frame_cond.notify_all()

    def get_frame(self, timeout: float = 0.05, copy: bool = True):
        """Thread-safe frame retrieval, matching CameraThread.get_frame() contract."""
        if self._error_on_next is not None:
            err = self._error_on_next
            self._error_on_next = None
            raise err

        with self._frame_cond:
            frame = self._frame_factory()
            self._frame_count += 1
            if copy:
                return frame.copy()
            return frame

    def inject_error(self, exc: Exception) -> None:
        """Next get_frame() call will raise this exception."""
        self._error_on_next = exc

    @property
    def frame_count(self) -> int:
        return self._frame_count


# ═══════════════════════════════════════════════════════════════════
# FakeDetector — controllable detection results
# ═══════════════════════════════════════════════════════════════════

class FakeDetector:
    """Mock face detector with programmatic control over outputs.

    Attributes:
        det_size: Current detection resolution (mutable for hot-swap tests).
        device: Simulated device string.
        _inference_error_count: Exposed for GPU-fallback test assertions.
    """

    def __init__(
        self,
        detections_factory: Optional[Callable[[np.ndarray], List[DetectionWithEmbedding]]] = None,
        det_size: int = 640,
        device: str = "cpu",
    ):
        self._factory = detections_factory or self._default_factory
        self.det_size = det_size
        self.device = device
        self._lock = threading.Lock()
        self._error_count = 0
        self._error_type = RuntimeError
        self._inference_error_count = 0
        self._model_reloads = 0
        self._call_count = 0

    # ── Default factory: 2 faces with embeddings ────────────────

    @staticmethod
    def _default_factory(frame) -> List[DetectionWithEmbedding]:
        """Return 2 fake detections with 512-dim normalized embeddings."""
        rng = np.random.RandomState(42)
        emb1 = rng.randn(512).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        emb2 = rng.randn(512).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)
        return [
            DetectionWithEmbedding(
                bbox=BBox(x1=100, y1=80, x2=200, y2=220),
                confidence=0.95,
                embedding=emb1,
                quality_pass=True,
            ),
            DetectionWithEmbedding(
                bbox=BBox(x1=400, y1=60, x2=520, y2=210),
                confidence=0.88,
                embedding=emb2,
                quality_pass=True,
            ),
        ]

    # ── Error injection ─────────────────────────────────────────

    def set_error_count(self, n: int, error_type: type = RuntimeError) -> None:
        """Next `n` calls will raise `error_type`."""
        self._error_count = n
        self._error_type = error_type

    # ── DetectorProtocol methods ────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Run detection only (no embeddings)."""
        with self._lock:
            self._call_count += 1
            if self._error_count > 0:
                self._error_count -= 1
                self._inference_error_count += 1
                raise self._error_type("injected error")
            dets = self._factory(frame)
            return [
                DetectionResult(bbox=d.bbox, confidence=d.confidence)
                for d in dets
            ]

    def detect_with_embeddings(self, frame: np.ndarray) -> List[DetectionWithEmbedding]:
        """Run detection + embedding extraction."""
        with self._lock:
            self._call_count += 1
            if self._error_count > 0:
                self._error_count -= 1
                self._inference_error_count += 1
                raise self._error_type("injected inference error")
            return self._factory(frame)

    def reload_model(self, det_size: int | None = None) -> None:
        """Simulate model hot-swap."""
        with self._lock:
            if det_size is not None:
                self.det_size = det_size
            self._model_reloads += 1

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def model_reloads(self) -> int:
        return self._model_reloads


# ═══════════════════════════════════════════════════════════════════
# FakeRecognizer — controllable recognition results
# ═══════════════════════════════════════════════════════════════════

class FakeRecognizer:
    """Mock face recognizer with controllable match outcomes.

    By default returns UNKNOWN_SENTINEL.  Use register() to add known faces.
    Supports simulated latency for large-gallery benchmarks.
    """

    def __init__(self, tolerance: float = 0.45):
        self.tolerance = tolerance
        self._cached_encodings: Optional[np.ndarray] = None
        self._cached_names: List[str] = []
        self._db_version = -1
        self._simulate_latency_ms: float = 0.0
        self._error_on_next: Optional[Exception] = None
        self._update_count = 0

    @property
    def cached_names(self) -> List[str]:
        return list(self._cached_names)

    def set_latency(self, ms: float) -> None:
        """Simulate recognition latency (for large-gallery benchmarks)."""
        self._simulate_latency_ms = ms

    def inject_error(self, exc: Exception) -> None:
        """Next recognize() call will raise this exception."""
        self._error_on_next = exc

    def register(self, names_and_encodings: List[tuple]) -> None:
        """Register known faces: [(name, encoding), ...]."""
        self._cached_names = [n for n, _ in names_and_encodings]
        encs = [e for _, e in names_and_encodings]
        if encs:
            mat = np.array(encs, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            np.divide(mat, np.maximum(norms, 1e-12), out=mat)
            self._cached_encodings = mat
        else:
            self._cached_encodings = None
        self._db_version += 1
        self._update_count += 1

    def update_cache(
        self,
        known_encodings: List[np.ndarray],
        known_names: List[str],
        db_version: int = 0,
    ) -> bool:
        """Match FaceRecognizer.update_cache() contract."""
        if db_version == self._db_version and self._cached_encodings is not None:
            return False
        if len(known_encodings) != len(known_names):
            raise ValueError(
                f"Length mismatch: {len(known_encodings)} vs {len(known_names)}"
            )
        if len(known_encodings) == 0:
            self._cached_encodings = None
            self._cached_names = []
        else:
            mat = np.array(known_encodings, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            np.divide(mat, np.maximum(norms, 1e-12), out=mat)
            self._cached_encodings = mat
            self._cached_names = list(known_names)
        self._db_version = db_version
        self._update_count += 1
        return True

    def recognize(
        self,
        unknown_encoding: np.ndarray,
        known_encodings=None,
        known_names=None,
    ):
        """1:N recognition — matches FaceRecognizer.recognize() contract."""
        if self._error_on_next is not None:
            err = self._error_on_next
            self._error_on_next = None
            raise err

        if self._simulate_latency_ms > 0:
            time.sleep(self._simulate_latency_ms / 1000.0)

        # Explicit args take priority
        if known_encodings is not None and len(known_encodings) > 0:
            return self._match(unknown_encoding, known_encodings, known_names)

        if self._cached_encodings is not None and len(self._cached_names) > 0:
            return self._match(
                unknown_encoding, self._cached_encodings, self._cached_names
            )

        return UNKNOWN_SENTINEL, 0.0

    def _match(self, query, gallery, names):
        """Internal cosine-similarity match."""
        if query is None:
            return UNKNOWN_SENTINEL, 0.0
        query = np.asarray(query, dtype=np.float32).ravel()
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        if isinstance(gallery, list):
            gallery = np.array(gallery, dtype=np.float32)
        # Handle dimension mismatch gracefully
        if query.shape[0] != gallery.shape[1]:
            return UNKNOWN_SENTINEL, 0.0
        similarities = query @ gallery.T
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        if best_sim < self.tolerance:
            return UNKNOWN_SENTINEL, 0.0
        if isinstance(names, np.ndarray):
            names = names.tolist()
        return names[best_idx], best_sim


# ═══════════════════════════════════════════════════════════════════
# Shared stress fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def fake_camera() -> FakeCamera:
    """Fresh FakeCamera for each test."""
    return FakeCamera()


@pytest.fixture
def fake_detector() -> FakeDetector:
    """Fresh FakeDetector with default 2-face output."""
    return FakeDetector()


@pytest.fixture
def fake_recognizer() -> FakeRecognizer:
    """Fresh FakeRecognizer."""
    return FakeRecognizer(tolerance=0.45)


@pytest.fixture
def fake_pipeline_components(fake_camera, fake_detector, fake_recognizer):
    """Assemble a full pipeline from fake components + real tracker + real db."""
    from face_hub import FaceHubPipeline, FaceTracker, FaceDatabase

    tracker = FaceTracker(smooth_frames=5)
    db = FaceDatabase(db_path="stress_test_db.json", encoding_path="stress_test_enc.npy")

    pipeline = FaceHubPipeline(fake_camera, fake_detector, fake_recognizer, tracker, db)
    yield pipeline

    # Teardown: clean up temp DB files
    pipeline.stop()
    import os
    for p in ["stress_test_db.json", "stress_test_enc.npy", "stress_test_enc.pkl"]:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


@pytest.fixture
def thread_counter():
    """Count threads before and after a test, assert no leak."""
    before = threading.active_count()
    yield
    after = threading.active_count()
    # Allow ±2 tolerance for pytest internals
    assert after - before <= 2, (
        f"Thread leak detected: {before} → {after} "
        f"(new threads: {threading.enumerate()[-max(0, after-before):]})"
    )


# ═══════════════════════════════════════════════════════════════════
# Stress test marker registration
# ═══════════════════════════════════════════════════════════════════

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "stress: stress/stability test (skip by default, use --run-stress)",
    )
    config.addinivalue_line(
        "markers",
        "slow: real model or long-running test (skip by default)",
    )
    config.addinivalue_line(
        "markers",
        "memory: memory regression test (skip by default)",
    )
    config.addinivalue_line(
        "markers",
        "hardware: requires camera/GPU (skip in CI)",
    )


def pytest_addoption(parser):
    parser.addoption("--run-stress", action="store_true", help="run stress tests")
    parser.addoption("--run-slow", action="store_true", help="run slow tests")
    parser.addoption("--run-memory", action="store_true", help="run memory tests")
    parser.addoption("--run-hardware", action="store_true", help="run hardware tests")


def pytest_collection_modifyitems(config, items):
    """Skip marked tests unless the corresponding flag is passed."""
    markers_map = {
        "stress": "--run-stress",
        "slow": "--run-slow",
        "memory": "--run-memory",
        "hardware": "--run-hardware",
    }
    for marker, flag in markers_map.items():
        if not config.getoption(flag, default=False):
            skip = pytest.mark.skip(reason=f"need {flag} to run")
            for item in items:
                if marker in item.keywords:
                    item.add_marker(skip)
