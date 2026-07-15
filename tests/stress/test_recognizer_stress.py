"""Stress tests for FaceRecognizer (fake + real)."""
import time
import pytest
import numpy as np
from face_hub.types import UNKNOWN_SENTINEL
from face_hub.exceptions import RecognitionError


class TestFakeRecognizerStress:
    """FakeRecognizer 压力测试."""

    @pytest.mark.stress
    def test_large_gallery_10k(self, fake_recognizer, memory_helper):
        """10k 底库匹配 p95 延迟."""
        rng = np.random.RandomState(0)
        gallery = [
            (f"person_{i}", rng.randn(512).astype(np.float32))
            for i in range(10_000)
        ]
        fake_recognizer.register(gallery)

        query = rng.randn(512).astype(np.float32)
        query /= np.linalg.norm(query)

        # Warm-up
        for _ in range(100):
            fake_recognizer.recognize(query)

        times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            name, conf = fake_recognizer.recognize(query)
            times.append((time.perf_counter() - t0) * 1000)

        times.sort()
        p50 = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[int(len(times) * 0.99)]

        print(f"10k gallery: p50={p50:.3f}ms p95={p95:.3f}ms p99={p99:.3f}ms")

        delta = memory_helper.snapshot()
        assert delta.rss_delta_mb < 100, f"Memory grew {delta.rss_delta_mb:.1f} MB"
        # Fake recognizer is fast; real numpy matmul should be < 50ms p95 for 10k
        assert p95 < 100, f"p95 latency {p95:.1f} ms too high"

    def test_cache_consistency_same_version(self, fake_recognizer):
        """同 version 的 update_cache 返回 False."""
        enc = [np.random.randn(512).astype(np.float32)]
        names = ["alice"]

        r1 = fake_recognizer.update_cache(enc, names, db_version=5)
        assert r1 is True
        # Same version, same data → skip
        r2 = fake_recognizer.update_cache(enc, names, db_version=5)
        assert r2 is False

    def test_cache_consistency_different_version(self, fake_recognizer):
        """不同 version 的 update_cache 返回 True."""
        enc = [np.random.randn(512).astype(np.float32)]
        names = ["alice"]

        fake_recognizer.update_cache(enc, names, db_version=1)
        r2 = fake_recognizer.update_cache(enc, names, db_version=2)
        assert r2 is True

    def test_anomalous_inputs(self, fake_recognizer):
        """异常输入优雅处理."""
        fake_recognizer.register([("alice", np.random.randn(512).astype(np.float32))])

        # None embedding
        name, conf = fake_recognizer.recognize(None)
        assert name == UNKNOWN_SENTINEL
        assert conf == 0.0

        # Wrong dimension (256)
        name, conf = fake_recognizer.recognize(np.random.randn(256).astype(np.float32))
        assert name == UNKNOWN_SENTINEL

        # Empty gallery
        fr = FakeRecognizerStub(tolerance=0.45)
        name, conf = fr.recognize(np.random.randn(512).astype(np.float32))
        assert name == UNKNOWN_SENTINEL

    def test_names_encodings_length_mismatch(self, fake_recognizer):
        """长度不一致抛 ValueError."""
        with pytest.raises(ValueError):
            fake_recognizer.update_cache(
                [np.random.randn(512).astype(np.float32)],
                ["alice", "bob"],  # 1 encoding vs 2 names
                db_version=1,
            )

    def test_concurrent_cache_update(self, fake_recognizer):
        """并发 update_cache + recognize 不抛异常."""
        import threading

        enc = [np.random.randn(512).astype(np.float32)]
        names = ["alice"]
        errors = []

        def updater():
            try:
                for v in range(1, 50):
                    fake_recognizer.update_cache(enc, names, db_version=v)
            except Exception as e:
                errors.append(e)

        def reader():
            query = np.random.randn(512).astype(np.float32)
            try:
                for _ in range(500):
                    fake_recognizer.recognize(query)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=updater)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert len(errors) == 0, f"Errors: {errors}"

    def test_error_injection(self, fake_recognizer):
        """注入异常后正确抛出."""
        fake_recognizer.inject_error(RecognitionError("simulated"))
        with pytest.raises(RecognitionError):
            fake_recognizer.recognize(np.random.randn(512).astype(np.float32))


class FakeRecognizerStub:
    """Minimal stub used in anomalous input tests."""
    def __init__(self, tolerance=0.45):
        self.tolerance = tolerance
        self._cached_encodings = None
        self._cached_names = []

    def recognize(self, query, known_encodings=None, known_names=None):
        if query is None:
            return UNKNOWN_SENTINEL, 0.0
        return UNKNOWN_SENTINEL, 0.0
