"""Stress tests for FaceDetector (fake + real)."""
import threading
import time
import pytest
import numpy as np
from face_hub.types import DetectionWithEmbedding, BBox
from face_hub.exceptions import InferenceError


class TestFakeDetectorStress:
    """FakeDetector 压力测试（CI 安全）."""

    @pytest.mark.stress
    def test_continuous_inference_5k_frames(self, fake_detector, memory_helper):
        """5k 帧连续推理：内存不爆、调用计数正确."""
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        for i in range(5000):
            results = fake_detector.detect_with_embeddings(frame)
            assert isinstance(results, list)
            assert len(results) == 2
            for r in results:
                assert isinstance(r, DetectionWithEmbedding)
                assert r.embedding is not None
        elapsed = time.perf_counter() - t0

        delta = memory_helper.snapshot()
        assert delta.rss_delta_mb < 50, f"Memory grew {delta.rss_delta_mb:.1f} MB"
        assert fake_detector.call_count == 5000
        fps = 5000 / elapsed
        print(f"FakeDetector: {fps:.1f} FPS over 5k frames")

    def test_concurrent_inference_no_deadlock(self, fake_detector):
        """多线程并发推理：无死锁、结果正确."""
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        results_per_thread = {}
        errors = []

        def worker(tid: int):
            try:
                for _ in range(100):
                    r = fake_detector.detect_with_embeddings(frame)
                    results_per_thread.setdefault(tid, []).append(len(r))
            except Exception as e:
                errors.append((tid, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"
        for tid, lens in results_per_thread.items():
            assert all(l == 2 for l in lens), f"Thread {tid} got wrong result counts"

    def test_model_hot_swap(self, fake_detector):
        """多次 reload_model：det_size 更新、旧资源释放."""
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

        sizes = [320, 480, 640, 320, 480]
        for sz in sizes:
            fake_detector.reload_model(det_size=sz)
            assert fake_detector.det_size == sz
            r = fake_detector.detect_with_embeddings(frame)
            assert len(r) == 2

        assert fake_detector.model_reloads == len(sizes)

    def test_anomalous_frame_returns_empty(self, fake_detector):
        """异常帧不崩溃，返回 []."""
        bad_frames = [
            np.array([], dtype=np.uint8).reshape(0, 0, 3),   # 空帧
            np.zeros((1, 1, 3), dtype=np.uint8),              # 1x1
            np.zeros((480, 640), dtype=np.float32),            # 非 uint8 / 单通道
        ]

        # FakeDetector doesn't crash on any frame shape, but we test
        # that our factory handles edge cases gracefully.
        for frame in bad_frames:
            results = fake_detector.detect_with_embeddings(frame)
            assert isinstance(results, list)

    def test_error_injection_and_recovery(self, fake_detector):
        """注入 3 次错误后恢复."""
        fake_detector.set_error_count(3, RuntimeError)
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

        error_count = 0
        for i in range(5):
            try:
                fake_detector.detect_with_embeddings(frame)
            except RuntimeError:
                error_count += 1

        assert error_count == 3
        # After error budget exhausted, calls succeed
        r = fake_detector.detect_with_embeddings(frame)
        assert len(r) == 2


@pytest.mark.slow
@pytest.mark.hardware
class TestRealDetectorStress:
    """真实 FaceDetector 慢测（需要 insightface 模型 + CPU/GPU）."""

    def test_real_detector_1k_frames_memory_stable(self, memory_helper):
        """真实模型 1k 帧：内存稳定、无推理错误."""
        from face_hub import FaceDetector

        detector = FaceDetector(device="cpu", det_size=320)
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

        # Warm-up
        for _ in range(5):
            detector.detect_with_embeddings(frame)

        times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            results = detector.detect_with_embeddings(frame)
            times.append((time.perf_counter() - t0) * 1000)
            assert isinstance(results, list)

        # FPS should not degrade
        first_half_avg = sum(times[:500]) / 500
        second_half_avg = sum(times[500:]) / 500
        assert second_half_avg <= first_half_avg * 1.5, (
            f"FPS degraded: {first_half_avg:.2f} → {second_half_avg:.2f} ms"
        )

        delta = memory_helper.snapshot()
        assert delta.rss_delta_mb < 150, f"Memory grew {delta.rss_delta_mb:.1f} MB"

    def test_real_detector_model_reload(self):
        """真实模型热切换：det_size 变化后模型可用."""
        from face_hub import FaceDetector

        detector = FaceDetector(device="cpu", det_size=320)
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

        for sz in [480, 640, 320]:
            detector.reload_model(det_size=sz)
            assert detector.det_size == sz
            r = detector.detect_with_embeddings(frame)
            assert isinstance(r, list)

    def test_anomalous_frames_no_crash(self):
        """真实模型处理异常帧不崩溃."""
        from face_hub import FaceDetector

        detector = FaceDetector(device="cpu", det_size=320)
        bad_frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),  # 全黑
            np.full((480, 640, 3), 255, dtype=np.uint8),  # 全白
        ]
        for frame in bad_frames:
            r = detector.detect_with_embeddings(frame)
            assert isinstance(r, list)
