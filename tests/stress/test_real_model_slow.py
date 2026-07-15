"""Real-model benchmarks (nightly / --run-slow only).

Generates or updates tests/stress/baseline.json with performance thresholds.
"""
import json
import time
import pytest
import numpy as np
from pathlib import Path

BASELINE_PATH = Path(__file__).parent / "baseline.json"


def compute_percentiles(values):
    values.sort()
    n = len(values)
    p50 = values[n // 2]
    p95 = values[int(n * 0.95)]
    p99 = values[int(n * 0.99)]
    return {
        "mean": sum(values) / n,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "min": values[0],
        "max": values[-1],
        "samples": n,
    }


@pytest.mark.slow
@pytest.mark.hardware
class TestRealModelBenchmark:
    """真实模型性能基准（生成 baseline.json）."""

    def test_detector_benchmark(self):
        """FaceDetector 性能基准."""
        from face_hub import FaceDetector
        import tracemalloc

        detector = FaceDetector(device="cpu", det_size=640)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Warm-up
        for _ in range(5):
            detector.detect_with_embeddings(frame)

        tracemalloc.start()
        times = []
        for _ in range(300):
            t0 = time.perf_counter()
            detector.detect_with_embeddings(frame)
            times.append((time.perf_counter() - t0) * 1000)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        stats = compute_percentiles(times)
        stats["peak_memory_mb"] = peak / (1024 * 1024)
        stats["det_size"] = 640
        stats["device"] = "cpu"

        # Write baseline
        baseline = {}
        if BASELINE_PATH.exists():
            with open(BASELINE_PATH) as f:
                baseline = json.load(f)

        baseline["detector_cpu_640"] = stats
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_PATH, "w") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)

        print(f"Detector baseline: {stats['mean']:.1f}ms avg, p95={stats['p95']:.1f}ms")
        assert stats["mean"] > 0

    def test_recognizer_benchmark(self):
        """FaceRecognizer 大库性能基准."""
        from face_hub import FaceRecognizer

        rng = np.random.RandomState(0)
        gallery = [rng.randn(512).astype(np.float32) for _ in range(1000)]
        names = [f"person_{i}" for i in range(1000)]
        query = rng.randn(512).astype(np.float32)
        query /= np.linalg.norm(query)

        recognizer = FaceRecognizer(tolerance=0.45)
        recognizer.update_cache(gallery, names, db_version=1)

        # Warm-up
        for _ in range(500):
            recognizer.recognize(query)

        times = []
        for _ in range(10_000):
            t0 = time.perf_counter()
            recognizer.recognize(query)
            times.append((time.perf_counter() - t0) * 1000)

        stats = compute_percentiles(times)
        stats["gallery_size"] = 1000

        baseline = {}
        if BASELINE_PATH.exists():
            with open(BASELINE_PATH) as f:
                baseline = json.load(f)

        baseline["recognizer_1k"] = stats
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_PATH, "w") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)

        print(f"Recognizer (1k gallery): {stats['mean']:.3f}ms avg")
        assert stats["mean"] < 10, f"Recognition too slow: {stats['mean']:.3f}ms"

    def test_pipeline_end_to_end_benchmark(self):
        """端到端流水线性能基准 (300 帧)."""
        from face_hub import FaceHubPipeline, FaceDetector, FaceRecognizer, FaceTracker, FaceDatabase

        class FakeCam:
            running = False

            def start(self):
                self.running = True

            def stop(self):
                self.running = False

            def get_frame(self, timeout=None, copy=True):
                return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detector = FaceDetector(device="cpu", det_size=640)
        recognizer = FaceRecognizer(tolerance=0.45)
        tracker = FaceTracker(smooth_frames=5)
        db = FaceDatabase(
            db_path="bench_real_db.json",
            encoding_path="bench_real_enc.npy",
        )

        pipeline = FaceHubPipeline(FakeCam(), detector, recognizer, tracker, db)
        pipeline.start()

        times = []
        for _ in range(300):
            t0 = time.perf_counter()
            pipeline.process_frame()
            times.append((time.perf_counter() - t0) * 1000)

        pipeline.stop()

        # Cleanup
        for p in ["bench_real_db.json", "bench_real_enc.npy", "bench_real_enc.pkl"]:
            try:
                import os
                os.remove(p)
            except FileNotFoundError:
                pass

        stats = compute_percentiles(times)
        stats["det_size"] = 640
        stats["device"] = "cpu"

        baseline = {}
        if BASELINE_PATH.exists():
            with open(BASELINE_PATH) as f:
                baseline = json.load(f)

        baseline["pipeline_e2e_cpu_640"] = stats
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_PATH, "w") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)

        fps = 1000.0 / stats["mean"] if stats["mean"] > 0 else 0
        print(f"Pipeline E2E: {stats['mean']:.1f}ms avg ({fps:.1f} FPS)")
