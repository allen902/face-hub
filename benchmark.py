"""
FaceVision runtime performance benchmark.

Measures end-to-end pipeline FPS and per-component latency:
  - FaceDetector.detect_with_embeddings
  - FaceRecognizer.recognize
  - FaceTracker.update
  - FaceVisionPipeline.process_frame

Usage:
    # Live camera benchmark
    python benchmark.py --live --frames 300

    # Synthetic frames (no camera required)
    python benchmark.py --synthetic --frames 300 --det_size 640

    # Component-only benchmark (no model reload between tests)
    python benchmark.py --synthetic --component --frames 1000
"""

from __future__ import annotations

import argparse
import statistics
import time
import tracemalloc
from collections import deque
from typing import Callable, List

import numpy as np

from face_vision import (
    CameraThread,
    FaceDatabase,
    FaceDetector,
    FaceRecognizer,
    FaceTracker,
    FaceVisionPipeline,
)


def percentile(values: List[float], p: float) -> float:
    """Return the p-th percentile of a sorted list."""
    if not values:
        return 0.0
    k = (len(values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize(name: str, latencies_ms: List[float]) -> None:
    """Print latency statistics."""
    latencies_ms.sort()
    total = sum(latencies_ms)
    n = len(latencies_ms)
    mean = total / n
    p50 = percentile(latencies_ms, 50)
    p95 = percentile(latencies_ms, 95)
    p99 = percentile(latencies_ms, 99)
    fps = 1000.0 / mean if mean > 0 else 0.0
    print(f"\n{name} ({n} samples)")
    print(f"  mean : {mean:7.2f} ms  ({fps:6.2f} FPS)")
    print(f"  p50  : {p50:7.2f} ms")
    print(f"  p95  : {p95:7.2f} ms")
    print(f"  p99  : {p99:7.2f} ms")
    print(f"  min  : {latencies_ms[0]:7.2f} ms")
    print(f"  max  : {latencies_ms[-1]:7.2f} ms")


def benchmark_detector(detector: FaceDetector, frame: np.ndarray, iterations: int) -> None:
    """Benchmark detection + embedding extraction."""
    # Warm-up
    for _ in range(3):
        detector.detect_with_embeddings(frame)

    times: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.detect_with_embeddings(frame)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    summarize("FaceDetector.detect_with_embeddings", times)


def benchmark_recognizer(recognizer: FaceRecognizer, encoding: np.ndarray, iterations: int) -> None:
    """Benchmark 1:N recognition."""
    # Seed cache with a small gallery
    rng = np.random.RandomState(0)
    gallery = [rng.randn(512).astype(np.float32) for _ in range(50)]
    names = [f"person_{i}" for i in range(50)]
    recognizer.update_cache(gallery, names, db_version=1)

    # Warm-up
    for _ in range(100):
        recognizer.recognize(encoding)

    times: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        recognizer.recognize(encoding)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    summarize("FaceRecognizer.recognize", times)


def benchmark_tracker(tracker: FaceTracker, detector: FaceDetector, frame: np.ndarray, iterations: int) -> None:
    """Benchmark tracker update."""
    recognizer = FaceRecognizer(tolerance=0.45)
    # Seed cache
    rng = np.random.RandomState(1)
    gallery = [rng.randn(512).astype(np.float32) for _ in range(20)]
    names = [f"person_{i}" for i in range(20)]
    recognizer.update_cache(gallery, names, db_version=1)

    detections = detector.detect_with_embeddings(frame)
    if not detections:
        print("\nFaceTracker.update: no detections on sample frame, skipping")
        return

    # Warm-up
    for _ in range(10):
        tracker.update(detections, recognizer=recognizer)

    times: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        tracker.update(detections, recognizer=recognizer)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    summarize("FaceTracker.update", times)


def benchmark_pipeline_live(camera_id: int, width: int, height: int,
                            device: str, det_size: int, frames: int) -> None:
    """Benchmark the full pipeline with a live camera."""
    print(f"\nLive camera benchmark: camera={camera_id} {width}x{height}")
    print(f"Detector: device={device}, det_size={det_size}")

    camera = CameraThread(camera_id=camera_id, width=width, height=height)
    detector = FaceDetector(device=device, det_size=det_size)
    recognizer = FaceRecognizer(tolerance=0.45)
    tracker = FaceTracker(smooth_frames=5)
    db = FaceDatabase(db_path="bench_db.json", encoding_path="bench_enc.pkl")

    pipeline = FaceVisionPipeline(camera, detector, recognizer, tracker, db)
    pipeline.start()

    times: List[float] = []
    latencies: deque[float] = deque(maxlen=30)
    try:
        for i in range(frames):
            t0 = time.perf_counter()
            result = pipeline.process_frame()
            t1 = time.perf_counter()
            if result is not None:
                elapsed = (t1 - t0) * 1000.0
                times.append(elapsed)
                latencies.append(elapsed)
                if (i + 1) % 30 == 0:
                    avg = statistics.mean(latencies)
                    print(f"  frame {i+1:4d}: avg={avg:6.2f} ms  fps={1000/avg:5.1f}")
    finally:
        pipeline.stop()

    summarize("FaceVisionPipeline.process_frame (live)", times)


def benchmark_pipeline_synthetic(width: int, height: int, device: str,
                                 det_size: int, frames: int) -> None:
    """Benchmark the full pipeline on synthetic frames."""
    print(f"\nSynthetic frame benchmark: {width}x{height}")
    print(f"Detector: device={device}, det_size={det_size}")

    detector = FaceDetector(device=device, det_size=det_size)
    recognizer = FaceRecognizer(tolerance=0.45)
    tracker = FaceTracker(smooth_frames=5)
    db = FaceDatabase(db_path="bench_db.json", encoding_path="bench_enc.pkl")

    # Mock camera: only implements the attributes Pipeline needs
    class MockCamera:
        running = False
        def start(self):
            self.running = True
        def stop(self):
            self.running = False
        def get_frame(self, timeout=None, copy=True):
            return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    pipeline = FaceVisionPipeline(MockCamera(), detector, recognizer, tracker, db)
    pipeline.start()

    times: List[float] = []
    try:
        for i in range(frames):
            t0 = time.perf_counter()
            pipeline.process_frame()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    finally:
        pipeline.stop()

    summarize("FaceVisionPipeline.process_frame (synthetic)", times)


def benchmark_components(width: int, height: int, device: str, det_size: int) -> None:
    """Benchmark each component independently."""
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    print("\n=== Component benchmark ===")
    detector = FaceDetector(device=device, det_size=det_size)
    benchmark_detector(detector, frame, iterations=100)

    encoding = np.random.randn(512).astype(np.float32)
    recognizer = FaceRecognizer(tolerance=0.45)
    benchmark_recognizer(recognizer, encoding, iterations=10_000)

    tracker = FaceTracker(smooth_frames=5)
    benchmark_tracker(tracker, detector, frame, iterations=1_000)


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceVision performance benchmark")
    parser.add_argument("--live", action="store_true", help="use live camera")
    parser.add_argument("--synthetic", action="store_true", help="use synthetic frames")
    parser.add_argument("--component", action="store_true", help="run component-level benchmark")
    parser.add_argument("--camera", type=int, default=0, help="camera index (live mode)")
    parser.add_argument("--width", type=int, default=640, help="frame width")
    parser.add_argument("--height", type=int, default=360, help="frame height")
    parser.add_argument("--device", type=str, default="auto", help="cpu/cuda/auto")
    parser.add_argument("--det_size", type=int, default=640, help="320/480/640")
    parser.add_argument("--frames", type=int, default=300, help="number of frames")
    args = parser.parse_args()

    if not args.live and not args.synthetic and not args.component:
        # Default: component + synthetic pipeline
        args.component = True
        args.synthetic = True

    tracemalloc.start()
    t_start = time.perf_counter()

    if args.component:
        benchmark_components(args.width, args.height, args.device, args.det_size)

    if args.synthetic:
        benchmark_pipeline_synthetic(
            args.width, args.height, args.device, args.det_size, args.frames
        )

    if args.live:
        benchmark_pipeline_live(
            args.camera, args.width, args.height, args.device, args.det_size, args.frames
        )

    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nTotal benchmark time: {elapsed:.2f} s")
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
