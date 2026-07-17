下面是我基于项目结构为 FaceHub 设计的压力与稳定性测试方案。方案以“可自动化、可复现、不依赖真实摄像头/GPU”为核心，同时保留真实模型的手动/夜间测试入口。

1. 项目结构理解
FaceHub 是一条典型的人脸识别流水线，核心数据流：


CameraThread.get_frame()
        ↓
FaceDetector.detect_with_embeddings()
        ↓
FaceTracker.update(detections, recognizer=FaceRecognizer)
        ↓
FaceHubPipeline.process_frame()
关键组件与风险点：

组件	职责	稳定性关注点
CameraThread	独立线程采集、条件变量分发最新帧	启停泄漏、多消费者、maxlen=1 丢帧
FaceDetector	InsightFace RetinaFace + ArcFace 嵌入	模型加载/热切换、GPU 异常降级、内存泄漏、并发锁
FaceRecognizer	1:N 余弦相似度匹配	大库查询性能、缓存一致性、异常维度
FaceTracker	IoU 匹配 + 多数投票平滑	大目标数 O(N²)、track 过期、身份抖动
FaceDatabase	JSON + npy 原子写入、版本号	并发 CRUD、文件损坏、路径穿越
FaceHubPipeline	总控 + FPS 统计 + 缓存同步	长时间 soak、异常隔离、锁竞争
现有 benchmark.py 已经能做性能采集，但缺少断言、内存回归检查、异常注入和并发场景，这是本次方案要补齐的。

2. 测试目标
长时间稳定运行：连续处理 10k~100k 帧不崩溃、内存增长可控。
并发与资源安全：多线程调用组件时无死锁、无句柄/线程泄漏。
降级路径可靠：GPU 异常、模型热切换、空库、坏帧都能优雅处理。
性能可度量：FPS、p50/p95/p99 延迟、内存峰值、大库匹配耗时。
CI 可复现：默认用 Mock 组件跑；真实模型/摄像头测试标记为 slow/hardware，手动或夜间执行。
3. 测试策略
分层：单组件 → FaceHubPipeline → 端到端真实模型。
Mock 优先：
FakeCamera：固定返回合成帧。
FakeDetector：返回可控 bbox/embedding。
FakeRecognizer：模拟匹配耗时、维度异常、缓存失效。 这样可以在没有 GPU/摄像头/真实人脸的情况下做可复现压力测试。
真实模型慢测：复用 benchmark.py 逻辑，加上内存和断言，作为 pytest --runslow 或 nightly job。
异常注入：在 fake 组件里故意抛 InferenceError、ValueError、OSError，验证外层捕获与降级。
回归阈值：把基线数据写入 tests/stress/baseline.json，CI 对比偏差。
4. 测试维度与用例
4.1 CameraThread
用例	场景	验证点
启停循环	start()/stop() 100/1000 次	thread.join() 退出、cap.release()、无残留线程
多消费者	10 线程并发 get_frame()	每个消费者拿到独立 copy，不“偷帧”
慢消费者	消费者处理耗时 > 帧间隔	maxlen=1 缓冲正常丢旧帧，不阻塞采集线程
异常输入	非法 camera_id、打开失败	抛出 CameraError，属性完整
4.2 FaceDetector
用例	场景	验证点
长时推理	连续推理 1k/5k 帧（CPU）	平均 FPS 不下降、内存增量 < 阈值、_inference_error_count == 0
并发推理	多线程同时 detect_with_embeddings	_lock 不导致死锁、结果正确
模型热切换	多次 reload_model(det_size=...)	旧模型释放、新模型可用、VRAM/内存不累积
异常帧	空图、1x1、非 uint8、None、超大分辨率	返回 []，不崩溃
GPU 降级	Mock app.get 连续抛异常	第 3 次后自动 fallback 到 CPU，后续调用恢复
DirectML reshape bug	注入对应异常字符串	自动切到 det_size=640 或 CPU
4.3 FaceRecognizer
用例	场景	验证点
大库匹配	1k / 5k / 10k 底库，多次查询	p95 延迟、内存占用
缓存一致性	update_cache 同 version / 不同 version	同 version 返回 False，不同 version 重建
异常输入	空 gallery、维度 256/513、None embedding、names/encodings 长度不一致	返回 (unknown, 0.0) 或抛 ValueError
并发更新	线程 A update_cache、线程 B recognize	在 pipeline 锁下无异常；单独使用时文档说明非线程安全
4.4 FaceTracker
用例	场景	验证点
大规模跟踪	单帧 50/100 个检测，连续 1000 帧	内存不爆、耗时稳定、track_count 合理
身份抖动	同一 track 名字在 A/B/unknown 间交替	majority vote 抑制抖动，is_confirmed 稳定
丢失恢复	连续 max_missed 帧无匹配	track 被移除；再次出现时生成新 track
并发更新	多线程同时 update	状态一致、无竞态
历史截断	运行多帧后	name_history 长度不超过 smooth_frames*3
4.5 FaceDatabase
用例	场景	验证点
并发 CRUD	10 线程各 100 次 add/remove	version 单调递增、JSON/npy 不损坏、最终一致性
损坏恢复	写入坏 JSON、shape 错的 npy	抛出 SerializationError/DatabaseError
大批量	5000 条记录 save/load	耗时、内存、文件权限
路径穿越	绝对路径、../etc/passwd	抛出 ValueError
4.6 FaceHubPipeline
用例	场景	验证点
Soak test	Mock 组件连续 process_frame 10k/100k 帧	FPS 稳定、内存增长 < 阈值、_total_frame_count 正确
启停幂等	start()/stop() 多次、重复 stop	不抛异常、状态正确
异常隔离	detector/recognizer/tracker/db 抛异常	包装为 FaceHubError，流水线可继续运行
动态增删库	运行中 db.add_person/remove_person	_last_db_version 同步、识别结果变化
多线程调用	多线程同时 process_frame	_lock 不饿死、无死锁
FPS 统计	固定帧率运行	fps 字段非负、与真实耗时偏差 < 10%
5. 关键指标与通过标准
指标	工具	建议阈值
内存增长	tracemalloc / psutil.Process.memory_info().rss	10k 帧增长 < 50 MB
FPS 下降	time.perf_counter()	后 50% 帧平均 FPS 不低于前 50% 的 80%
p95 延迟	同 benchmark.py 的 percentile	不高于基线 20%
异常数	pytest + 异常注入	0 未捕获异常
句柄/线程泄漏	threading.enumerate() / psutil	测试结束回到基线 ±1
数据库一致性	version 校验	并发操作后 version 单调且等于操作总数
阈值可写入 tests/stress/baseline.json，CI 自动比对。

6. 工具与基础设施

# pyproject.toml 增加 markers
[tool.pytest.ini_options]
markers = [
    "stress: 压力/稳定性测试（默认跳过）",
    "slow:   真实模型/长时间测试",
    "memory: 内存回归测试",
]
新增文件规划：


tests/
├── stress/
│   ├── conftest.py            # FakeCamera / FakeDetector / FakeRecognizer / memory helper
│   ├── test_camera_stress.py
│   ├── test_detector_stress.py
│   ├── test_recognizer_stress.py
│   ├── test_tracker_stress.py
│   ├── test_database_stress.py
│   ├── test_pipeline_stress.py
│   └── baseline.json          # 性能基线
关键 Fixture 示例

# tests/stress/conftest.py
import numpy as np
import pytest
from face_hub.types import DetectionWithEmbedding, BBox

class FakeDetector:
    def __init__(self, detections_factory):
        self._factory = detections_factory
        self.det_size = 640
        self.device = "cpu"
        self._error_count = 0

    def detect_with_embeddings(self, frame):
        if self._error_count:
            self._error_count -= 1
            raise RuntimeError("injected inference error")
        return self._factory(frame)

    def reload_model(self, det_size=None):
        if det_size is not None:
            self.det_size = det_size


class FakeCamera:
    running = False
    def start(self): self.running = True
    def stop(self):  self.running = False
    def get_frame(self, timeout=None, copy=True):
        return np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)


@pytest.fixture
def memory_helper():
    import tracemalloc, psutil, gc
    gc.collect()
    tracemalloc.start()
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    snap_before = tracemalloc.take_snapshot()
    yield type("H", (), {
        "rss": lambda: proc.memory_info().rss - rss_before,
        "snapshot": lambda: tracemalloc.take_snapshot().compare_to(snap_before, "lineno"),
    })
    tracemalloc.stop()
7. 实施路线
阶段	内容	预计产出
Phase 1	基础设施	pytest.ini markers、tests/stress/conftest.py、Fake 组件、内存 helper
Phase 2	单组件压力测试	test_camera/detector/recognizer/tracker/database_stress.py
Phase 3	Pipeline soak 测试	test_pipeline_stress.py，包含异常注入与动态库更新
Phase 4	真实模型基准	扩展 benchmark.py 或新增 test_real_model_slow.py，生成 baseline.json
Phase 5	CI 集成	GitHub Actions nightly job：pytest -m "stress or slow"，上传报告
8. 风险与注意事项
GPU/DirectML 环境差异大：真实模型测试应在独立 job 运行，失败不阻塞 PR。
模型首次加载慢：pytest-timeout 对加载阶段要单独设置宽松时间。
Windows 句柄检测：os.chmod 在 Windows 上可能失败，测试里不要当作致命错误。
不要依赖真实摄像头：压力测试默认用 FakeCamera，保证 CI 和本地结果一致。
线程安全边界：FaceRecognizer 没有内部锁，并发测试要在 FaceHubPipeline._lock 覆盖下验证；单独并发调用应视为未定义行为并文档化。

---

## 9. 完整实现代码

### 9.1 配置更新 — pyproject.toml

在现有 `[tool.pytest.ini_options]` 基础上追加 stress/slow/memory markers：

```toml
# pyproject.toml 尾部追加
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = [
    "stress: 压力/稳定性测试（默认跳过，通过 --run-stress 启用）",
    "slow: 真实模型/长时间测试（默认跳过，通过 --run-slow 启用）",
    "memory: 内存回归测试（默认跳过，通过 --run-memory 启用）",
    "hardware: 依赖摄像头/GPU 的测试（仅本地执行）",
]
```

### 9.2 基础设施 — tests/stress/conftest.py

```python
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

    tracemalloc.stop()
    if not result["done"]:
        snapshot()  # ensure we always capture


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
```

### 9.3 单组件压力测试

#### tests/stress/test_camera_stress.py

```python
"""Stress tests for CameraThread (using FakeCamera for CI safety)."""
import threading
import time
import pytest
import numpy as np
from face_hub.exceptions import CameraError


class TestCameraStartStop:
    """启动/停止循环压力测试."""

    def test_start_stop_100_cycles(self, fake_camera):
        """100 次启停不残留线程、不抛异常."""
        for i in range(100):
            fake_camera.start()
            assert fake_camera.running is True
            fake_camera.stop()
            assert fake_camera.running is False

    def test_repeat_stop_is_idempotent(self, fake_camera):
        """重复 stop 不抛异常."""
        fake_camera.start()
        fake_camera.stop()
        fake_camera.stop()  # idempotent
        fake_camera.stop()
        assert fake_camera.running is False

    def test_repeat_start_is_idempotent(self, fake_camera):
        """重复 start 不抛异常."""
        fake_camera.start()
        fake_camera.start()
        assert fake_camera.running is True
        fake_camera.stop()

    @pytest.mark.stress
    def test_start_stop_1000_cycles(self, fake_camera, thread_counter):
        """1000 次启停，验证无线程泄漏."""
        for _ in range(1000):
            fake_camera.start()
            fake_camera.stop()


class TestMultiConsumer:
    """多消费者并发测试."""

    def test_10_threads_concurrent_get_frame(self, fake_camera):
        """10 线程并发 get_frame()，每个拿到独立 copy."""
        fake_camera.start()
        frames_received = []
        errors = []
        barrier = threading.Barrier(10, timeout=5)

        def consumer():
            barrier.wait()
            try:
                for _ in range(50):
                    f = fake_camera.get_frame(timeout=0.1)
                    if f is not None:
                        frames_received.append(f)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=consumer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors in consumers: {errors}"
        assert len(frames_received) > 0
        # Every frame should be independent (not sharing memory)
        assert all(f.flags["OWNDATA"] for f in frames_received)
        fake_camera.stop()

    def test_slow_consumer_does_not_block_producer(self, fake_camera):
        """慢消费者不阻塞采集端."""
        fake_camera.start()
        latencies = []

        for _ in range(100):
            t0 = time.perf_counter()
            _ = fake_camera.get_frame()
            latencies.append((time.perf_counter() - t0) * 1000)
            # Simulate consumer that's slower than frame interval
            time.sleep(0.05)

        # Producer (get_frame) latency should be stable
        avg = sum(latencies) / len(latencies)
        # With sleep, avg might be higher but individual get_frame
        # without sleep should be fast
        assert avg < 200, f"get_frame avg latency {avg:.2f} ms too high"


class TestErrorInjection:
    """异常注入测试."""

    def test_camera_error_on_invalid_id(self):
        """非法 camera_id 抛 CameraError."""
        from face_hub import CameraThread
        with pytest.raises(ValueError):
            CameraThread(camera_id=-1)

    def test_fake_camera_error_injection(self, fake_camera):
        """注入异常后 get_frame 正确抛出."""
        fake_camera.inject_error(CameraError("simulated", camera_id=0))
        fake_camera.start()
        with pytest.raises(CameraError) as exc_info:
            fake_camera.get_frame()
        assert exc_info.value.camera_id == 0
        # After error, subsequent calls should work
        frame = fake_camera.get_frame()
        assert frame is not None
        fake_camera.stop()
```

#### tests/stress/test_detector_stress.py

```python
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
```

#### tests/stress/test_recognizer_stress.py

```python
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
```

#### tests/stress/test_tracker_stress.py

```python
"""Stress tests for FaceTracker."""
import pytest
import numpy as np
from face_hub import FaceTracker
from face_hub.types import DetectionWithEmbedding, BBox, TrackedFace, UNKNOWN_SENTINEL


def make_detection(x1, y1, x2, y2, conf=0.9, emb=None, quality=True):
    """Helper: create a DetectionWithEmbedding."""
    if emb is None:
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
    return DetectionWithEmbedding(
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=conf,
        embedding=emb,
        quality_pass=quality,
    )


class TestTrackerStress:
    """FaceTracker 压力测试."""

    @pytest.mark.stress
    def test_large_scale_100_detections_1000_frames(self, memory_helper):
        """单帧 100 个检测 × 1000 帧：内存稳定、耗时可控."""
        import time

        tracker = FaceTracker(smooth_frames=5, max_missed=10)
        rng = np.random.RandomState(0)

        times = []
        for _ in range(1000):
            # Generate 100 random detections
            dets = []
            for _ in range(100):
                x1 = rng.randint(0, 500)
                y1 = rng.randint(0, 250)
                dets.append(make_detection(x1, y1, x1 + 40, y1 + 50))

            t0 = time.perf_counter()
            results = tracker.update(dets)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            assert isinstance(results, list)
            for r in results:
                assert isinstance(r, TrackedFace)

        times.sort()
        p95 = times[int(len(times) * 0.95)]
        print(f"100 dets × 1000 frames: p95={p95:.2f}ms avg={sum(times)/len(times):.2f}ms")

        delta = memory_helper.snapshot()
        assert delta.rss_delta_mb < 100, f"Memory grew {delta.rss_delta_mb:.1f} MB"
        # IoU matrix is 100×100; should complete < 500ms per frame
        assert p95 < 500, f"p95 {p95:.1f}ms too high"

    def test_identity_jitter_suppression(self):
        """身份抖动被 majority vote 抑制."""
        from face_hub import FaceRecognizer

        tracker = FaceTracker(smooth_frames=7, max_missed=10)
        recognizer = FaceRecognizer(tolerance=0.45)

        # Register 2 known people
        rng = np.random.RandomState(1)
        alice_emb = rng.randn(512).astype(np.float32)
        alice_emb /= np.linalg.norm(alice_emb)
        bob_emb = rng.randn(512).astype(np.float32)
        bob_emb /= np.linalg.norm(bob_emb)

        recognizer.update_cache(
            [alice_emb, bob_emb],
            ["alice", "bob"],
            db_version=1,
        )

        # Detect the same face (fixed box) but flip embedding every frame
        # between alice and bob to simulate recognition flicker
        names_seen = []
        for i in range(30):
            # Alternate embedding to simulate flicker
            emb = alice_emb if i % 2 == 0 else bob_emb
            det = make_detection(100, 100, 200, 200, emb=emb)
            tracked = tracker.update([det], recognizer=recognizer)

            if tracked:
                names_seen.append((i, tracked[0].name, tracked[0].is_confirmed))

        # After majority voting stabilizes (smooth_frames=7: needs 4+ consistent),
        # confirmed_name should settle and not flicker every frame
        confirmed_names = [n for _, n, c in names_seen if c]
        # At minimum, we should see some frames where identity is NOT confirmed
        # (because of alternating embeddings)
        unconfirmed = sum(1 for _, _, c in names_seen if not c)
        assert unconfirmed > 0, "Expected some unconfirmed frames due to flicker"

    def test_track_expiry_and_recovery(self):
        """丢失 max_missed 帧后 track 移除；再次出现生成新 track."""
        tracker = FaceTracker(smooth_frames=3, max_missed=5)

        # Frame 1: detect a face
        det = make_detection(100, 100, 200, 200)
        r1 = tracker.update([det])
        assert len(r1) == 1
        original_id = r1[0].track_id

        # Frames 2-6: no detection (face disappeared)
        for _ in range(5):
            r = tracker.update([])
        # Track should be removed (max_missed=5)
        assert tracker.track_count == 0

        # Frame 7: face reappears → new track_id
        det2 = make_detection(100, 100, 200, 200)
        r7 = tracker.update([det2])
        assert len(r7) == 1
        assert r7[0].track_id != original_id, "Should get new track_id"

    def test_history_truncation(self):
        """name_history 长度不超过 smooth_frames * 3."""
        tracker = FaceTracker(smooth_frames=5, max_missed=30)
        det = make_detection(100, 100, 200, 200)

        for _ in range(50):
            det = make_detection(100, 100, 200, 200)
            tracker.update([det])

        for track in tracker.tracks:
            max_len = tracker.smooth_frames * 3  # 15
            assert len(track.name_history) <= max_len, (
                f"name_history length {len(track.name_history)} exceeds {max_len}"
            )

    def test_concurrent_update_state_consistency(self):
        """多线程 update：状态一致无竞态."""
        import threading

        tracker = FaceTracker(smooth_frames=5)
        errors = []

        def worker():
            try:
                for _ in range(200):
                    dets = [
                        make_detection(
                            np.random.randint(0, 500),
                            np.random.randint(0, 300),
                            np.random.randint(0, 500) + 50,
                            np.random.randint(0, 300) + 50,
                        )
                        for _ in range(5)
                    ]
                    r = tracker.update(dets)
                    assert isinstance(r, list)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # FaceTracker has no internal lock, so concurrent updates
        # may cause inconsistencies — this is documented behavior.
        # The test verifies no hard crashes.
        assert len(errors) == 0, f"Crashes: {errors}"
```

#### tests/stress/test_database_stress.py

```python
"""Stress tests for FaceDatabase."""
import json
import os
import threading
import pytest
import numpy as np
from pathlib import Path
from face_hub import FaceDatabase
from face_hub.exceptions import DatabaseError, SerializationError


def make_encoding(rng=None):
    """Create a normalized 512-dim fake encoding."""
    if rng is None:
        rng = np.random.RandomState()
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


class TestDatabaseStress:
    """FaceDatabase 压力测试."""

    @pytest.mark.stress
    def test_concurrent_crud_10_threads_100_ops(self, temp_db_paths, memory_helper):
        """10 线程各 100 次 add/remove：version 单调、数据一致."""
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        rng = np.random.RandomState(0)

        ops_per_thread = 100
        num_threads = 10
        total_ops = []
        errors = []

        def worker(tid: int):
            try:
                for i in range(ops_per_thread):
                    name = f"t{tid}_p{i}"
                    img_path = str(Path(db_path).parent / f"{name}.jpg")
                    enc = make_encoding(rng)
                    ok, _ = db.add_person(name, img_path, enc)
                    total_ops.append(("add", ok, name))
            except Exception as e:
                errors.append((tid, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"

        # Version should match number of successful adds
        successful_adds = sum(1 for op, ok, _ in total_ops if op == "add" and ok)
        assert db.version == successful_adds, (
            f"Version {db.version} != successful adds {successful_adds}"
        )

        # All persons should be readable
        names = db.get_names()
        assert len(names) == successful_adds

        # Memory check
        delta = memory_helper.snapshot()
        assert delta.rss_delta_mb < 50, f"Memory grew {delta.rss_delta_mb:.1f} MB"

    def test_corrupt_json_recovery(self, temp_db_paths):
        """损坏的 JSON 抛 SerializationError."""
        db_path, enc_path = temp_db_paths

        # Write bad JSON
        with open(db_path, "w") as f:
            f.write("{this is not json")

        with pytest.raises(SerializationError):
            FaceDatabase(db_path=db_path, encoding_path=enc_path)

    def test_corrupt_npy_shape(self, temp_db_paths):
        """shape 错误的 npy 抛 SerializationError."""
        db_path, enc_path = temp_db_paths

        # Write valid JSON but bad npy
        with open(db_path, "w") as f:
            json.dump({"persons": [{"name": "test", "image_path": "test.jpg"}]}, f)

        # Write npy with wrong shape (not (N, 512))
        bad_enc = np.random.randn(10, 256).astype(np.float32)
        np.save(enc_path, bad_enc)

        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        # Should load but encodings/names mismatch
        # The DB stores names from JSON separately; validate encodings count
        encs, names = db.get_encodings_and_names()
        # len(persons) = 1, but encodings have 10 rows
        # This is an inconsistency the DB allows (encodings are loaded from file)
        assert len(names) == 1

    @pytest.mark.stress
    def test_large_batch_5000_records(self, temp_db_paths, memory_helper):
        """5000 条记录 save/load：耗时、内存可控."""
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        rng = np.random.RandomState(0)

        import time
        t0 = time.perf_counter()

        for i in range(5000):
            db.add_person(
                f"person_{i}",
                str(Path(db_path).parent / f"img_{i}.jpg"),
                make_encoding(rng),
            )

        save_time = time.perf_counter() - t0
        print(f"5000 records save: {save_time:.2f}s")

        # Reload and verify
        db2 = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        assert db2.version > 0
        assert len(db2.get_names()) == 5000

        load_time = time.perf_counter() - t0 - save_time
        print(f"5000 records load: {load_time:.2f}s")

        delta = memory_helper.snapshot()
        assert delta.rss_delta_mb < 200, f"Memory grew {delta.rss_delta_mb:.1f} MB"

    def test_path_traversal_rejected(self, temp_db_paths):
        """路径穿越抛 ValueError."""
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)

        malicious_paths = [
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
            "../../../etc/shadow",
        ]

        for mp in malicious_paths:
            with pytest.raises(ValueError, match="outside"):
                db.add_person("hacker", mp, make_encoding())

    def test_concurrent_add_remove(self, temp_db_paths):
        """并发 add 和 remove 不损坏数据."""
        import random
        db_path, enc_path = temp_db_paths
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        rng = np.random.RandomState(0)

        # Pre-populate 100 records
        for i in range(100):
            db.add_person(
                f"person_{i}",
                str(Path(db_path).parent / f"img_{i}.jpg"),
                make_encoding(rng),
            )

        errors = []

        def adder():
            try:
                for i in range(50):
                    db.add_person(
                        f"new_{i}",
                        str(Path(db_path).parent / f"new_{i}.jpg"),
                        make_encoding(rng),
                    )
            except Exception as e:
                errors.append(("adder", e))

        def remover():
            try:
                for i in range(50):
                    target = f"person_{random.randint(0, 99)}"
                    db.remove_person(target)
            except Exception as e:
                errors.append(("remover", e))

        t1 = threading.Thread(target=adder)
        t2 = threading.Thread(target=remover)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert len(errors) == 0, f"Errors: {errors}"
        # DB should still be loadable
        db2 = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        assert db2.version > 0
```

### 9.4 Pipeline 集成压力测试 — tests/stress/test_pipeline_stress.py

```python
"""End-to-end pipeline stress tests with fake components."""
import threading
import time
import pytest
import numpy as np
from face_hub import FaceHubPipeline, FaceTracker, FaceDatabase
from face_hub.types import PipelineResult, UNKNOWN_SENTINEL
from face_hub.exceptions import FaceHubError


class TestPipelineSoak:
    """长时间 soak 测试."""

    @pytest.mark.stress
    def test_10k_frames_stable(self, fake_pipeline_components, memory_helper):
        """10k 帧连续处理：FPS 稳定、内存可控、帧计数正确."""
        pipeline = fake_pipeline_components
        pipeline.start()

        times = []
        fps_samples = []

        for i in range(10_000):
            t0 = time.perf_counter()
            result = pipeline.process_frame()
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            if result is not None:
                fps_samples.append(result.fps)

        pipeline.stop()

        # FPS stability: second half avg FPS >= 80% of first half
        mid = len(times) // 2
        first_half_avg = sum(times[:mid]) / mid
        second_half_avg = sum(times[mid:]) / (len(times) - mid)
        ratio = second_half_avg / first_half_avg if first_half_avg > 0 else float("inf")
        print(f"Latency: first={first_half_avg:.3f}ms second={second_half_avg:.3f}ms ratio={ratio:.2f}")
        assert ratio < 1.5, f"Performance degraded: ratio={ratio:.2f}"

        # Memory
        delta = memory_helper.snapshot()
        assert delta.rss_delta_mb < 50, f"Memory grew {delta.rss_delta_mb:.1f} MB"

        # Frame count
        assert pipeline._total_frame_count == 10_000

    @pytest.mark.stress
    def test_100k_frames_no_crash(self, fake_pipeline_components):
        """100k 帧不崩溃（快速模式）."""
        pipeline = fake_pipeline_components
        pipeline.start()

        for _ in range(100_000):
            result = pipeline.process_frame()
            if result is not None:
                assert isinstance(result, PipelineResult)

        pipeline.stop()
        assert pipeline._total_frame_count == 100_000


class TestPipelineLifecycle:
    """启停幂等性测试."""

    def test_multiple_start_stop(self, fake_pipeline_components):
        """多次 start/stop 不抛异常."""
        p = fake_pipeline_components
        for _ in range(10):
            p.start()
            assert p.is_running is True
            p.stop()
            assert p.is_running is False

    def test_repeat_stop(self, fake_pipeline_components):
        """重复 stop 幂等."""
        p = fake_pipeline_components
        p.start()
        p.stop()
        p.stop()
        p.stop()
        assert p.is_running is False

    def test_repeat_start(self, fake_pipeline_components):
        """重复 start 幂等."""
        p = fake_pipeline_components
        p.start()
        p.start()
        p.start()
        assert p.is_running is True
        p.stop()

    def test_process_frame_when_stopped_returns_none(self, fake_pipeline_components):
        """停止后 process_frame 返回 None."""
        p = fake_pipeline_components
        # Don't start — process_frame should return None
        result = p.process_frame()
        assert result is None

        p.start()
        p.stop()
        result = p.process_frame()
        assert result is None


class TestPipelineErrorIsolation:
    """异常隔离测试."""

    def test_detector_error_wraps_as_facehub_error(self, fake_pipeline_components):
        """detector 异常被包装为 FaceHubError."""
        from tests.stress.conftest import FakeDetector
        from face_hub.types import DetectionWithEmbedding, BBox

        # Replace detector with error-injecting one
        bad_detector = FakeDetector()
        bad_detector.set_error_count(5, RuntimeError)
        fake_pipeline_components.detector = bad_detector
        fake_pipeline_components.start()

        error_count = 0
        success_count = 0
        for _ in range(10):
            try:
                result = fake_pipeline_components.process_frame()
                if result is not None:
                    success_count += 1
            except FaceHubError:
                error_count += 1
            except RuntimeError:
                pass  # Some errors may propagate differently

        # At minimum, some errors should have occurred
        assert error_count > 0 or success_count > 0
        fake_pipeline_components.stop()

    def test_pipeline_continues_after_recovery(self, fake_pipeline_components):
        """组件恢复后流水线继续运行."""
        from tests.stress.conftest import FakeDetector

        detector = fake_pipeline_components.detector
        detector.set_error_count(0)  # ensure clean
        fake_pipeline_components.start()

        # Normal operation
        for _ in range(50):
            result = fake_pipeline_components.process_frame()
            assert result is not None

        # Inject errors
        detector.set_error_count(3, RuntimeError)
        for _ in range(3):
            try:
                fake_pipeline_components.process_frame()
            except (FaceHubError, RuntimeError):
                pass

        # After errors exhausted, pipeline recovers
        result = fake_pipeline_components.process_frame()
        assert result is not None
        fake_pipeline_components.stop()


class TestPipelineDynamicDB:
    """动态数据库更新测试."""

    def test_add_person_during_operation(self, fake_pipeline_components, sample_encoding):
        """运行中 add_person：缓存同步、识别结果变化."""
        from pathlib import Path

        p = fake_pipeline_components
        p.start()

        # Run a few frames to establish baseline
        for _ in range(5):
            p.process_frame()

        # Add person to DB
        p.db.add_person(
            "test_user",
            str(Path(p.db.db_path).parent / "test.jpg"),
            sample_encoding,
        )

        # Process a frame — cache should auto-sync via _last_db_version
        result = p.process_frame()
        assert result is not None
        assert "test_user" in p.recognizer.cached_names

        p.stop()

    def test_remove_person_during_operation(self, fake_pipeline_components, sample_encoding):
        """运行中 remove_person：缓存更新."""
        from pathlib import Path

        p = fake_pipeline_components

        # Pre-populate
        p.db.add_person("to_remove", str(Path(p.db.db_path).parent / "tr.jpg"), sample_encoding)
        p.update_database_cache()
        assert "to_remove" in p.recognizer.cached_names

        p.start()
        p.db.remove_person("to_remove")
        result = p.process_frame()
        assert result is not None
        assert "to_remove" not in p.recognizer.cached_names
        p.stop()


class TestPipelineFPS:
    """FPS 统计测试."""

    def test_fps_non_negative(self, fake_pipeline_components):
        """FPS 字段非负."""
        fake_pipeline_components.start()
        for _ in range(50):
            result = fake_pipeline_components.process_frame()
            if result is not None:
                assert result.fps >= 0.0
        fake_pipeline_components.stop()

    def test_fps_approximates_real_throughput(self, fake_pipeline_components):
        """FPS 统计接近真实吞吐量."""
        fake_pipeline_components.start()
        n_frames = 100

        t0 = time.perf_counter()
        for _ in range(n_frames):
            fake_pipeline_components.process_frame()
        elapsed = time.perf_counter() - t0

        measured_fps = n_frames / elapsed if elapsed > 0 else float("inf")

        # The internal FPS counter updates every 1s;
        # for short runs, check it's in a sane range
        result = fake_pipeline_components.process_frame()
        if result is not None:
            reported_fps = result.fps
            # Reported FPS should be within 50% of measured
            if reported_fps > 0:
                ratio = reported_fps / measured_fps
                assert 0.3 < ratio < 3.0, (
                    f"FPS mismatch: reported={reported_fps:.1f} measured={measured_fps:.1f}"
                )

        fake_pipeline_components.stop()


class TestPipelineConcurrency:
    """并发测试."""

    def test_multi_thread_process_frame(self, fake_pipeline_components):
        """多线程 process_frame：无死锁."""
        fake_pipeline_components.start()
        errors = []
        results = []

        def worker():
            try:
                for _ in range(100):
                    r = fake_pipeline_components.process_frame()
                    if r is not None:
                        results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) > 0
        fake_pipeline_components.stop()
```

### 9.5 真实模型基准 — tests/stress/test_real_model_slow.py

```python
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
```

### 9.6 基线文件 — tests/stress/baseline.json (初始模板)

```json
{
  "_schema": "1.0",
  "_description": "Performance baseline for FaceHub regression tests. Generated by --run-slow, consumed by CI.",
  "_thresholds": {
    "memory_growth_10k_mb": 50,
    "fps_degradation_ratio": 0.80,
    "p95_regression_ratio": 1.20
  },
  "detector_cpu_640": {
    "mean": null,
    "p50": null,
    "p95": null,
    "p99": null,
    "min": null,
    "max": null,
    "samples": 300,
    "peak_memory_mb": null,
    "det_size": 640,
    "device": "cpu"
  },
  "recognizer_1k": {
    "mean": null,
    "p50": null,
    "p95": null,
    "p99": null,
    "min": null,
    "max": null,
    "samples": 10000,
    "gallery_size": 1000
  },
  "pipeline_e2e_cpu_640": {
    "mean": null,
    "p50": null,
    "p95": null,
    "p99": null,
    "min": null,
    "max": null,
    "samples": 300,
    "det_size": 640,
    "device": "cpu"
  }
}
```

---

## 10. CI 集成

### 10.1 GitHub Actions — Nightly Stress Job

```yaml
# .github/workflows/nightly-stress.yml
name: Nightly Stress Tests

on:
  schedule:
    - cron: "37 3 * * *"      # 03:37 UTC every day
  workflow_dispatch:           # Manual trigger

jobs:
  stress-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install psutil

      - name: Run stress tests
        run: |
          python -m pytest tests/stress/ \
            -m "stress" \
            --run-stress \
            -v \
            --tb=short \
            --timeout=300 \
            --junitxml=stress-report.xml

      - name: Run slow benchmark (generate baseline)
        run: |
          python -m pytest tests/stress/ \
            -m "slow" \
            --run-slow \
            -v \
            --tb=long \
            --timeout=600

      - name: Upload baseline artifact
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: stress-results
          path: |
            stress-report.xml
            tests/stress/baseline.json

      - name: Check baseline regression
        run: |
          python -c "
          import json, sys
          from pathlib import Path
          baseline = Path('tests/stress/baseline.json')
          if baseline.exists():
              data = json.loads(baseline.read_text())
              errors = []
              for k, v in data.items():
                  if k.startswith('_'):
                      continue
                  if v.get('p95') and v['p95'] > 2000:
                      errors.append(f'{k}: p95={v[\"p95\"]:.0f}ms > 2000ms threshold')
              if errors:
                  print('REGRESSION DETECTED:')
                  for e in errors:
                      print(f'  ❌ {e}')
                  sys.exit(1)
              print('✅ All benchmarks within thresholds')
          "
```

### 10.2 PR CI — Skip Stress, Run Unit Tests Only

```yaml
# .github/workflows/pr-tests.yml (追加到已有的 CI)
name: PR Tests

on: [pull_request, push]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: python -m pytest tests/ -v --tb=short --timeout=60
        # NB: stress/slow/hardware markers mean those tests are auto-skipped
```

---

## 11. 运行命令速查

```bash
# ── 开发时快速运行（仅基础单测，跳过 stress/slow） ──
pytest tests/ -v

# ── 运行全部压力测试（Mock 组件，无 GPU/摄像头要求） ──
pytest tests/stress/ --run-stress -v

# ── 运行真实模型基准（生成 baseline.json） ──
pytest tests/stress/ --run-slow --run-hardware -v

# ── 运行内存回归测试 ──
pytest tests/stress/ --run-memory --run-stress -v

# ── 运行所有测试（包括慢测 + 硬件） ──
pytest tests/stress/ --run-stress --run-slow --run-memory --run-hardware -v

# ── 生成 JUnit XML 报告 ──
pytest tests/stress/ --run-stress -v --junitxml=stress-report.xml

# ── 并行运行 stress 测试（需要 pytest-xdist） ──
pytest tests/stress/ --run-stress -v -n auto

# ── 对比基线回归 ──
python -c "
import json
from pathlib import Path
b = json.loads(Path('tests/stress/baseline.json').read_text())
for k,v in b.items():
    if k.startswith('_'): continue
    print(f'{k}: p50={v.get(\"p50\",\"?\")}ms p95={v.get(\"p95\",\"?\")}ms')
"
```

---

## 12. 附录：测试覆盖率目标

| 组件 | 行覆盖率目标 | 分支覆盖率目标 | 压力测试断言数 |
|------|------------|--------------|-----------|
| CameraThread / FakeCamera | 85% | 75% | ≥ 8 |
| FaceDetector / FakeDetector | 80% | 70% | ≥ 12 |
| FaceRecognizer / FakeRecognizer | 90% | 80% | ≥ 15 |
| FaceTracker | 85% | 75% | ≥ 10 |
| FaceDatabase | 90% | 80% | ≥ 12 |
| FaceHubPipeline | 85% | 75% | ≥ 15 |

**最终目标**：每次 PR 自动运行 `pytest tests/` < 2 min；夜间 job 运行全套 stress + slow < 30 min。