# FaceVision v1.0 — 新仓库实施计划

> **目标：** 在新仓库中构建纯 Python 库。
> - **新仓库名：** `FaceVision`（也是主包名）
> - **原则：** 保留所有核心算法，移除 GUI 依赖，所有配置通过 `__init__` 参数传入，`print`→`logging`，返回值用 `dataclass`

---

## 一、新仓库目录结构

```
FaceVision/                              # Git 仓库根目录（项目名）
├── pyproject.toml                       # [新写] name = "facevision"
├── README.md                            # [新写] 中英双语说明
├── LICENSE                              # [新写] MIT
│
├── .github/
│   └── workflows/
│       └── publish.yml                  # [新写] 三平台 CI + 自动发布
│
├── face_hub/                         # Python 包（小写+下划线）
│   ├── __init__.py                      #   公开 API，__version__ = "1.0.0"
│   ├── types.py                         #   [新写] dataclass 类型 + UNKNOWN_SENTINEL
│   ├── exceptions.py                    #   [新写] 自定义异常层级
│   ├── detector_protocol.py             #   [新写] 检测器抽象接口
│   ├── pipeline.py                      #   [新写] 整合全部组件
│   │
│   └── engine/                          #   【原来文件】核心算法引擎
│       ├── __init__.py
│       ├── config.py
│       ├── camera.py
│       ├── face_detector.py
│       ├── face_recognizer.py
│       ├── face_tracker.py
│       └── face_database.py
│
├── tests/                               # [新写] 全部测试
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_types.py
│   ├── test_exceptions.py
│   ├── test_config.py
│   ├── test_face_database.py
│   ├── test_face_recognizer.py
│   ├── test_face_tracker.py
│   ├── test_face_detector.py
│   └── test_pipeline.py
│
└── docs/                                # [新写] 中英文 API 文档
    ├── index.md
    ├── mkdocs.yml
    ├── zh/ (index, quickstart, installation, api/*, examples/*)
    └── en/ (同上结构)
```

**包内导入关系（单包，无冲突）：**
- `face_hub/engine/face_detector.py` → `from face_hub.types import ...` / `from face_hub.exceptions import ...`
- `face_hub/engine/face_recognizer.py` → `from face_hub.types import UNKNOWN_SENTINEL`
- `face_hub/engine/face_tracker.py` → `from face_hub.types import UNKNOWN_SENTINEL, TrackedFace, BBox`
- `face_hub/pipeline.py` → `from face_hub.engine.face_detector import FaceDetector`
- `face_hub/detector_protocol.py` → `from face_hub.types import ...`
- `face_hub/__init__.py` → 从 `face_hub.*` 和 `face_hub.engine.*` 收集全部 API
```

**包结构（单一顶层包 `FaceVision`，`engine/` 为算法子包）：**
- `face_hub/` — 主包，用户唯一入口。`__init__.py` re-export 全部公开 API
- `face_hub/engine/` — 算法引擎子包，包含全部核心模块

---

## 二、`face_hub/engine/` — 核心算法引擎

内部导入关系：
  - `face_hub/engine/face_detector.py` → `from face_hub.types import ...` / `from face_hub.exceptions import ...`
  - `face_hub/engine/face_recognizer.py` → `from face_hub.types import UNKNOWN_SENTINEL`
  - `face_hub/engine/face_tracker.py` → `from face_hub.types import UNKNOWN_SENTINEL, TrackedFace, BBox`
  - `face_hub/pipeline.py` → `from face_hub.engine.face_detector import FaceDetector`
  - `face_hub/detector_protocol.py` → 抽象接口，用户可实现以接入自己的模型
  - `face_hub/__init__.py` → 从 `face_hub.*` 和 `face_hub.engine.*` 收集全部 API

---

### 文件 3-0：`face_hub/engine/__init__.py`

```python
"""face_hub engine — core algorithms for detection, recognition, tracking."""
```

### 文件 3-1：`face_hub/engine/config.py`

**改动：** 删除 `load_settings()` / `save_settings()` / `APP_SETTINGS`，仅保留常量。

```python
"""
FaceVision default configuration constants.

The library never reads any config file. Callers should:
  1. Read their own config file (JSON / YAML / TOML etc.)
  2. Merge with DEFAULT_SETTINGS as fallback
  3. Pass final values to each component's __init__
"""

from copy import deepcopy


DEFAULT_SETTINGS = {
    "device": "cuda",              # cpu / cuda (cuda auto-detects CUDA→DML→CPU)
    "confidence": 0.50,            # detection confidence (RetinaFace: ≥0.45)
    "tolerance": 0.45,             # cosine similarity: 0.40=strict, 0.45=recommended, 0.50=loose
    "cam_width": 640,
    "cam_height": 360,
    "cam_fps": 30,
    "proc_fps": 30,                # ML processing FPS cap (0=unlimited)
    "det_size": 640,               # 320=fast, 480=balanced, 640=accurate
    "track_smooth": 5,             # 3=fast, 5=recommended, 8=stable
    "min_face_size": 60,
    "quality_filter": True,
}


def get_default_settings() -> dict:
    """Return a deep copy of default settings (safe to mutate)."""
    return deepcopy(DEFAULT_SETTINGS)
```

---

### 文件 3-2：`face_hub/engine/camera.py`

**改动：**

| 行 | 原代码 | 改为 |
|----|--------|------|
| L10 | 无 `import logging` | `import logging` |
| L13 | 无 logger | `logger = logging.getLogger("face_hub.camera")` |
| L14 | `os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"` | 加 `sys.platform` 守卫，仅 Windows 设置 |
| L19 | `cv2.CAP_DSHOW` 硬编码 | `_get_backend()` 按平台选择 |
| L52 | `raise RuntimeError(...)` | `from face_hub.exceptions import CameraError` + `raise CameraError(...)` |
| L56 | `print(f"[Camera] 实际分辨率:...")` | `logger.info("Actual resolution: %dx%d", int(actual_w), int(actual_h))` |
| L122 | `cv2.CAP_DSHOW` 硬编码 | `_get_backend()` |
| 新增 | — | `_get_backend()` 静态方法 |

**其余代码逐字保留。** 完整代码如下：

```python
"""
Camera capture thread module.
Runs camera capture in a dedicated thread, provides thread-safe latest-frame access.

Platform backends:
  - Windows:  DShow (cv2.CAP_DSHOW)
  - macOS:    AVFoundation (cv2.CAP_AVFOUNDATION)
  - Linux:    V4L2 (cv2.CAP_V4L2)
"""

import cv2
import threading
import time
import os
import sys
import logging
from collections import deque

# Suppress DShow hardware transform warnings on Windows
if sys.platform == "win32":
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

logger = logging.getLogger("face_hub.camera")


class CameraThread:
    """Camera capture thread — pure acquisition, no processing."""

    def __init__(self, camera_id=0, width=640, height=360, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.cap = None
        self.running = False
        self._frame_buffer = deque(maxlen=1)
        self.lock = threading.Lock()
        self.thread = None
        self._actual_fps = 0.0
        self._fps_lock = threading.Lock()
        self._frame_available = threading.Event()

    # ── Platform-aware backend selection ──────────────────────

    @staticmethod
    def _get_backend():
        """Return the optimal OpenCV backend for the current platform."""
        if sys.platform == "win32":
            return cv2.CAP_DSHOW
        elif sys.platform == "darwin":
            # AVFoundation is default on macOS, but explicit avoids warnings
            return cv2.CAP_AVFOUNDATION
        else:
            # Linux: V4L2 is the standard
            return cv2.CAP_V4L2

    # ── Public API ────────────────────────────────────────────

    @property
    def actual_fps(self):
        with self._fps_lock:
            return self._actual_fps

    def start(self):
        """Start camera capture."""
        if self.running:
            return

        self.cap = cv2.VideoCapture(self.camera_id, self._get_backend())
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            from face_hub.exceptions import CameraError
            raise CameraError(f"Cannot open camera (ID={self.camera_id})")

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info("Actual resolution: %dx%d", int(actual_w), int(actual_h))

        self.running = True
        self._frame_available.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop camera capture."""
        self.running = False
        self._frame_available.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        self.cap = None

    def get_frame(self, timeout=0.05, copy=True):
        """
        Thread-safe latest frame retrieval.

        Args:
            timeout: Max seconds to wait for a new frame.
            copy: If True (default), returns a safe copy. Set False for
                  zero-copy mode — caller must NOT mutate the returned frame.

        Returns:
            Frame as numpy array, or None if timeout expired.
        """
        if self._frame_available.wait(timeout):
            with self.lock:
                if len(self._frame_buffer) > 0:
                    frame = self._frame_buffer[0]
                    self._frame_available.clear()
                    return frame.copy() if copy else frame
        return None

    def _loop(self):
        """Main capture loop."""
        interval = 1.0 / max(self.target_fps, 1)
        last_time = time.time()
        fps_counter = 0
        fps_timer = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)

            with self.lock:
                self._frame_buffer.append(frame)
                self._frame_available.set()

            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                with self._fps_lock:
                    self._actual_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            elapsed = now - last_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
            last_time = time.time()

    @staticmethod
    def list_cameras(max_test=5):
        """List available camera indices (suppresses DShow warnings)."""
        try:
            cv_log_level = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LEVEL_ERROR)
        except Exception:
            cv_log_level = None

        available = []
        for i in range(max_test):
            try:
                cap = cv2.VideoCapture(i, CameraThread._get_backend())
                if cap.isOpened():
                    available.append(i)
                    cap.release()
            except Exception:
                pass

        if cv_log_level is not None:
            try:
                cv2.utils.logging.setLogLevel(cv_log_level)
            except Exception:
                pass
        return available
```

---

### 文件 3-3：`face_hub/engine/face_detector.py`

**改动：** 增加 `_get_backend()` 方法，原 `_resolve_providers` 保持不变（已自动按平台探测 CUDA/DirectML/CPU）。

> **v1.0 多平台 GPU 支持：**
> ```
> 平台        GPU 后端              ONNX Provider
> ──────────────────────────────────────────────────
> Windows     DirectML              DmlExecutionProvider
> Windows     NVIDIA CUDA           CUDAExecutionProvider
> Linux       NVIDIA CUDA           CUDAExecutionProvider
> macOS       ❌ v1.0 仅 CPU         (CoreML 延后至 v1.1)
> 全部        无 GPU / 回退          CPUExecutionProvider
> ```
> `_resolve_providers()` 自动按 `ort.get_available_providers()` 探测可用后端，
> 优先级：CUDA → DirectML → CPU。macOS 上 `device="cuda"` 等同于 `device="cpu"`。
>
> **macOS CoreML 延后至 v1.1 的原因：**
> - ONNX Runtime CoreML 默认 `NeuralNetwork` 格式算子覆盖率仅 ~9%，实测比纯 CPU 慢 41%
> - 需用 `MLProgram` 格式 + `ModelCacheDirectory` 缓存 + 逐算子调参才能跑出收益
> - M 系列芯片的 AMX 协处理器 + CPU 推理已经足够快，v1.0 没必要引入 CoreML 的不确定性
> - v1.1 将新增 `device="coreml"` 选项，届时自动启用上述优化

#### A. print → logger（共 15 处）

| # | 原 print 语句 | 替换为 | 级别 |
|---|---|---|---|
| 1 | `print("[FaceDetector] [WARN] Inference falling back to CPU mode")` | `logger.warning("Inference falling back to CPU mode")` | WARNING |
| 2 | `print("[FaceDetector] [OK] Model loaded (device=...")` | `logger.info("Model loaded (device=%s, det_size=%d)", gpu_name, self.det_size)` | INFO |
| 3 | `print("[FaceDetector] [WARN] GPU load failed: {e}")` | `logger.warning("GPU load failed: %s", e)` | WARNING |
| 4 | `print("[FaceDetector] → 回退到 CPU…")` | `logger.info("Falling back to CPU…")` | INFO |
| 5 | `print("[FaceDetector] [OK] Warm-up complete...")` | `logger.info("Warm-up complete (detected %d fake faces)", len(faces))` | INFO |
| 6 | `print("[FaceDetector] [WARN] Warm-up failed...")` | `logger.warning("Warm-up failed (size=%d): %s", self.det_size, err_msg)` | WARNING |
| 7 | `print("[FaceDetector] → DirectML 不兼容 det_size=...")` | `logger.warning("DirectML incompatible with det_size=%d, retrying with 640", self.det_size)` | WARNING |
| 8 | `print("[FaceDetector] [OK] Auto-adjusted to det_size=640")` | `logger.info("Auto-adjusted to det_size=640 (DirectML compat)")` | INFO |
| 9 | `print("[FaceDetector] → 640 重试也失败: {inner}")` | `logger.error("640 retry also failed: %s", inner)` | ERROR |
| 10 | `print("[FaceDetector] → DirectML 推理失败，回退到 CPU…")` | `logger.warning("DirectML inference failed, falling back to CPU…")` | WARNING |
| 11 | `print("[FaceDetector] → 推理异常，尝试切换到 CPU…")` | `logger.warning("Inference anomaly, attempting CPU switch…")` | WARNING |
| 12 | `print("[FaceDetector] [WARN] GPU inference error...")` | `logger.warning("GPU inference error (%s: %s), switching to CPU…", err_type, e)` | WARNING |
| 13 | `print("[FaceDetector] [ERROR] CPU fallback also failed: {e2}")` | `logger.error("CPU fallback also failed: %s", e2)` | ERROR |
| 14 | `print("[FaceDetector] detect: ...全部低于阈值...")` | `logger.debug("detect: %d faces all below threshold %.2f", len(faces), self.confidence)` | DEBUG |
| 15 | `print("[FaceDetector] detect_with_emb: ...全部低于阈值...")` | `logger.debug("detect_with_emb: %d faces all below threshold %.2f", len(faces), self.confidence)` | DEBUG |

#### B. 新增导入

```python
# 在现有 import 之后追加
from face_hub.types import DetectionResult, DetectionWithEmbedding, BBox
from face_hub.exceptions import ModelLoadError, InferenceError
```

#### C. `__init__` — 新增 `"auto"` 默认值

```python
def __init__(self, confidence=0.50, device="auto", det_size=640,
             quality_filter=True, min_face_size=80):
    if device == "auto":
        device = "cuda"
    # ... 其余不变 ...
```

#### D. `detect()` 返回 `List[DetectionResult]`

```python
def detect(self, frame) -> List[DetectionResult]:
    # ... 推理逻辑不变 ...
    results = []
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        conf = float(face.det_score) if hasattr(face, 'det_score') else 0.95
        if conf >= self.confidence:
            results.append(DetectionResult(
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=conf,
            ))
    return results
```

#### E. `detect_with_embeddings()` 返回 `List[DetectionWithEmbedding]`

```python
def detect_with_embeddings(self, frame) -> List[DetectionWithEmbedding]:
    # ... 推理逻辑不变 ...
    results = []
    for face in faces:
        # ... 质量过滤逻辑不变 ...
        results.append(DetectionWithEmbedding(
            bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            confidence=det_conf,
            embedding=emb,
            quality_pass=quality_pass,
        ))
    return results
```

#### F. `_load_model()` 异常包装

```python
def _load_model(self, fallback_to_cpu=False):
    try:
        self.app, gpu_name = self._create_app(self.det_size)
        logger.info("Model loaded (device=%s, det_size=%d)", gpu_name, self.det_size)
    except ImportError as e:
        raise ModelLoadError("insightface is not installed") from e
    except Exception as e:
        if not fallback_to_cpu and self.device == "cuda":
            logger.warning("GPU load failed: %s", e)
            logger.info("Falling back to CPU…")
            self.device = "cpu"
            self._load_model(fallback_to_cpu=True)
        else:
            raise ModelLoadError(f"Failed to load FaceDetector model: {e}") from e
```

#### G. `_handle_inference_error()` 异常包装

```python
def _handle_inference_error(self, e, frame):
    self._inference_error_count += 1
    err_type = type(e).__name__

    if self.device == "cuda":
        logger.warning("GPU inference error (%s: %s), switching to CPU…", err_type, e)
        try:
            self.device = "cpu"
            self._load_model(fallback_to_cpu=True)
            with self._lock:
                return self.app.get(frame)
        except Exception as e2:
            raise InferenceError(f"Inference failed on both GPU and CPU: {e2}") from e2
    raise InferenceError(f"Inference error on {self.device}: {e}") from e
```

#### H. 保持不变的方法

以下方法逐字复制，不做改动：
- `_resolve_providers(force_cpu=False)` — 仅 `print` 改 `logger`
- `_create_app(det_size)` — 完全不变
- `_warmup()` — 仅 `print` 改 `logger`
- `_face_quality(face_roi)` — 完全不变（静态方法）
- `_face_roi(frame, bbox, expand=0.20)` — 完全不变（接受 tuple bbox）
- `_run_inference(frame)` — 完全不变
- `extract_face_roi(frame, face_rect)` — 完全不变
- `reload_model(det_size=None)` — 完全不变

---

### 文件 3-4：`face_hub/engine/face_recognizer.py`

**改动：仅 1 行**

```python
# 第 10 行：
# 原: from i18n import UNKNOWN_SENTINEL
# 新:
from face_hub.types import UNKNOWN_SENTINEL
```

**其余全部逐字保留。** logger 名保持 `"face_hub.recognizer"`。完整代码略（复制原文件改一行即可）。

---

### 文件 3-5：`face_hub/engine/face_tracker.py`

**改动：两处**

#### A. 导入（文件顶部）

```python
# 原:
from i18n import UNKNOWN_SENTINEL

# 新:
import numpy as np
import logging

from face_hub.types import UNKNOWN_SENTINEL, TrackedFace, BBox

logger = logging.getLogger("face_hub.tracker")
```

#### B. `update()` 返回值构建（原 L252-267）

```python
# 原:
results = []
for track in self.tracks:
    display_name, display_conf, is_confirmed = track.resolve_identity(self.smooth_frames)
    results.append({
        'bbox': track.bbox,
        'name': display_name,
        'conf': display_conf,
        'det_conf': 0.0,
        'track_id': track.id,
        'is_confirmed': is_confirmed,
        'quality_pass': True,
    })
return results

# 新:
results = []
for track in self.tracks:
    display_name, display_conf, is_confirmed = track.resolve_identity(self.smooth_frames)
    x1, y1, x2, y2 = track.bbox
    results.append(TrackedFace(
        track_id=track.id,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        name=display_name,
        confidence=display_conf,
        det_confidence=0.0,
        is_confirmed=is_confirmed,
        quality_pass=True,
    ))
return results
```

**其余全部逐字保留：**
- `_iou(boxA, boxB)` — 不变
- `FaceTrack` 全部（`__slots__`, `__init__`, `update`, `mark_missed`, `is_stale`, `_majority_vote`, `resolve_identity`）— 不变
- `FaceTracker.__init__`, IoU 匹配逻辑（L175-250）, `reset`, `track_count` — 不变

---

### 文件 3-6：`face_hub/engine/face_database.py`

**改动：** 新增导入 + `save()`/`load()` 加异常包装。

#### A. 新增导入

```python
import logging
from face_hub.exceptions import DatabaseError

logger = logging.getLogger("face_hub.database")
```

#### B. `save()`

```python
def save(self):
    try:
        data = {"persons": self.persons}
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        with open(self.encoding_path, "wb") as f:
            pickle.dump(self.encodings, f)
    except (IOError, OSError, pickle.PickleError) as e:
        raise DatabaseError(f"Failed to save database: {e}") from e
```

#### C. `load()`

```python
def load(self):
    try:
        if self.db_path.exists():
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.persons = data.get("persons", [])
        if self.encoding_path.exists():
            with open(self.encoding_path, "rb") as f:
                self.encodings = pickle.load(f)
        self._version += 1
    except (json.JSONDecodeError, IOError, pickle.UnpicklingError) as e:
        raise DatabaseError(f"Failed to load database: {e}") from e
```

**其余方法逐字保留：** `add_person`, `remove_person`, `remove_persons`, `get_names`, `get_encodings_and_names`, `clear`。

---

## 三、`face_hub/` — 新写文件（从零编写）

---

### 文件 4-1：`face_hub/types.py`

```python
"""
FaceVision type definitions.
All public API return types are dataclasses defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ── Global constants ──────────────────────────────────────────────

UNKNOWN_SENTINEL = "unknown"
"""
Stable sentinel for "not recognized as any registered person".
Import via: from face_hub.types import UNKNOWN_SENTINEL
"""


# ── Geometry ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class BBox:
    """Immutable bounding box (hashable)."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_tuple(self) -> tuple:
        """Convert to (x1, y1, x2, y2) for OpenCV functions."""
        return (self.x1, self.y1, self.x2, self.y2)


# ── Detection results ─────────────────────────────────────────────

@dataclass
class DetectionResult:
    """
    Returned by FaceDetector.detect().
    """
    bbox: BBox
    confidence: float          # detection confidence 0.0~1.0

    @classmethod
    def from_tuple(cls, t: tuple) -> "DetectionResult":
        return cls(bbox=BBox(x1=t[0], y1=t[1], x2=t[2], y2=t[3]), confidence=t[4])


@dataclass
class DetectionWithEmbedding:
    """
    Returned by FaceDetector.detect_with_embeddings().
    embedding is None if feature extraction failed (rare).
    """
    bbox: BBox
    confidence: float                               # detection confidence 0.0~1.0
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    quality_pass: bool = True                       # quality filter passed

    @classmethod
    def from_tuple(cls, t: tuple) -> "DetectionWithEmbedding":
        return cls(
            bbox=BBox(x1=t[0], y1=t[1], x2=t[2], y2=t[3]),
            confidence=t[4],
            embedding=t[5],
            quality_pass=t[6],
        )

    @property
    def has_embedding(self) -> bool:
        return self.embedding is not None


# ── Tracking results ──────────────────────────────────────────────

@dataclass
class TrackedFace:
    """
    Returned by FaceTracker.update().
    """
    track_id: int
    bbox: BBox
    name: str                       # recognized name, or UNKNOWN_SENTINEL
    confidence: float               # cosine similarity 0.0~1.0
    det_confidence: float           # detection confidence
    is_confirmed: bool              # identity confirmed via majority vote
    quality_pass: bool

    @property
    def is_known(self) -> bool:
        return self.name != UNKNOWN_SENTINEL


# ── Pipeline result ───────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Returned by FaceHubPipeline.process_frame().
    """
    frame: np.ndarray = field(repr=False)
    raw_detections: List[DetectionWithEmbedding] = field(default_factory=list)
    tracked_faces: List[TrackedFace] = field(default_factory=list)
    fps: float = 0.0

    @property
    def known_faces(self) -> List[TrackedFace]:
        return [t for t in self.tracked_faces if t.is_known]

    @property
    def unknown_count(self) -> int:
        return sum(1 for t in self.tracked_faces if not t.is_known)

    @property
    def total_faces(self) -> int:
        return len(self.tracked_faces)
```

---

### 文件 4-2：`face_hub/exceptions.py`

```python
"""
FaceVision exception hierarchy.
All library exceptions inherit from FaceHubError.
"""


class FaceHubError(Exception):
    """Base exception for all FaceVision errors."""
    pass


class ModelLoadError(FaceHubError):
    """Model loading failed (insightface not installed / no ONNX provider / corrupt model)."""
    pass


class InferenceError(FaceHubError):
    """ML inference runtime error (GPU crash + CPU fallback also failed)."""
    pass


class CameraError(FaceHubError):
    """Camera error (not connected / in use / unsupported resolution)."""
    pass


class DatabaseError(FaceHubError):
    """Database read/write error (JSON parse / pickle corrupt / disk full / permission)."""
    pass


class RecognitionError(FaceHubError):
    """Recognition matching error (encoding dimension mismatch / empty cache)."""
    pass
```

---

---

### 文件 4-3：`face_hub/detector_protocol.py`（新写）

```python
"""
Detector protocol — abstract interface for face detection + embedding.

Built-in implementation: face_hub.engine.face_detector.FaceDetector (insightface).
Users can implement this protocol to plug in custom models (YOLO, MediaPipe,
commercial SDK, etc.) without modifying the rest of the pipeline.
"""

from __future__ import annotations

from typing import Protocol, List, runtime_checkable
import numpy as np

from face_hub.types import DetectionResult, DetectionWithEmbedding


@runtime_checkable
class DetectorProtocol(Protocol):
    """
    Protocol for face detection and embedding extraction.

    Any object implementing these three methods can be passed to
    FaceHubPipeline as the detector.

    Minimal implementation: just implement detect_with_embeddings().
    detect() and reload_model() have default no-op fallbacks.
    """

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect all faces in a BGR frame.

        Args:
            frame: BGR image as (H, W, 3) numpy array.

        Returns:
            List of DetectionResult, sorted by confidence descending.
        """
        ...

    def detect_with_embeddings(
        self, frame: np.ndarray
    ) -> List[DetectionWithEmbedding]:
        """
        Detect faces and extract feature embeddings in one pass.

        This is the primary method called by the pipeline. Implementations
        should return embeddings that are L2-normalized 512-dim float32 vectors
        (or set the confidence threshold in FaceRecognizer accordingly).

        Args:
            frame: BGR image as (H, W, 3) numpy array.

        Returns:
            List of DetectionWithEmbedding, sorted by confidence descending.
        """
        ...

    def reload_model(self, det_size: int = None) -> None:
        """
        Reload the underlying model with new parameters (optional).

        Called when the user changes detection resolution at runtime.
        Default no-op is acceptable if the implementation doesn't support
        hot-reload.
        """
        pass
```

> **使用示例（用户自定义检测器）：**
> ```python
> from face_hub import DetectorProtocol, DetectionWithEmbedding, BBox
>
> class MyYoloDetector:
>     """Use YOLOv8-face + a custom recognition model."""
>
>     def detect_with_embeddings(self, frame):
>         # 1. YOLO face detection
>         boxes = self.yolo_model(frame)
>         # 2. Extract embeddings with your own model
>         results = []
>         for box in boxes:
>             roi = frame[box.y1:box.y2, box.x1:box.x2]
>             emb = self.my_embedder(roi)
>             results.append(DetectionWithEmbedding(
>                 bbox=BBox(box.x1, box.y1, box.x2, box.y2),
>                 confidence=box.conf,
>                 embedding=emb,
>                 quality_pass=True,
>             ))
>         return results
>
>     def detect(self, frame):
>         # Optional: detection-only mode
>         boxes = self.yolo_model(frame)
>         return [DetectionResult(bbox=..., confidence=...) for ...]
>
> # Plug into pipeline:
> pipeline = FaceHubPipeline(
>     camera, MyYoloDetector(), recognizer, tracker, db
> )
> ```
>
> **约束：** `DetectorProtocol` 使用 `typing.Protocol`（结构化子类型），
> 任何实现了 `detect_with_embeddings(frame) -> List[DetectionWithEmbedding]`
> 的对象都自动满足协议，无需显式继承。


### 文件 4-4：`face_hub/pipeline.py`（新写）

```python

from __future__ import annotations

import time
import threading
import logging
from typing import Optional

import numpy as np

from typing import List
from face_hub.types import PipelineResult, DetectionResult, DetectionWithEmbedding
from face_hub.exceptions import FaceHubError
from face_hub.detector_protocol import DetectorProtocol

from face_hub.engine.face_recognizer import FaceRecognizer
from face_hub.engine.face_tracker import FaceTracker
from face_hub.engine.face_database import FaceDatabase
from face_hub.engine.camera import CameraThread

logger = logging.getLogger("face_hub.pipeline")


class FaceHubPipeline:
    """
    Full face recognition pipeline.

    Usage:
        pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
        pipeline.start()

        while True:
            result = pipeline.process_frame()
            if result is None:
                continue
            for face in result.known_faces:
                print(f"{face.name} ({face.confidence:.2%})")

        pipeline.stop()
    """

    def __init__(
        self,
        camera: CameraThread,
        detector: DetectorProtocol,  # ← 接受任何满足协议的对象
        recognizer: FaceRecognizer,
        tracker: FaceTracker,
        db: FaceDatabase,
    ):
        self.camera = camera
        self.detector = detector
        self.recognizer = recognizer
        self.tracker = tracker
        self.db = db

        self._running = False
        self._lock = threading.Lock()
        self._frame_count = 0
        self._debug_frame_count = 0     # independent counter for debug logging
        self._fps_timer = time.time()
        self._current_fps = 0.0

    # ── Lifecycle ────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """Start pipeline (starts camera if needed)."""
        if self._running:
            return
        if not self.camera.running:
            self.camera.start()
        self._running = True
        self._fps_timer = time.time()
        self._frame_count = 0
        logger.info("Pipeline started")

    def stop(self):
        """Stop pipeline (stops camera)."""
        self._running = False
        if self.camera.running:
            self.camera.stop()
        logger.info("Pipeline stopped")

    def reset_tracker(self):
        """Reset face tracker (e.g. after settings change)."""
        self.tracker.reset()
        logger.info("Tracker reset")

    # ── Database cache ───────────────────────────────────────────

    def update_database_cache(self) -> bool:
        """
        Sync recognizer encoding cache with database.
        Returns True if cache was rebuilt.
        """
        known_encodings, known_names = self.db.get_encodings_and_names()
        return self.recognizer.update_cache(
            known_encodings, known_names, self.db.version
        )

    # ── Core processing ──────────────────────────────────────────

    def process_frame(
        self, frame: Optional[np.ndarray] = None
    ) -> Optional[PipelineResult]:
        """
        Process one frame through the full pipeline.

        Args:
            frame: BGR numpy array. If None, fetches from camera.

        Returns:
            PipelineResult, or None if no frame available.

        Thread-safe via internal lock.
        """
        with self._lock:
            return self._process_frame_impl(frame)

    def _process_frame_impl(self, frame) -> Optional[PipelineResult]:
        """Internal — caller must hold self._lock."""

        # 1. Acquire frame
        if frame is None:
            frame = self.camera.get_frame()
            if frame is None:
                return None

        try:
            # 2. Sync encoding cache
            known_encodings, known_names = self.db.get_encodings_and_names()
            self.recognizer.update_cache(
                known_encodings, known_names, self.db.version
            )

            # 3. Detect + extract embeddings
            detections = self.detector.detect_with_embeddings(frame)

            # 4. Track + recognize
            tracked_faces = self.tracker.update(
                detections,
                recognizer=self.recognizer,
            )

            # 5. FPS counter
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = now

            # 6. Periodic debug log (independent counter, not reset by FPS)
            self._debug_frame_count += 1
            if self._debug_frame_count % 30 == 0:
                n_valid = sum(1 for d in detections if d.quality_pass)
                logger.debug(
                    "Frame #%d: %d faces, %d quality-pass, %d tracks",
                    self._frame_count, len(detections), n_valid,
                    self.tracker.track_count,
                )

            return PipelineResult(
                frame=frame,
                raw_detections=detections,
                tracked_faces=tracked_faces,
                fps=self._current_fps,
            )

        except FaceHubError:
            raise
        except Exception as e:
            raise FaceHubError(f"Pipeline processing failed: {e}") from e

    # ── Convenience methods ──────────────────────────────────────

    def detect_only(self, frame: np.ndarray) -> "List[DetectionResult]":
        """Run detection only (no tracking/recognition)."""
        with self._lock:
            return self.detector.detect(frame)

    def extract_embeddings(self, frame: np.ndarray) -> "List[DetectionWithEmbedding]":
        """Run detection + embedding extraction (no tracking)."""
        with self._lock:
            return self.detector.detect_with_embeddings(frame)
```

---

### 文件 4-5：`face_hub/__init__.py`

```python
"""
FaceVision — Real-time Face Recognition Library

Usage:
    from face_hub import (
        FaceHubPipeline, FaceDetector, FaceRecognizer,
        CameraThread, FaceTracker, FaceDatabase,
    )
    from face_hub.types import UNKNOWN_SENTINEL, PipelineResult, TrackedFace
"""

__version__ = "1.0.0"
__author__ = "AllenDeng"

# ── Core components (from engine subpackage) ─────────────────
from face_hub.engine.camera import CameraThread
from face_hub.engine.face_detector import FaceDetector
from face_hub.engine.face_recognizer import FaceRecognizer
from face_hub.engine.face_tracker import FaceTracker
from face_hub.engine.face_database import FaceDatabase

# ── Pipeline (from face_hub/ main package) ───────────────────
from face_hub.pipeline import FaceHubPipeline
from face_hub.detector_protocol import DetectorProtocol

# ── Types ──────────────────────────────────────────────────────
from face_hub.types import (
    UNKNOWN_SENTINEL,
    BBox,
    DetectionResult,
    DetectionWithEmbedding,
    TrackedFace,
    PipelineResult,
)

# ── Exceptions ─────────────────────────────────────────────────
from face_hub.exceptions import (
    FaceHubError,
    ModelLoadError,
    InferenceError,
    CameraError,
    DatabaseError,
    RecognitionError,
)

# ── Config ─────────────────────────────────────────────────────
from face_hub.engine.config import DEFAULT_SETTINGS, get_default_settings

__all__ = [
    # Core
    "CameraThread",
    "FaceDetector",
    "FaceRecognizer",
    "FaceTracker",
    "FaceDatabase",
    "FaceHubPipeline",
    "DetectorProtocol",
    # Types
    "UNKNOWN_SENTINEL",
    "BBox",
    "DetectionResult",
    "DetectionWithEmbedding",
    "TrackedFace",
    "PipelineResult",
    # Exceptions
    "FaceHubError",
    "ModelLoadError",
    "InferenceError",
    "CameraError",
    "DatabaseError",
    "RecognitionError",
    # Config
    "DEFAULT_SETTINGS",
    "get_default_settings",
]
```

---

## 四、`tests/` — 测试文件（新写）

> **交付规则：所有测试必须通过，否则不视为交付完成。**
> 测试框架：`pytest`。运行：`pytest tests/ -v`

### 5.1 `tests/conftest.py` — 共享 fixtures

```python
"""
Shared pytest fixtures for FaceVision tests.
"""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_encoding():
    """Return a normalized 512-dim fake encoding."""
    vec = np.random.randn(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def sample_frame():
    """Return a dummy BGR frame (480x640)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def temp_db_paths():
    """Create temporary database paths, cleanup after test."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_db.json")
    enc_path = os.path.join(tmpdir, "test_enc.pkl")
    yield db_path, enc_path
    # Cleanup
    for p in [db_path, enc_path]:
        if os.path.exists(p):
            os.remove(p)
    os.rmdir(tmpdir)
```

### 5.2 `tests/test_types.py`

```python
"""Test dataclass types and UNKNOWN_SENTINEL."""

import numpy as np
from face_hub.types import (
    UNKNOWN_SENTINEL, BBox, DetectionResult,
    DetectionWithEmbedding, TrackedFace, PipelineResult,
)


class TestBBox:
    def test_create(self):
        b = BBox(x1=10, y1=20, x2=100, y2=200)
        assert b.x1 == 10
        assert b.width == 90
        assert b.height == 180

    def test_to_tuple(self):
        b = BBox(1, 2, 3, 4)
        assert b.to_tuple() == (1, 2, 3, 4)

    def test_frozen(self):
        b = BBox(1, 2, 3, 4)
        try:
            b.x1 = 99
            assert False, "Should raise FrozenInstanceError"
        except Exception:
            pass

    def test_hashable(self):
        a = BBox(1, 2, 3, 4)
        b = BBox(1, 2, 3, 4)
        c = BBox(5, 6, 7, 8)
        assert hash(a) == hash(b)
        assert hash(a) != hash(c)
        d = {a: "test"}
        assert d[b] == "test"


class TestDetectionResult:
    def test_from_tuple(self):
        r = DetectionResult.from_tuple((10, 20, 100, 200, 0.95))
        assert r.bbox.x1 == 10
        assert r.confidence == 0.95


class TestDetectionWithEmbedding:
    def test_from_tuple(self, sample_encoding):
        r = DetectionWithEmbedding.from_tuple(
            (10, 20, 100, 200, 0.95, sample_encoding, True)
        )
        assert r.bbox.width == 90
        assert r.confidence == 0.95
        assert r.has_embedding is True
        assert r.quality_pass is True

    def test_no_embedding(self):
        r = DetectionWithEmbedding.from_tuple((0, 0, 50, 50, 0.8, None, False))
        assert r.has_embedding is False


class TestTrackedFace:
    def test_is_known(self):
        known = TrackedFace(
            track_id=1, bbox=BBox(0, 0, 10, 10),
            name="Alice", confidence=0.9, det_confidence=0.95,
            is_confirmed=True, quality_pass=True,
        )
        assert known.is_known is True

    def test_is_unknown(self):
        unknown = TrackedFace(
            track_id=2, bbox=BBox(0, 0, 10, 10),
            name=UNKNOWN_SENTINEL, confidence=0.0, det_confidence=0.8,
            is_confirmed=False, quality_pass=True,
        )
        assert unknown.is_known is False


class TestPipelineResult:
    def test_known_faces_filter(self):
        faces = [
            TrackedFace(1, BBox(0,0,10,10), "Alice", 0.9, 0.95, True, True),
            TrackedFace(2, BBox(10,10,20,20), UNKNOWN_SENTINEL, 0.0, 0.8, False, True),
        ]
        r = PipelineResult(frame=np.zeros((10,10,3)), tracked_faces=faces)
        assert len(r.known_faces) == 1
        assert r.unknown_count == 1
        assert r.total_faces == 2


class TestUnknownSentinel:
    def test_value(self):
        assert UNKNOWN_SENTINEL == "unknown"
```

### 5.3 `tests/test_exceptions.py`

```python
"""Test exception hierarchy."""

from face_hub.exceptions import (
    FaceHubError, ModelLoadError, InferenceError,
    CameraError, DatabaseError, RecognitionError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        for cls in [ModelLoadError, InferenceError, CameraError,
                     DatabaseError, RecognitionError]:
            assert issubclass(cls, FaceHubError)

    def test_catch_by_base(self):
        try:
            raise ModelLoadError("test")
        except FaceHubError:
            pass  # should catch
        else:
            assert False

    def test_cause_chain(self):
        try:
            try:
                raise RuntimeError("root cause")
            except RuntimeError as e:
                raise ModelLoadError("wrapper") from e
        except ModelLoadError as ex:
            assert isinstance(ex.__cause__, RuntimeError)
            assert "root cause" in str(ex.__cause__)
```

### 5.4 `tests/test_config.py`

```python
"""Test config constants."""

from face_hub import DEFAULT_SETTINGS, get_default_settings


class TestDefaultSettings:
    def test_required_keys(self):
        required = [
            "device", "confidence", "tolerance", "cam_width", "cam_height",
            "cam_fps", "proc_fps", "det_size", "track_smooth",
            "min_face_size", "quality_filter",
        ]
        for key in required:
            assert key in DEFAULT_SETTINGS, f"Missing key: {key}"

    def test_deepcopy_isolation(self):
        a = get_default_settings()
        b = get_default_settings()
        a["confidence"] = 0.99
        assert b["confidence"] == 0.50
        assert a is not b
```

### 5.5 `tests/test_face_database.py`

```python
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
```

### 5.6 `tests/test_face_recognizer.py`

```python
"""Test FaceRecognizer cosine similarity matching and cache."""

import numpy as np
from face_hub import FaceRecognizer, UNKNOWN_SENTINEL


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
```

### 5.7 `tests/test_face_tracker.py`

```python
"""Test FaceTracker IoU matching, majority vote, identity confirmation."""

import numpy as np
from face_hub import FaceTracker, UNKNOWN_SENTINEL, BBox, DetectionWithEmbedding


def make_detection(x1, y1, x2, y2, embedding=None, quality=True):
    return DetectionWithEmbedding(
        bbox=BBox(x1, y1, x2, y2),
        confidence=0.95,
        embedding=embedding if embedding is not None else np.random.randn(512).astype(np.float32),
        quality_pass=quality,
    )


class TestFaceTracker:
    def test_init(self):
        t = FaceTracker(smooth_frames=5)
        assert t.track_count == 0

    def test_first_detection_creates_track(self):
        t = FaceTracker(smooth_frames=3)
        dets = [make_detection(10, 10, 100, 100)]
        results = t.update(dets)
        assert len(results) == 1
        assert results[0].name == UNKNOWN_SENTINEL
        assert results[0].is_confirmed is False

    def test_iou_matching_same_person(self):
        t = FaceTracker(smooth_frames=3, iou_threshold=0.20)
        dets1 = [make_detection(10, 10, 100, 100)]
        t.update(dets1)

        # Slightly moved bbox — should match same track
        dets2 = [make_detection(12, 8, 102, 98)]
        results = t.update(dets2)
        assert len(results) == 1
        assert t.track_count == 1  # still one track, not two

    def test_no_iou_creates_new_track(self):
        t = FaceTracker(smooth_frames=3, iou_threshold=0.20)
        dets1 = [make_detection(10, 10, 100, 100)]
        t.update(dets1)

        # Far away detection — new track
        dets2 = [make_detection(300, 300, 400, 400)]
        results = t.update(dets2)
        assert len(results) == 2  # both tracks alive (old not stale yet)

    def test_stale_track_removal(self):
        t = FaceTracker(smooth_frames=3, max_missed=2)
        dets = [make_detection(10, 10, 100, 100)]
        t.update(dets)
        assert t.track_count == 1

        # Two empty updates → track becomes stale
        t.update([])
        assert t.track_count == 1  # 1 miss
        t.update([])
        assert t.track_count == 0  # 2 misses = stale

    def test_reset(self):
        t = FaceTracker()
        t.update([make_detection(0, 0, 10, 10)])
        assert t.track_count == 1
        t.reset()
        assert t.track_count == 0
```

### 5.8 `tests/test_face_detector.py`

```python
"""
Test FaceDetector model loading and device fallback.
SKIP in CI (requires insightface model download + GPU).
Run locally only: pytest tests/test_face_detector.py
"""
import pytest
import numpy as np
from face_hub import FaceDetector


class TestFaceDetectorConstruction:
    def test_cpu_init(self):
        """Detector should load on CPU without error."""
        d = FaceDetector(device="cpu", det_size=320)
        assert d.device == "cpu"
        assert d.app is not None
        assert d.det_size == 320

    def test_auto_falls_back_to_cpu(self):
        """device='auto' should at minimum succeed (even if no GPU)."""
        d = FaceDetector(device="auto", det_size=320)
        assert d.app is not None

    def test_detect_on_dummy_frame(self, sample_frame):
        """Detection should return list (may be empty on random noise)."""
        d = FaceDetector(device="cpu", det_size=320)
        results = d.detect(sample_frame)
        assert isinstance(results, list)

    def test_detect_with_embeddings_dummy(self, sample_frame):
        """Embedding extraction should return list."""
        d = FaceDetector(device="cpu", det_size=320)
        results = d.detect_with_embeddings(sample_frame)
        assert isinstance(results, list)
```

### 5.9 `tests/test_pipeline.py`

```python
"""Test FaceHubPipeline integration (using mock camera)."""

import numpy as np
from face_hub import (
    FaceHubPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, PipelineResult,
)


class MockCamera:
    """Minimal camera stub for pipeline tests."""
    running = False
    def start(self):
        self.running = True
    def stop(self):
        self.running = False
    def get_frame(self, timeout=None, copy=True):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class TestPipeline:
    def test_init(self, temp_db_paths):
        db_path, enc_path = temp_db_paths
        camera = MockCamera()
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        # Use CPU to avoid GPU dependency in tests
        detector = FaceDetector(device="cpu", det_size=320)
        recognizer = FaceRecognizer(tolerance=0.45)
        tracker = FaceTracker(smooth_frames=3)

        pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
        assert pipeline.is_running is False

    def test_start_stop(self, temp_db_paths):
        db_path, enc_path = temp_db_paths
        camera = MockCamera()
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        detector = FaceDetector(device="cpu", det_size=320)
        recognizer = FaceRecognizer()
        tracker = FaceTracker()

        pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
        pipeline.start()
        assert pipeline.is_running is True
        pipeline.stop()
        assert pipeline.is_running is False

    def test_process_frame_with_explicit_frame(self, temp_db_paths, sample_frame):
        db_path, enc_path = temp_db_paths
        camera = MockCamera()
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        detector = FaceDetector(device="cpu", det_size=320)
        recognizer = FaceRecognizer()
        tracker = FaceTracker()

        pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
        # process_frame with explicit frame (no camera needed)
        result = pipeline.process_frame(frame=sample_frame)
        assert isinstance(result, PipelineResult)
        assert result.fps >= 0.0

    def test_detect_only(self, temp_db_paths, sample_frame):
        db_path, enc_path = temp_db_paths
        camera = MockCamera()
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        detector = FaceDetector(device="cpu", det_size=320)
        recognizer = FaceRecognizer()
        tracker = FaceTracker()

        pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
        results = pipeline.detect_only(sample_frame)
        assert isinstance(results, list)

    def test_update_database_cache(self, temp_db_paths, sample_encoding):
        db_path, enc_path = temp_db_paths
        camera = MockCamera()
        db = FaceDatabase(db_path=db_path, encoding_path=enc_path)
        detector = FaceDetector(device="cpu", det_size=320)
        recognizer = FaceRecognizer()
        tracker = FaceTracker()

        pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
        db.add_person("Test", "/tmp/test.jpg", sample_encoding)
        rebuilt = pipeline.update_database_cache()
        assert rebuilt is True
        assert "Test" in recognizer.cached_names
```

---

## 五、实施步骤

> **交付门禁：步骤 20 全部测试通过后方可交付。**

```
步骤  1: 创建 face_hub/ 目录 + face_hub/engine/ 子目录
步骤  2: 创建 tests/ 目录
步骤  3: 创建 face_hub/engine/__init__.py      ← 二 文件3-0
步骤  4: 创建 face_hub/engine/config.py        ← 二 文件3-1
步骤  5: 创建 face_hub/engine/camera.py        ← 二 文件3-2
步骤  6: 创建 face_hub/engine/face_detector.py ← 二 文件3-3
步骤  7: 创建 face_hub/engine/face_recognizer.py ← 二 文件3-4（复制原文件改1行）
步骤  8: 创建 face_hub/engine/face_tracker.py  ← 二 文件3-5（复制原文件改2处）
步骤  9: 创建 face_hub/engine/face_database.py ← 二 文件3-6（复制原文件改3处）
步骤 10: 创建 face_hub/types.py                ← 三 文件4-1
步骤 11: 创建 face_hub/exceptions.py           ← 三 文件4-2
步骤 12: 创建 face_hub/detector_protocol.py    ← 三 文件4-3
步骤 13: 创建 face_hub/pipeline.py             ← 三 文件4-4
步骤 14: 创建 face_hub/__init__.py             ← 三 文件4-5
步骤 15: 创建 tests/conftest.py                  ← 四 5.1
步骤 16: 创建 8 个 tests/test_*.py               ← 四 5.2-5.9
步骤 17: pip install -e . && pytest tests/ -v --ignore=tests/test_face_detector.py
```

> **打包/发布/文档步骤见 [RELEASE_PLAN.md](RELEASE_PLAN.md)**

---

## 六、验证清单

- [ ] `pip install -e .` 成功
- [ ] `python -c "from face_hub import *"` 无 ImportError
- [ ] `python -c "from face_hub.types import UNKNOWN_SENTINEL; print(UNKNOWN_SENTINEL)"` → `unknown`
- [ ] `python -c "from face_hub import FaceHubError; raise FaceHubError('test')"` 正常
- [ ] `python -c "from face_hub import DEFAULT_SETTINGS; print(DEFAULT_SETTINGS['device'])"` → `cuda`
- [ ] `python -c "from face_hub import FaceDetector; d = FaceDetector(device='cpu', det_size=320)"` 模型加载成功
- [ ] `python -c "from face_hub import FaceRecognizer; r = FaceRecognizer()"` 无错误
- [ ] `python -c "from face_hub import FaceDatabase; db = FaceDatabase(db_path='test_db.json'); print(len(db.get_names()))"` → `0`
- [ ] `python -c "from face_hub import CameraThread; print(CameraThread.list_cameras())"` 列出摄像头
- [ ] `python -c "from face_hub import FaceHubPipeline"` 无 ImportError
- [ ] `python -c "from face_hub import BBox, DetectionResult, DetectionWithEmbedding, TrackedFace, PipelineResult"` 全部可导入
- [ ] **`pytest tests/ -v` 全部通过（交付门禁）**

---

## 七、注意事项

1. **单一顶层包：** `face_hub/` 是唯一 Python 包；`engine/` 子包包含全部核心算法。引擎模块从 `face_hub.types` / `face_hub.exceptions` 导入类型和异常
2. **多平台摄像头：** `CameraThread._get_backend()` 自动按平台选择 DShow(Win) / AVFoundation(macOS) / V4L2(Linux)
3. **v1.0 多平台 GPU：** `FaceDetector._resolve_providers()` 优先级 CUDA → DirectML(Win) → CPU；macOS 仅 CPU（CoreML 延后 v1.1）
4. **insightface 模型：** 自动下载到 `~/.insightface`，全平台一致
5. **`device="auto"`：** 等价于 `"cuda"`，自动探测最优 GPU（macOS 上等同于 `"cpu"`）
6. **BBox vs tuple：** IoU 计算、cv2 调用内部仍用 tuple 保证性能；仅公开 API 返回 BBox dataclass
7. **线程安全：** `Pipeline.process_frame()` 有 `threading.Lock`；`FaceRecognizer.update_cache()` 由 Pipeline 内部串行化
8. **async：** v1.0 仅同步接口；v1.1+ 用 `asyncio.to_thread()` 包装
9. **代码风格：** 全部注释/报错使用英文；遵循 PEP8 规范（见第十章）


---

## 八、第三方依赖清单

> 以下为 `pip install facevision` 时自动安装的依赖及其用途。

### 8.1 核心依赖（必装）

| 库 | 最低版本 | License | 用途 |
|----|----------|---------|------|
| **opencv-python** | ≥4.8.0 | Apache 2.0 | 摄像头采集、图像预处理、可视化 |
| **insightface** | ≥0.7.3 | MIT (代码) / **模型非商用** ⚠️ | RetinaFace 检测 + ArcFace (buffalo_l) 特征提取 |
| **onnxruntime** | ≥1.18.0 | MIT | ONNX 推理引擎 (CPU) |
| **numpy** | ≥1.24.0 | BSD-3-Clause | 特征向量矩阵运算 |
| **Pillow** | ≥10.0.0 | MIT-CMU | 图片读写（insightface 内部依赖） |

### 8.2 GPU 加速依赖（可选）

| 库 | 平台 | License | 安装方式 |
|----|------|---------|----------|
| **onnxruntime-directml** | Windows | MIT | `pip uninstall onnxruntime && pip install facevision[gpu-win]` |
| **onnxruntime-gpu** | Linux | MIT | `pip uninstall onnxruntime && pip install facevision[gpu-linux]` |

### 8.3 间接依赖（自动安装）

| 库 | License | 来源 | 用途 |
|----|---------|------|------|
| **scikit-learn** | BSD-3-Clause | insightface | — |
| **scipy** | BSD-3-Clause | insightface | — |
| **tqdm** | MPL 2.0 / MIT 双许可 | insightface | 模型下载进度条 |
| **protobuf** | BSD-3-Clause | onnxruntime | ONNX 模型解析 |
| **flatbuffers** | Apache 2.0 | onnxruntime | 内部序列化 |
| **sympy** | BSD-3-Clause | onnxruntime | 图优化 |

### 8.4 开发/测试依赖

| 库 | License | 用途 |
|----|---------|------|
| **pytest** | MIT | 测试框架 |
| **build** | MIT | PEP 517 构建 |
| **twine** | Apache 2.0 | PyPI 上传校验 |


### 8.5 License 兼容性分析

> **结论：FaceVision 使用 MIT License 没有法律风险。** 所有依赖均为 MIT 兼容协议。

#### 协议兼容性矩阵

| 依赖协议 | 是否 MIT 兼容 | 说明 |
|----------|:--:|------|
| MIT | ✅ | 完全兼容 |
| BSD-3-Clause | ✅ | 仅要求保留版权声明，可自由再许可为 MIT |
| Apache 2.0 | ✅ | 与 MIT 单向兼容；需保留 NOTICE 文件 |
| MIT-CMU | ✅ | MIT 变体，完全兼容 |
| MPL 2.0 (tqdm) | ✅ | 文件级 copyleft，不传染项目 |

#### ⚠️ 唯一风险点：insightface 预训练模型

```
insightface 代码    → MIT License ✅  可商用
insightface 模型    → 非商用研究目的  ⚠️ 不可商用
  ├── buffalo_l     → RetinaFace + ArcFace 权重
  ├── antelopev2    → 3D 重建模型
  └── inswapper     → 换脸模型
```

**FaceVision 自动下载的 `buffalo_l` 模型受 insightface 非商用限制。** 这意味着：
- 个人项目、学术研究、学习用途 → ✅ 完全自由
- 商业产品中使用 `FaceVision` → ⚠️ 需向 insightface 申请模型商用授权
- 模型文件不会随 `pip install facevision` 打包（运行时自动下载到 `~/.insightface`），但下载即视为接受 insightface 的模型许可

**规避方案（如需商用）:**
1. 联系 `recognition-oss-pack@insightface.ai` 申请模型商用授权
2. 替换为自己的 RetinaFace + ArcFace 模型文件，放入 `~/.insightface/models/buffalo_l/`
3. 使用 InspireFace SDK（insightface 官方跨平台 C++ SDK，含商用许可选项）


---

## 九、代码规范

### 9.1 语言

- **注释、docstring、报错信息、logger 输出：** 全部英文
- **本计划文档：** 中文（面向开发者阅读）

### 9.2 PEP8 要点

| 规则 | 说明 |
|------|------|
| 缩进 | 4 空格，禁止 Tab |
| 行宽 | ≤ 100 字符（代码）/ ≤ 79 字符（docstring） |
| 空行 | 模块级函数/类之间空 2 行；类内方法之间空 1 行 |
| import | 标准库 → 第三方 → 本地，每组之间空 1 行；禁止 `from module import *`（`__init__.py` 除外） |
| 命名 | `ClassName` / `function_name` / `variable_name` / `CONSTANT` / `_private` |
| docstring | 所有公开函数/类必须有 `"""triple-quote"""` docstring |
| 类型注解 | 所有公开方法必须有参数类型和返回值类型注解 |
| 逗号尾随 | 多行列表/字典最后一项加逗号 |
| 字符串 | 统一双引号 `"..."`，docstring 用三重双引号 `"""..."""` |
| 异常 | 抛自定义异常（`FaceHubError` 子类），不抛裸 `Exception` / `print`后继续 |

### 9.3 示例

```python
# ✅ 正确
def detect_with_embeddings(self, frame: np.ndarray) -> List[DetectionWithEmbedding]:
    """Run face detection and extract embeddings in one pass.

    Args:
        frame: BGR image as (H, W, 3) numpy array.

    Returns:
        List of DetectionWithEmbedding, empty if no faces found.

    Raises:
        InferenceError: If both GPU and CPU inference fail.
    """
    if self.app is None or frame is None or frame.size == 0:
        return []

# ❌ 错误
def detect_with_embeddings(self, frame):   # 无类型注解，无 docstring
    if self.app is None or frame is None or frame.size == 0:
        print("无效输入")                   # 中文 + print
        return []
```


---

## 十、运行时内存与拷贝优化

> 实时人脸识别对延迟敏感，需尽可能减少不必要的 `np.ndarray` / `frame` 拷贝。

### 10.1 数据流中的拷贝点分析

```
CameraThread._loop  ──frame.copy()──→  CameraThread.get_frame()
        │
        ▼
Pipeline._process_frame_impl()
        │
        ├─→ detector.detect_with_embeddings(frame)
        │       ├─ insightface 内部 (GPU→CPU 传回，不可避免)
        │       ├─ face.bbox.astype(int)  ← 微小拷贝 (4 个 int)
        │       ├─ face.normed_embedding  ← 已是引用，无需拷贝
        │       └─ _face_quality(): cv2.cvtColor → Laplacian  ← 质量检查必要开销
        │
        ├─→ tracker.update(detections, recognizer)
        │       └─ recognizer.recognize(embedding)
        │               └─ np.asarray(enc, dtype=float32)  ← 当 enc 已是 float32 时多余
        │
        └─→ PipelineResult(frame=frame)  ← frame 作为引用传入，不拷贝
```

### 10.2 具体优化措施

#### A. `CameraThread` — 提供零拷贝模式

```python
def get_frame(self, timeout=0.05, copy=True):
    """..."""
    if self._frame_available.wait(timeout):
        with self.lock:
            if len(self._frame_buffer) > 0:
                frame = self._frame_buffer[0]
                self._frame_available.clear()
                return frame.copy() if copy else frame  # ← 零拷贝选项
    return None
```

> **默认 `copy=True`** 保证线程安全；高级用户在确认不会并发访问时可传 `copy=False`。

#### B. `FaceRecognizer.recognize()` — 避免无意义的 dtype 转换

```python
# 优化前：每次都 asarray + astype
unknown_encoding = np.asarray(unknown_encoding, dtype=np.float32).ravel()

# 优化后：已经是 float32 就跳过
if not isinstance(unknown_encoding, np.ndarray):
    unknown_encoding = np.asarray(unknown_encoding, dtype=np.float32)
elif unknown_encoding.dtype != np.float32:
    unknown_encoding = unknown_encoding.astype(np.float32)
unknown_encoding = unknown_encoding.ravel()  # ravel 优先返回 view
```

> insightface 返回的 `normed_embedding` 始终是 `float32`，优化后每次 `recognize()` 节省一次 `(512,)` 数组的 alloc+copy。

#### C. `FaceDatabase` — 返回只读引用

```python
def get_encodings_and_names(self) -> tuple:
    """Return (encodings, names). DO NOT mutate the returned lists."""
    return self.encodings, [p["name"] for p in self.persons]
    #                                            ↑ 名字列表每次新建，可缓存

# 优化：缓存名字列表
def __init__(self, ...):
    ...
    self._cached_names = []  # 与 self.persons 保持同步

def get_encodings_and_names(self) -> tuple:
    return self.encodings, self._cached_names
```

#### D. `Pipeline._process_frame_impl()` — frame 零拷贝传递

```python
def _process_frame_impl(self, frame) -> Optional[PipelineResult]:
    if frame is None:
        frame = self.camera.get_frame(copy=False)  # ← 零拷贝
        if frame is None:
            return None
    # ... 下游只读 frame，不修改 ...
    return PipelineResult(frame=frame, ...)  # ← 引用传递
```

> **前提：** 下游代码（detector、tracker）不对 frame 做原地修改。insightface 的内部实现确实是只读的。

#### E. `FaceTrack` — 已使用 `__slots__`

```python
class FaceTrack:
    __slots__ = ('id', 'bbox', 'name_history', 'conf_history',
                 'quality_history', 'frames_since_update', 'total_frames',
                 'confirmed_name', 'confirmed_conf', 'latest_name', 'latest_conf')
```

> 相比 `__dict__`，`__slots__` 节省 ~60% 内存且属性访问更快。已实现，无需改动。

#### F. 编码缓存预分配

```python
# FaceRecognizer.update_cache():
# 当 known_encodings 数量不变时，复用已有 ndarray 而非重新创建
if (self._cached_encodings is not None
        and len(known_encodings) == self._cached_encodings.shape[0]):
    self._cached_encodings[:] = known_encodings  # in-place update
else:
    self._cached_encodings = np.array(known_encodings, dtype=np.float32)
```

### 10.3 优化总结

| 优化点 | 节省 | 风险 |
|--------|------|------|
| `get_frame(copy=False)` | 每帧省 ~1MB alloc+copy (1080p) | 调用方不得修改返回帧 |
| `recognize()` dtype 检查 | 每次识别省 1 次 `(512,)` alloc | 无 |
| `get_encodings_and_names()` 缓存 | 避免重复创建名字列表 | 需在 add/remove 时同步 |
| Pipeline frame 引用传递 | 每帧省 ~1MB alloc+copy | 下游只读保证 |
| `update_cache` in-place | 编码数量不变时免 alloc | 需保证外部不再持有旧引用 |


---

## 十一、中英文 API 文档

> 文档目录 `docs/`，使用 Markdown 编写，中英独立目录。
> 每个公开 API 均需提供中英两份文档，内容一一对应。

### 11.1 文档工具链

```bash
# 本地预览
pip install mkdocs mkdocs-material
cd docs && mkdocs serve

# 部署到 GitHub Pages
mkdocs gh-deploy
```

`docs/mkdocs.yml`（中英双语配置）：

```yaml
site_name: FaceVision
theme: material
nav:
  - English: en/
  - 中文: zh/
```

### 11.2 文档内容规范

每份 API 文档必须包含以下章节（中英一致）：

```
# <ClassName> — <一句话描述>

## 构造参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|

## 属性
| 属性 | 类型 | 说明 |
|------|------|------|

## 方法
### method_name(param1, param2)
> 一句话说明

**Parameters:**
- `param1` (Type): description
- `param2` (Type): description

**Returns:**
- (Type): description

**Raises:**
- `ExceptionType`: when

**Example:**
```python
...
```
```

### 11.3 文档清单

| 文档 | 中文路径 | 英文路径 | 覆盖内容 |
|------|----------|----------|----------|
| 首页 | `zh/index.md` | `en/index.md` | 项目介绍、特性、平台支持 |
| 安装 | `zh/installation.md` | `en/installation.md` | pip install、GPU 选装、平台说明 |
| 快速开始 | `zh/quickstart.md` | `en/quickstart.md` | 5 分钟跑通完整流程 |
| Pipeline | `zh/api/pipeline.md` | `en/api/pipeline.md` | FaceHubPipeline 全部方法 |
| 检测器 | `zh/api/detector.md` | `en/api/detector.md` | FaceDetector + DetectorProtocol |
| 识别器 | `zh/api/recognizer.md` | `en/api/recognizer.md` | FaceRecognizer + 阈值调参 |
| 追踪器 | `zh/api/tracker.md` | `en/api/tracker.md` | FaceTracker + IoU/投票算法说明 |
| 数据库 | `zh/api/database.md` | `en/api/database.md` | FaceDatabase CRUD + 持久化 |
| 摄像头 | `zh/api/camera.md` | `en/api/camera.md` | CameraThread + 多平台后端 |
| 类型 | `zh/api/types.md` | `en/api/types.md` | BBox, DetectionResult, TrackedFace 等 |
| 异常 | `zh/api/exceptions.md` | `en/api/exceptions.md` | 异常层级 + 每种异常的触发条件 |
| 示例：基础 | `zh/examples/basic_webcam.md` | `en/examples/basic_webcam.md` | 打开摄像头实时识别 |
| 示例：自定义 | `zh/examples/custom_detector.md` | `en/examples/custom_detector.md` | 实现 DetectorProtocol 接入自有模型 |

### 11.4 快速开始文档模板

**中文版 (`zh/quickstart.md`)：**

```markdown
# 快速开始

## 安装

pip install facevision

## 5 分钟示例

from face_hub import (
    FaceHubPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread,
)

# 1. 初始化组件
db = FaceDatabase(db_path="face_db.json")
detector = FaceDetector(device="auto", det_size=640)
recognizer = FaceRecognizer(tolerance=0.45)
tracker = FaceTracker(smooth_frames=5)
camera = CameraThread(camera_id=0, width=640, height=360)

# 2. 组装流水线
pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
pipeline.start()

# 3. 循环处理
while True:
    result = pipeline.process_frame()
    if result is None:
        continue
    for face in result.known_faces:
        print(f"{face.name} ({face.confidence:.0%})")

pipeline.stop()
```

**英文版 (`en/quickstart.md`)：** 结构相同，英文撰写。

### 11.5 API 文档模板（以 FaceDetector 为例）

**中文版 (`zh/api/detector.md`)：**

````markdown
# FaceDetector

基于 insightface RetinaFace 的人脸检测器，支持 GPU 自动检测与回退。

## 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `confidence` | `float` | `0.50` | 检测置信度阈值 (0.0~1.0)，低于此值的人脸被丢弃 |
| `device` | `str` | `"auto"` | 推理设备：`"cpu"` / `"cuda"` / `"auto"`. `"auto"` 自动探测最优 GPU |
| `det_size` | `int` | `640` | 检测模型输入尺寸：320(快) / 480(均衡) / 640(精准) |
| `quality_filter` | `bool` | `True` | 是否启用模糊度过滤 |
| `min_face_size` | `int` | `80` | 最小人脸尺寸 (px)，小于此值标记为低质量 |

## 方法

### detect(frame)

仅检测人脸，不提取特征。

**参数:**
- `frame` (`np.ndarray`): BGR 图像 (H, W, 3)

**返回:**
- `List[DetectionResult]`: 检测到的人脸列表

### detect_with_embeddings(frame)

检测人脸并提取 512 维特征向量。

**参数:**
- `frame` (`np.ndarray`): BGR 图像 (H, W, 3)

**返回:**
- `List[DetectionWithEmbedding]`: 人脸列表，含特征向量

### reload_model(det_size=None)

运行时切换检测尺寸。

**参数:**
- `det_size` (`int | None`): 新尺寸，`None` 表示保持当前值

## 自定义检测器

实现 `DetectorProtocol` 接口即可接入自有模型：

```python
from face_hub import DetectorProtocol, DetectionWithEmbedding, BBox

class MyDetector:
    def detect_with_embeddings(self, frame):
        ...  # 你的检测 + 特征提取逻辑
        return [DetectionWithEmbedding(...)]

pipeline = FaceHubPipeline(camera, MyDetector(), ...)
```
````

**英文版 (`en/api/detector.md`)：** 结构相同，英文撰写。

### 11.6 类型文档模板 (`zh/api/types.md`)

```markdown
# 数据类型

所有公开 API 返回类型均使用 dataclass，保证类型安全。

## BBox

不可变边界框 (x1, y1, x2, y2)。

| 属性 | 类型 | 说明 |
|------|------|------|
| `x1` | `int` | 左上角 x |
| `y1` | `int` | 左上角 y |
| `x2` | `int` | 右下角 x |
| `y2` | `int` | 右下角 y |
| `width` | `int` | 宽度 (x2 - x1) |
| `height` | `int` | 高度 (y2 - y1) |

方法: `to_tuple()` → `(x1, y1, x2, y2)`

## DetectionResult

`detect()` 返回的单条结果。

| 属性 | 类型 | 说明 |
|------|------|------|
| `bbox` | `BBox` | 边界框 |
| `confidence` | `float` | 检测置信度 0.0~1.0 |

## DetectionWithEmbedding

`detect_with_embeddings()` 返回的单条结果。继承 `DetectionResult` 语义。

| 属性 | 类型 | 说明 |
|------|------|------|
| `bbox` | `BBox` | 边界框 |
| `confidence` | `float` | 检测置信度 |
| `embedding` | `np.ndarray \| None` | 512 维 L2 归一化特征向量 |
| `quality_pass` | `bool` | 质量过滤是否通过 |
| `has_embedding` | `bool` | 特征是否提取成功 |

## TrackedFace

`FaceTracker.update()` 返回的单条追踪结果。

| 属性 | 类型 | 说明 |
|------|------|------|
| `track_id` | `int` | 追踪 ID |
| `bbox` | `BBox` | 当前帧边界框 |
| `name` | `str` | 识别姓名，`UNKNOWN_SENTINEL` 表示未匹配 |
| `confidence` | `float` | 余弦相似度 0.0~1.0 |
| `det_confidence` | `float` | 检测置信度 |
| `is_confirmed` | `bool` | 身份是否经多数投票确认 |
| `quality_pass` | `bool` | 质量过滤是否通过 |
| `is_known` | `bool` | 是否为已注册人员 (name != UNKNOWN_SENTINEL) |

## PipelineResult

`FaceHubPipeline.process_frame()` 的完整返回。

| 属性 | 类型 | 说明 |
|------|------|------|
| `frame` | `np.ndarray` | 当前帧 |
| `raw_detections` | `List[DetectionWithEmbedding]` | 原始检测结果 |
| `tracked_faces` | `List[TrackedFace]` | 追踪后的识别结果 |
| `fps` | `float` | 当前处理帧率 |
| `known_faces` | `List[TrackedFace]` | 筛选已识别人员 |
| `unknown_count` | `int` | 未识别的人数 |
| `total_faces` | `int` | 总人脸数 |

## 全局常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `UNKNOWN_SENTINEL` | `"unknown"` | 未匹配到已注册人员时的默认名称 |
```

### 11.7 文档实施说明

文档随代码一同提交，放在 `docs/` 目录下。本地预览：

```bash
pip install mkdocs mkdocs-material
cd docs && mkdocs serve
```

在线部署（GitHub Pages）：
```bash
mkdocs gh-deploy
```
