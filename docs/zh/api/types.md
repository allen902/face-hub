# 类型与数据结构

FaceHub 定义了以下核心数据类型，它们在各组件之间传递数据。

---

## BBox

边界框，使用 (x1, y1, x2, y2) 坐标。

```python
class BBox:
    x1: int  # 左上角 x 坐标
    y1: int  # 左上角 y 坐标
    x2: int  # 右下角 x 坐标
    y2: int  # 右下角 y 坐标
```

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `width` | `int` | 边框宽度 (`x2 - x1`) |
| `height` | `int` | 边框高度 (`y2 - y1`) |
| `center` | `Tuple[int, int]` | 中心点坐标 `(cx, cy)` |
| `area` | `int` | 边框面积 (`width * height`) |

### 方法

#### to_tuple()

返回 `(x1, y1, x2, y2)` 元组。

```python
bbox = BBox(10, 20, 100, 200)
x1, y1, x2, y2 = bbox.to_tuple()
print(x1, y1, x2, y2)  # 10 20 100 200
```

#### to_xywh()

返回 `(x, y, w, h)` 元组（左上角坐标 + 宽高）。

```python
x, y, w, h = bbox.to_xywh()
print(x, y, w, h)  # 10 20 90 180
```

---

## DetectionResult

仅检测的结果（无特征向量）。由 `FaceDetector.detect()` 和 `FaceHubPipeline.detect_only()` 返回。

```python
class DetectionResult:
    bbox: BBox          # 边界框
    confidence: float   # 检测置信度 (0.0~1.0)
```

---

## DetectionWithEmbedding

检测 + 特征向量的结果。由 `FaceDetector.detect_with_embeddings()` 和 `FaceHubPipeline.extract_embeddings()` 返回。

```python
class DetectionWithEmbedding:
    bbox: BBox          # 边界框
    confidence: float   # 检测置信度 (0.0~1.0)
    embedding: Optional[np.ndarray]  # 512 维特征向量（L2 归一化后），提取失败时为 None
    quality_pass: bool  # 是否通过质量过滤（模糊度、尺寸）
```

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `has_embedding` | `bool` | `embedding` 是否非空且为 `float32` |

---

## TrackedFace

追踪后的单个人脸，包含身份和边框信息。由 `FaceTracker.update()` 返回。

```python
class TrackedFace:
    track_id: int       # 追踪 ID（递增，跨帧稳定）
    bbox: BBox          # 当前帧的边界框
    name: str           # 识别的姓名（未匹配时为 UNKNOWN_SENTINEL）
    confidence: float   # 匹配的余弦相似度 (0.0~1.0)，未匹配时为 0.0
    is_confirmed: bool  # 身份是否被多数投票确认
```

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `is_known` | `bool` | 姓名 ≠ `UNKNOWN_SENTINEL` 且已确认（`is_confirmed & name != UNKNOWN_SENTINEL`） |

---

## PipelineResult

完整流水线处理一帧后的结果。由 `FaceHubPipeline.process_frame()` 返回。

```python
class PipelineResult:
    frame: np.ndarray               # BGR 格式的当前帧
    tracked_faces: List[TrackedFace]  # 所有已追踪的人脸
    fps: float                      # 流水线处理帧率
```

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `known_faces` | `List[TrackedFace]` | `tracked_faces` 中 `is_known=True` 的子集 |
| `unknown_count` | `int` | `tracked_faces` 中 `is_known=False` 的数量 |
| `total_faces` | `int` | `len(tracked_faces)` |

---

## PhotoFace

一张照片中发现的一张人脸。存放在 `PhotoClassificationResult.faces` 中。

```python
from face_hub import PhotoFace, BBox

face = PhotoFace(
    photo_id="聚会1.jpg",
    bbox=BBox(100, 50, 300, 250),
    det_confidence=0.92,
    label="person_001",
    similarity=0.71,
)
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `photo_id` | `str` | 人脸所在照片的 id |
| `bbox` | `BBox` | 人脸边界框 |
| `det_confidence` | `float` | 检测置信度 `[0.0, 1.0]` |
| `label` | `str` | 人员姓名、聚类标签（`person_001`……）或 `UNKNOWN_SENTINEL` |
| `similarity` | `float` | 与匹配人员 / 聚类质心的余弦相似度 |

---

## PhotoGroup

包含同一个人物的一组照片。

| 属性 | 类型 | 说明 |
|------|------|------|
| `label` | `str` | 人员姓名或聚类标签 |
| `photo_ids` | `List[str]` | 去重后的照片 id，按首次出现顺序 |
| `face_count` | `int` | 该人物在组内的人脸总数 |

### 计算属性

| 属性 | 说明 |
|------|------|
| `photo_count` | `len(photo_ids)` |

!!! note
    包含多人的照片会同时出现在多个分组中，因此各分组的 `photo_count`
    之和**不等于** `total_photos`。

---

## PhotoClassificationResult

由 `PhotoClassifier.classify_photos()` 和便捷函数 `classify_photos()`
返回。用法详见[照片分类器](classifier.md)。

```python
result = classifier.classify_photos(photos)

print(result.groups)            # {"person_001": PhotoGroup(...), ...}
print(result.no_face_photos)    # 未检测到可用人脸的照片
print(result.summary())         # {"person_001": 3, "person_002": 1}
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `groups` | `Dict[str, PhotoGroup]` | 标签 → 分组，按首次出现顺序 |
| `faces` | `List[PhotoFace]` | 所有照片中的全部可用人脸 |
| `no_face_photos` | `List[str]` | 未检测到可用人脸的照片 |
| `unreadable_photos` | `List[str]` | 解码失败的照片 |
| `total_photos` | `int` | 恒等于 `len(images)` |

### 计算属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `labels` | `List[str]` | 全部分组标签，按首次出现顺序 |
| `total_faces` | `int` | `len(faces)` |

### 方法

| 方法 | 返回 | 说明 |
|------|------|------|
| `photos_of(label)` | `List[str]` | 某人 / 某分组的照片 id（标签不存在时为空） |
| `labels_of(photo_id)` | `List[str]` | 一张照片中发现的全部标签 |
| `summary()` | `Dict[str, int]` | 标签 → 照片数，附 `__no_face__` / `__unreadable__` 条目 |

---

## 常量

```python
from face_hub.types import UNKNOWN_SENTINEL

# 当人脸未匹配到任何已注册人员时，返回此哨兵值
UNKNOWN_SENTINEL = "unknown"
```

---

## 属性使用示例

```python
import cv2
import numpy as np
from face_hub import BBox, DetectionResult, DetectionWithEmbedding, TrackedFace, PipelineResult
from face_hub.types import UNKNOWN_SENTINEL

# === BBox ===
bbox = BBox(x1=50, y1=30, x2=250, y2=300)
print(f"中心: {bbox.center}")       # (150, 165)
print(f"尺寸: {bbox.width}x{bbox.height}")  # 200x270
print(f"面积: {bbox.area}")         # 54000

# === DetectionResult ===
det = DetectionResult(bbox=bbox, confidence=0.95)
print(f"检测置信度: {det.confidence:.2f}")

# === DetectionWithEmbedding ===
embedding = np.random.randn(512).astype(np.float32)
embedding = embedding / np.linalg.norm(embedding)  # L2 归一化
det_emb = DetectionWithEmbedding(
    bbox=bbox, confidence=0.95,
    embedding=embedding, quality_pass=True,
)
print(f"有特征: {det_emb.has_embedding}")   # True
print(f"通过质量: {det_emb.quality_pass}")  # True

# === TrackedFace ===
tracked = TrackedFace(
    track_id=1, bbox=bbox,
    name="Alice", confidence=0.85,
    is_confirmed=True,
)
print(f"已确认: {tracked.is_confirmed}")  # True
print(f"已识别: {tracked.is_known}")      # True

# 未识别的追踪
unknown_tracked = TrackedFace(
    track_id=2, bbox=bbox,
    name=UNKNOWN_SENTINEL, confidence=0.0,
    is_confirmed=True,
)
print(f"已识别: {unknown_tracked.is_known}")  # False

# === PipelineResult ===
result = PipelineResult(
    frame=np.zeros((360, 640, 3), dtype=np.uint8),
    tracked_faces=[tracked, unknown_tracked],
    fps=25.0,
)
print(f"总人脸: {result.total_faces}")       # 2
print(f"已识别: {result.known_faces}")       # [TrackedFace(track_id=1, name='Alice')]
print(f"未识别: {result.unknown_count}")      # 1
print(f"FPS: {result.fps}")                 # 25.0
```

## 在应用中使用类型的完整示例

```python
import cv2
from face_hub import (
    FaceHubPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread,
)
from face_hub.types import UNKNOWN_SENTINEL

pipeline = FaceHubPipeline(
    CameraThread(0), FaceDetector(device="cpu"),
    FaceRecognizer(), FaceTracker(), FaceDatabase(),
)
pipeline.start()

while True:
    result = pipeline.process_frame()
    if result is None:
        continue

    # 使用 PipelineResult 的属性
    frame = result.frame
    print(f"FPS: {result.fps:.1f} | "
          f"总计: {result.total_faces} | "
          f"已识别: {len(result.known_faces)} | "
          f"未知: {result.unknown_count}")

    # 遍历人脸
    for face in result.tracked_faces:
        x1, y1, x2, y2 = face.bbox.to_tuple()

        # 根据状态选择颜色
        if face.is_known:
            color = (0, 255, 0)        # 已确认 + 已识别
            label = f"{face.name} {face.confidence:.0%}"
        elif face.is_confirmed:
            color = (0, 165, 255)      # 已确认但未识别
            label = UNKNOWN_SENTINEL
        else:
            color = (0, 0, 255)        # 未确认
            label = face.name

        # 绘制
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 使用 BBox 属性
        cv2.circle(frame, face.bbox.center, 3, color, -1)

    cv2.imshow("FaceHub", frame)
    if cv2.waitKey(1) == 27:
        break

pipeline.stop()
```

## 类型关系图

```
DetectionResult        (仅边框 + 置信度)
DetectionWithEmbedding (边框 + 置信度 + 特征向量)
         │
         ▼
    FaceTracker.update()
         │
         ▼
    TrackedFace         (边框 + 追踪ID + 姓名 + 置信度 + 确认状态)
         │
         ▼ 聚合到
    PipelineResult      (帧 + 所有追踪 + FPS)
```

## 何时使用哪种类型

| 你想做什么 | 使用的方法 | 返回类型 |
|-----------|-----------|---------|
| 只要知道人脸在哪 | `detector.detect()` / `pipeline.detect_only()` | `List[DetectionResult]` |
| 需要特征向量（注册/离线比对） | `detector.detect_with_embeddings()` / `pipeline.extract_embeddings()` | `List[DetectionWithEmbedding]` |
| 实时视频流中追踪+识别 | `tracker.update()` 或 `pipeline.process_frame()` | `List[TrackedFace]` |
| 获取完整的一帧结果 | `pipeline.process_frame()` | `PipelineResult` |
| 按人脸对照片集分组 | `classify_photos()` / `PhotoClassifier.classify_photos()` | `PhotoClassificationResult` |
