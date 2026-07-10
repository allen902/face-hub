# FaceDetector

基于 [insightface](https://github.com/deepinsight/insightface) RetinaFace 的人脸检测器。它能检测人脸、提取 512 维 ArcFace 特征向量，支持 GPU 自动检测（CUDA / DirectML）并带 CPU 回退，还内置了模糊度和尺寸质量过滤器。

---

## 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `confidence` | `float` | `0.50` | 检测置信度阈值 (0.0~1.0)，低于此值的人脸被丢弃 |
| `device` | `str` | `"auto"` | 推理设备：`"cpu"` / `"cuda"` / `"auto"`。`"auto"` 自动探测最优 GPU：CUDA → DirectML → CPU |
| `det_size` | `int` | `640` | 检测模型输入尺寸：320(快) / 480(均衡) / 640(精准) |
| `quality_filter` | `bool` | `True` | 是否启用基于模糊度的质量过滤 |
| `min_face_size` | `int` | `80` | 最小人脸尺寸 (px)，小于此值标记为低质量 |

## 方法

### detect(frame)

仅检测人脸，不提取特征向量。

**参数:**
- `frame` (`np.ndarray`): BGR 图像 (H, W, 3)

**返回:**
- `List[DetectionResult]`：按置信度从高到低排序。

### detect_with_embeddings(frame)

检测人脸并提取 512 维特征向量。

**参数:**
- `frame` (`np.ndarray`): BGR 图像 (H, W, 3)

**返回:**
- `List[DetectionWithEmbedding]`：按置信度从高到低排序。

### reload_model(det_size=None)

运行时重新加载检测模型，可切换输入尺寸。

**参数:**
- `det_size` (`int | None`): 新尺寸（320/480/640），`None` 表示保持当前值。

---

## 基础检测示例

```python
import cv2
from face_hub import FaceDetector

detector = FaceDetector(device="auto", det_size=640)
frame = cv2.imread("photo.jpg")

results = detector.detect(frame)
print(f"检测到 {len(results)} 张人脸")

for face in results:
    x1, y1, x2, y2 = face.bbox.to_tuple()
    print(f"  人脸框: ({x1},{y1},{x2},{y2}), 置信度={face.confidence:.2f}")
```

## 检测 + 特征提取

特征提取是进行人脸识别的前提，提取到的 512 维向量可用于注册或比对：

```python
faces = detector.detect_with_embeddings(frame)

for face in faces:
    print(f"置信度: {face.confidence:.2f}")
    print(f"质量过滤: {'通过' if face.quality_pass else '未通过'}")

    if face.has_embedding:
        print(f"特征维度: {face.embedding.shape}")  # (512,)
        print(f"数据类型: {face.embedding.dtype}")    # float32
        print(f"L2 范数: {np.linalg.norm(face.embedding):.4f}")  # ≈ 1.0
```

## 从照片注册新用户

这是最常见的用法：读取照片 → 检测人脸 → 提取特征 → 存入数据库：

```python
import cv2
import numpy as np
from face_hub import FaceDetector, FaceDatabase

detector = FaceDetector(device="cpu")
db = FaceDatabase()

frame = cv2.imread("alice.jpg")

# 检测并提取特征
faces = detector.detect_with_embeddings(frame)

if not faces:
    raise ValueError("未检测到人脸，请使用清晰、正面的人脸照片")

# 取置信度最高的人脸
face = faces[0]

if not face.has_embedding:
    raise ValueError("特征提取失败")

# 存入数据库
ok, msg = db.add_person("Alice", "alice.jpg", face.embedding)
if ok:
    print("Alice 注册成功！")
else:
    print(f"注册失败: {msg}")
```

## 运行时切换检测精度

根据场景动态切换速度和精度：

```python
# 常规场景：用快速模式
detector = FaceDetector(det_size=320)
fast_faces = detector.detect_with_embeddings(frame)

# 遇到重要场景：临时切换到高精度模式
detector.reload_model(det_size=640)
accurate_faces = detector.detect_with_embeddings(frame)

# 切回快速模式
detector.reload_model(det_size=320)
```

## 设备选择策略

```python
from face_hub import FaceDetector

# 自动探测最佳 GPU（推荐）
detector = FaceDetector(device="auto")

# 强制使用 CPU（适用于无 GPU 服务器或调试）
detector_cpu = FaceDetector(device="cpu")

# 尝试 GPU，失败后自动回退到 CPU
detector_gpu = FaceDetector(device="cuda")
```

## GPU 回退与优雅降级

```python
from face_hub import FaceDetector
from face_hub.exceptions import ModelLoadError, InferenceError

try:
    # 先尝试 GPU
    detector = FaceDetector(device="cuda")
    print("✅ 使用 GPU 推理")
except (ModelLoadError, InferenceError):
    # GPU 不可用，回退到 CPU
    print("⚠️ GPU 不可用，回退到 CPU")
    detector = FaceDetector(device="cpu")
```

## 质量过滤配置

质量过滤可以自动跳过模糊或太小的人脸，减少误识别：

```python
# 严格模式：仅保留高清、大尺寸人脸
detector_strict = FaceDetector(
    confidence=0.60,        # 更高的检测阈值
    min_face_size=120,      # 最小人脸 120px
    quality_filter=True,    # 启用模糊度过滤
)

# 宽松模式：尽量多检测（适合人群计数等场景）
detector_loose = FaceDetector(
    confidence=0.35,
    min_face_size=40,
    quality_filter=False,   # 关闭模糊度过滤
)

# 检查质量过滤结果
for face in detector_strict.detect_with_embeddings(frame):
    if face.quality_pass:
        print("✅ 优质人脸")
    else:
        print("❌ 人脸太小或太模糊，已跳过")
```

## 多张人脸处理的完整流程

```python
import cv2
from face_hub import FaceDetector, FaceDatabase, FaceRecognizer

detector = FaceDetector(device="auto")
db = FaceDatabase()
recognizer = FaceRecognizer()

# 同步缓存
encodings, names = db.get_encodings_and_names()
recognizer.update_cache(encodings, names, db.version)

frame = cv2.imread("group_photo.jpg")
faces = detector.detect_with_embeddings(frame)

for i, face in enumerate(faces):
    x1, y1, x2, y2 = face.bbox.to_tuple()

    # 绘制边框
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 识别
    if face.has_embedding:
        name, conf = recognizer.recognize(face.embedding)
        if conf > 0:
            label = f"{name} ({conf:.0%})"
        else:
            label = f"unknown"
    else:
        label = "no embedding"

    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("result.jpg", frame)
print(f"处理完成，共检测到 {len(faces)} 张人脸")
```

## 自定义检测器

实现 `DetectorProtocol` 接口即可接入 YOLO、MediaPipe 等任意模型：

```python
from typing import List
import numpy as np
from face_hub import DetectorProtocol, DetectionWithEmbedding, DetectionResult, BBox
from face_hub import FaceHubPipeline, FaceRecognizer, FaceTracker, FaceDatabase, CameraThread

class MyYoloDetector:
    """YOLO 人脸检测 + 自定义特征提取模型"""

    def __init__(self):
        # 加载你的模型
        self.yolo_model = ...      # 你的检测模型
        self.embedder = ...        # 你的特征提取模型

    def detect_with_embeddings(self, frame: np.ndarray) -> List[DetectionWithEmbedding]:
        """检测人脸并提取特征（流水线调用此方法）"""
        boxes = self.yolo_model(frame)
        results = []
        for box in boxes:
            # 裁剪人脸区域
            roi = frame[box.y1:box.y2, box.x1:box.x2]
            # 提取特征
            embedding = self.embedder(roi)
            results.append(DetectionWithEmbedding(
                bbox=BBox(box.x1, box.y1, box.x2, box.y2),
                confidence=box.conf,
                embedding=embedding,
                quality_pass=True,
            ))
        return results

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """仅检测（detect_only 会调用此方法）"""
        boxes = self.yolo_model(frame)
        return [
            DetectionResult(
                bbox=BBox(box.x1, box.y1, box.x2, box.y2),
                confidence=box.conf,
            )
            for box in boxes
        ]

# 使用自定义检测器
camera = CameraThread()
db = FaceDatabase()
pipeline = FaceHubPipeline(
    camera,
    MyYoloDetector(),
    FaceRecognizer(),
    FaceTracker(),
    db,
)
pipeline.start()
```

只要对象实现了 `detect_with_embeddings(frame)` 方法，就自动满足协议，无需显式继承。

## 性能参考

| det_size | 特点 | 适用场景 |
|----------|------|----------|
| 320 | 速度快，精度较低 | 实时预览、低端设备 |
| 480 | 速度与精度平衡 | 一般使用推荐 |
| 640 | 精度最高，速度最慢 | 注册人脸、离线处理 |

## 注意事项

- 首次调用 `FaceDetector()` 会自动下载 insightface 的 `buffalo_l` 模型（约 200 MB），请确保网络畅通。
- 提取的特征向量是 L2 归一化的 512 维 `float32` 向量，适合余弦相似度匹配。
- `detect()` 和 `detect_with_embeddings()` 返回的结果均按置信度从高到低排序。
- `det_size=640` 时，模型输入会被调整为 640x640，检测精度最高，但帧率最低。