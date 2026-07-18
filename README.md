

# FaceHub
### 🌐 [English](#facehub) | [中文](#facehub中文)



> Real-time face recognition library — detection, embedding, tracking, and matching.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/face-hub.svg)](https://pypi.org/project/face-hub/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://allen902.github.io/face-hub/)
[![Tests](https://github.com/allen902/face-hub/actions/workflows/publish.yml/badge.svg)](https://github.com/allen902/face-hub/actions/workflows/publish.yml)

## Features

- **Detection**: insightface RetinaFace with GPU auto-detection (CUDA / DirectML) and CPU fallback.
- **Embedding**: ArcFace 512-dim L2-normalized features.
- **Recognition**: 1:N cosine-similarity matching with a versioned encoding cache.
- **Tracking**: IoU-based multi-face tracker with majority-vote identity smoothing.
- **Camera**: cross-platform capture thread (Windows DShow, macOS AVFoundation, Linux V4L2).
- **Protocol**: `DetectorProtocol` lets you plug in your own detector (YOLO, MediaPipe, etc.).
- **Photo classification**: group photo collections by the faces in them — gallery matching or fully automatic clustering.

## Installation

```bash
pip install face-hub
```

Optional GPU backends:

```bash
# Windows DirectML
pip uninstall -y onnxruntime
pip install face-hub[gpu-win]

# Linux NVIDIA CUDA
pip uninstall -y onnxruntime
pip install face-hub[gpu-linux]
```

## Quick Start

```python
from face_hub import (
    FaceHubPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread,
)

# 1. Initialize components
db = FaceDatabase(db_path="face_db.json")
detector = FaceDetector(device="auto", det_size=640)
recognizer = FaceRecognizer(tolerance=0.45)
tracker = FaceTracker(smooth_frames=5)
camera = CameraThread(camera_id=0, width=640, height=360)

# 2. Assemble the pipeline
pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
pipeline.start()

# 3. Loop
try:
    while True:
        result = pipeline.process_frame()
        if result is None:
            continue
        for face in result.known_faces:
            print(f"{face.name} ({face.confidence:.0%})")
finally:
    pipeline.stop()
```

## Custom Detector

Any object satisfying `DetectorProtocol` can be plugged into the pipeline:

```python
from face_hub import DetectorProtocol, DetectionWithEmbedding, BBox

class MyYoloDetector:
    def detect_with_embeddings(self, frame):
        boxes = self.yolo_model(frame)
        return [
            DetectionWithEmbedding(
                bbox=BBox(x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2),
                confidence=b.conf,
                embedding=self.embedder(frame[b.y1:b.y2, b.x1:b.x2]),
                quality_pass=True,
            )
            for b in boxes
        ]

pipeline = FaceHubPipeline(camera, MyYoloDetector(), recognizer, tracker, db)
```

## Photo Classification by Face

Group a folder of photos by the people in them. Without a gallery, faces are
clustered automatically into anonymous groups (`person_001`, …); with a
registered gallery, known faces are filed under their names and strangers are
still clustered. A photo containing several people appears in several groups.

```python
from face_hub import classify_photos

result = classify_photos(["party1.jpg", "party2.jpg", "party3.jpg"])

for label, group in result.groups.items():
    print(label, "→", group.photo_ids)

print(result.no_face_photos)   # photos with no usable face
print(result.summary())        # {"person_001": 2, ...}
```

With a registered gallery (known people filed under their names):

```python
from face_hub import FaceDetector, FaceRecognizer, PhotoClassifier

detector = FaceDetector(device="auto")
recognizer = FaceRecognizer(tolerance=0.45)
recognizer.update_cache(known_encodings, known_names, db_version=1)

classifier = PhotoClassifier(detector, recognizer=recognizer, cluster_threshold=0.45)
result = classifier.classify_photos(photos, progress_callback=lambda d, t, p: print(f"{d}/{t}"))
```

## Documentation

📖 **[Online Documentation](https://allen902.github.io/face-hub/)** — Full API reference in English & 中文

To preview locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve -f docs/mkdocs.yml
```
## Download Quantity
[![Monthly Downloads](https://static.pepy.tech/badge/face-hub/month)](https://pepy.tech/project/face-hub)

[![Total Downloads](https://static.pepy.tech/badge/face-hub)](https://pepy.tech/project/face-hub)

## Acknowledgements
We would like to express our sincere gratitude to **Leon Jane** for voluntarily providing his facial sample data and fully participating in the verification and testing of all functions of the **Face-hub** library. Many program bugs were successfully identified through his efforts, making a crucial contribution to feature improvement and stability optimization of this project.
## License

The FaceHub **code** is released under the [MIT License](LICENSE).

> ⚠️ The pre-trained `buffalo_l` model downloaded automatically by insightface is subject to insightface's own model license and is for non-commercial research use unless separate authorization is obtained. See the documentation for details.

---


# FaceHub（中文）
<div align="right">

### 🌐 [English](#facehub) | [中文](#facehub中文)

</div>


> 实时人脸识别库 — 检测、特征提取、追踪、匹配。

## 特性

- **检测**：insightface RetinaFace，自动检测 CUDA / DirectML GPU 并回退 CPU。
- **特征**：ArcFace 512 维 L2 归一化特征向量。
- **识别**：1:N 余弦相似度匹配，带版本号缓存。
- **追踪**：基于 IoU 的多目标追踪 + 多数投票身份平滑。
- **摄像头**：跨平台采集线程（Windows DShow、macOS AVFoundation、Linux V4L2）。
- **协议**：`DetectorProtocol` 允许接入自定义检测器（YOLO、MediaPipe 等）。
- **照片分类**：按人脸对照片集自动分组 —— 支持注册人脸库匹配或全自动聚类。

## 安装

```bash
pip install face-hub
```

可选 GPU 后端：

```bash
# Windows DirectML
pip uninstall -y onnxruntime
pip install face-hub[gpu-win]

# Linux NVIDIA CUDA
pip uninstall -y onnxruntime
pip install face-hub[gpu-linux]
```

## 快速开始

```python
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
try:
    while True:
        result = pipeline.process_frame()
        if result is None:
            continue
        for face in result.known_faces:
            print(f"{face.name} ({face.confidence:.0%})")
finally:
    pipeline.stop()
```

## 自定义检测器

任何满足 `DetectorProtocol` 的对象都可以接入流水线，示例见上文英文部分。

## 按人脸分类照片

把一个文件夹的照片按其中的人物自动分组。无人脸库时，人脸会按特征相似度
自动聚类为匿名分组（`person_001`……）；提供注册人脸库时，认识的人直接归入
其姓名分组，陌生人仍会单独聚类。包含多人的照片会同时出现在多个分组中。

```python
from face_hub import classify_photos

result = classify_photos(["聚会1.jpg", "聚会2.jpg", "聚会3.jpg"])

for label, group in result.groups.items():
    print(label, "→", group.photo_ids)

print(result.no_face_photos)   # 未检测到可用人脸的照片
print(result.summary())        # {"person_001": 2, ...}
```

使用注册人脸库（认识的人归入姓名分组）：

```python
from face_hub import FaceDetector, FaceRecognizer, PhotoClassifier

detector = FaceDetector(device="auto")
recognizer = FaceRecognizer(tolerance=0.45)
recognizer.update_cache(known_encodings, known_names, db_version=1)

classifier = PhotoClassifier(detector, recognizer=recognizer, cluster_threshold=0.45)
result = classifier.classify_photos(photos, progress_callback=lambda d, t, p: print(f"{d}/{t}"))
```

## 文档

📖 **[在线文档](https://allen902.github.io/face-hub/)** — 完整中英 API 文档

## 下载量
[![Monthly Downloads](https://static.pepy.tech/badge/face-hub/month)](https://pepy.tech/project/face-hub)

[![Total Downloads](https://static.pepy.tech/badge/face-hub)](https://pepy.tech/project/face-hub)

## 致谢
在此特别向 **Leon Jane** 致以诚挚谢意。其无偿提供其人脸样本数据，并完整参与 **Face-hub** 库各项功能的验证测试，有效排查多处程序缺陷，为本项目的功能完善与稳定性优化作出关键贡献。
## 许可

FaceHub **代码** 采用 [MIT License](LICENSE)。

> ⚠️ insightface 自动下载的 `buffalo_l` 预训练模型受其模型许可约束，默认仅供非商用研究使用；商业使用需单独获取授权。详见文档。
