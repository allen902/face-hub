# FaceVision

> Real-time face recognition library — detection, embedding, tracking, and matching.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/allen902/face_vision/actions/workflows/publish.yml/badge.svg)](https://github.com/allen902/face_vision/actions/workflows/publish.yml)

## Features

- **Detection**: insightface RetinaFace with GPU auto-detection (CUDA / DirectML) and CPU fallback.
- **Embedding**: ArcFace 512-dim L2-normalized features.
- **Recognition**: 1:N cosine-similarity matching with a versioned encoding cache.
- **Tracking**: IoU-based multi-face tracker with majority-vote identity smoothing.
- **Camera**: cross-platform capture thread (Windows DShow, macOS AVFoundation, Linux V4L2).
- **Protocol**: `DetectorProtocol` lets you plug in your own detector (YOLO, MediaPipe, etc.).

## Installation

```bash
pip install face-vision
```

Optional GPU backends:

```bash
# Windows DirectML
pip uninstall -y onnxruntime
pip install face-vision[gpu-win]

# Linux NVIDIA CUDA
pip uninstall -y onnxruntime
pip install face-vision[gpu-linux]
```

## Quick Start

```python
from face_vision import (
    FaceVisionPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread,
)

# 1. Initialize components
db = FaceDatabase(db_path="face_db.json")
detector = FaceDetector(device="auto", det_size=640)
recognizer = FaceRecognizer(tolerance=0.45)
tracker = FaceTracker(smooth_frames=5)
camera = CameraThread(camera_id=0, width=640, height=360)

# 2. Assemble the pipeline
pipeline = FaceVisionPipeline(camera, detector, recognizer, tracker, db)
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
from face_vision import DetectorProtocol, DetectionWithEmbedding, BBox

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

pipeline = FaceVisionPipeline(camera, MyYoloDetector(), recognizer, tracker, db)
```

## Documentation

Full API docs (English / 中文) are in the `docs/` directory and can be served with:

```bash
pip install mkdocs mkdocs-material
cd docs && mkdocs serve
```

## License

The FaceVision **code** is released under the [MIT License](LICENSE).

> ⚠️ The pre-trained `buffalo_l` model downloaded automatically by insightface is subject to insightface's own model license and is for non-commercial research use unless separate authorization is obtained. See the documentation for details.

---

# FaceVision（中文）

> 实时人脸识别库 — 检测、特征提取、追踪、匹配。

## 特性

- **检测**：insightface RetinaFace，自动检测 CUDA / DirectML GPU 并回退 CPU。
- **特征**：ArcFace 512 维 L2 归一化特征向量。
- **识别**：1:N 余弦相似度匹配，带版本号缓存。
- **追踪**：基于 IoU 的多目标追踪 + 多数投票身份平滑。
- **摄像头**：跨平台采集线程（Windows DShow、macOS AVFoundation、Linux V4L2）。
- **协议**：`DetectorProtocol` 允许接入自定义检测器（YOLO、MediaPipe 等）。

## 安装

```bash
pip install face-vision
```

可选 GPU 后端：

```bash
# Windows DirectML
pip uninstall -y onnxruntime
pip install face-vision[gpu-win]

# Linux NVIDIA CUDA
pip uninstall -y onnxruntime
pip install face-vision[gpu-linux]
```

## 快速开始

```python
from face_vision import (
    FaceVisionPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread,
)

# 1. 初始化组件
db = FaceDatabase(db_path="face_db.json")
detector = FaceDetector(device="auto", det_size=640)
recognizer = FaceRecognizer(tolerance=0.45)
tracker = FaceTracker(smooth_frames=5)
camera = CameraThread(camera_id=0, width=640, height=360)

# 2. 组装流水线
pipeline = FaceVisionPipeline(camera, detector, recognizer, tracker, db)
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

## 文档

完整中英 API 文档位于 `docs/` 目录，可通过 MkDocs 本地预览。

## 许可

FaceVision **代码** 采用 [MIT License](LICENSE)。

> ⚠️ insightface 自动下载的 `buffalo_l` 预训练模型受其模型许可约束，默认仅供非商用研究使用；商业使用需单独获取授权。详见文档。
