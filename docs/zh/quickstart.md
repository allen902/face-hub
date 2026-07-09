# 快速开始

## 安装

```bash
pip install face-hub
```

## 5 分钟示例

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

## 注册人员

```python
import numpy as np

encoding = np.random.randn(512).astype(np.float32)  # 实际应来自 detector
db.add_person("Alice", "/path/to/photo.jpg", encoding)
```

## 自定义检测器

实现 `DetectorProtocol` 即可接入自有模型，详见 [自定义检测器示例](examples/custom_detector.md)。
