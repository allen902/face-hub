# 示例：基础摄像头识别

打开摄像头并实时识别已注册人员。

```python
from face_vision import (
    FaceVisionPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread,
)

# 初始化
db = FaceDatabase(db_path="face_db.json")
detector = FaceDetector(device="auto", det_size=640)
recognizer = FaceRecognizer(tolerance=0.45)
tracker = FaceTracker(smooth_frames=5)
camera = CameraThread(camera_id=0, width=640, height=360)

# 组装流水线
pipeline = FaceVisionPipeline(camera, detector, recognizer, tracker, db)
pipeline.start()

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

# 实际使用时从 detector.detect_with_embeddings(frame) 获取 encoding
encoding = np.random.randn(512).astype(np.float32)
db.add_person("Alice", "photos/alice.jpg", encoding)
```
