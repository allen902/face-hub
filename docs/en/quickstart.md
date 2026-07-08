# Quick Start

## Install

```bash
pip install facevision
```

## 5-Minute Example

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

## Register a Person

```python
import numpy as np

# In practice obtain the embedding from detector.detect_with_embeddings(frame)
encoding = np.random.randn(512).astype(np.float32)
db.add_person("Alice", "/path/to/photo.jpg", encoding)
```

## Custom Detector

Implement `DetectorProtocol` to plug in your own model. See
[Custom Detector Example](examples/custom_detector.md).
