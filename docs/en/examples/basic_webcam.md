# Example: Basic Webcam Recognition

Open the webcam and recognize registered people in real time.

```python
from face_hub import (
    FaceHubPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread,
)

# Initialize
db = FaceDatabase(db_path="face_db.json")
detector = FaceDetector(device="auto", det_size=640)
recognizer = FaceRecognizer(tolerance=0.45)
tracker = FaceTracker(smooth_frames=5)
camera = CameraThread(camera_id=0, width=640, height=360)

# Assemble pipeline
pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
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

## Register a Person

```python
import numpy as np

# In practice get the embedding from detector.detect_with_embeddings(frame)
encoding = np.random.randn(512).astype(np.float32)
db.add_person("Alice", "photos/alice.jpg", encoding)
```
