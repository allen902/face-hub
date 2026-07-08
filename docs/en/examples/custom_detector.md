# Example: Custom Detector

Implement `DetectorProtocol` to plug YOLO, MediaPipe, or any other model into the
FaceVision pipeline.

```python
import numpy as np
from face_vision import (
    DetectorProtocol, DetectionWithEmbedding, BBox,
    FaceVisionPipeline, FaceRecognizer, FaceTracker, FaceDatabase, CameraThread,
)


class MyYoloDetector:
    """YOLO face detection + a custom embedding model."""

    def __init__(self):
        self.yolo_model = ...       # load your detector
        self.embedder = ...         # load your embedding model

    def detect_with_embeddings(self, frame):
        boxes = self.yolo_model(frame)
        results = []
        for box in boxes:
            roi = frame[box.y1:box.y2, box.x1:box.x2]
            emb = self.embedder(roi)
            results.append(DetectionWithEmbedding(
                bbox=BBox(box.x1, box.y1, box.x2, box.y2),
                confidence=box.conf,
                embedding=emb,
                quality_pass=True,
            ))
        return results

    def detect(self, frame):
        boxes = self.yolo_model(frame)
        return [
            DetectionResult(
                bbox=BBox(box.x1, box.y1, box.x2, box.y2),
                confidence=box.conf,
            )
            for box in boxes
        ]


# Use the custom detector
camera = CameraThread()
db = FaceDatabase()
pipeline = FaceVisionPipeline(
    camera,
    MyYoloDetector(),
    FaceRecognizer(),
    FaceTracker(),
    db,
)
pipeline.start()
```

Any object implementing `detect_with_embeddings(frame)` automatically satisfies
the protocol without explicit inheritance.
