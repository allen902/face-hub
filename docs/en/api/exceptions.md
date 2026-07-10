# Exceptions

FaceHub uses a custom exception hierarchy. All library exceptions inherit from
`FaceHubError`, so you can catch every FaceHub-specific problem with a single
`except` block, or handle individual error types separately.

---

## Hierarchy

```
FaceHubError
├── ModelLoadError
├── InferenceError
├── CameraError
├── DatabaseError
└── RecognitionError
```

---

## Reference

| Exception | Trigger |
|-----------|---------|
| `ModelLoadError` | insightface is not installed, no ONNX provider available, or the model file is corrupt |
| `InferenceError` | GPU inference failed and CPU fallback also failed |
| `CameraError` | Cannot open camera, camera is in use, or the requested resolution is unsupported |
| `DatabaseError` | SQLite error (disk full, permission denied, corrupt database file) |
| `RecognitionError` | Encoding dimension mismatch, empty cache, or other matching errors |

---

## Catch All FaceHub Errors

```python
from face_hub import FaceDetector
from face_hub.exceptions import FaceHubError

try:
    detector = FaceDetector(device="cuda")
except FaceHubError as e:
    print(f"FaceHub error: {e}")
```

## Handle Specific Errors

```python
from face_hub import FaceDetector, CameraThread
from face_hub.exceptions import (
    ModelLoadError,
    InferenceError,
    CameraError,
)

try:
    detector = FaceDetector(device="cuda")
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
except InferenceError as e:
    print(f"Inference failed on both GPU and CPU: {e}")

try:
    camera = CameraThread(camera_id=0, width=1920, height=1080)
    camera.start()
except CameraError as e:
    print(f"Camera problem: {e}")
    # Try a lower resolution or a different camera index
```

## Database Errors

```python
from face_hub import FaceDatabase
from face_hub.exceptions import DatabaseError

try:
    db = FaceDatabase("/read_only_volume/registry.db")
    db.add_person("Alice", "alice.jpg", alice_embedding)
except DatabaseError as e:
    print(f"Database error: {e}")
    # Recover by switching to a writable path
    db = FaceDatabase("/tmp/registry.db")
    db.add_person("Alice", "alice.jpg", alice_embedding)
```

## Corrupt Database Recovery

```python
import os
from face_hub import FaceDatabase
from face_hub.exceptions import DatabaseError

DB_PATH = "registry.db"

try:
    db = FaceDatabase(DB_PATH)
    persons = db.list_persons()
except DatabaseError:
    print(f"Database at {DB_PATH} is corrupt — re-initializing")
    # Backup the corrupt file, then start fresh
    if os.path.exists(DB_PATH):
        os.rename(DB_PATH, DB_PATH + ".corrupted")
    db = FaceDatabase(DB_PATH)
    # Now re-register from original photos
```

## Recognition Errors

```python
import numpy as np
from face_hub import FaceRecognizer
from face_hub.exceptions import RecognitionError

recognizer = FaceRecognizer()

# Wrong embedding dimension
try:
    recognizer.recognize(np.zeros(128, dtype=np.float32))
except RecognitionError as e:
    print(f"Recognition failed: {e}")
    # Embeddings must be 512-dim float32

# Empty cache recognition
recognizer = FaceRecognizer()
try:
    name, conf = recognizer.recognize(query_embedding)
except RecognitionError as e:
    print(f"No gallery loaded: {e}")
    # Load gallery first: recognizer.update_cache(...)
```

## Pipeline-Level Error Handling

```python
from face_hub import FaceHubPipeline
from face_hub.exceptions import FaceHubError

pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
pipeline.start()

try:
    while True:
        try:
            result = pipeline.process_frame()
        except FaceHubError as e:
            print(f"Pipeline error: {e}")
            continue

        if result is None:
            continue

        # ... render ...
finally:
    pipeline.stop()
```

## Full Production-Grade Error Handling

```python
from face_hub import (
    FaceHubPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread,
)
from face_hub.exceptions import (
    FaceHubError, ModelLoadError, CameraError,
    DatabaseError, InferenceError, RecognitionError,
)
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_pipeline():
    """Create pipeline with graceful fallbacks."""
    # Detector: try GPU, fall back to CPU
    try:
        detector = FaceDetector(device="auto")
    except ModelLoadError:
        logger.warning("Model failed to load on GPU, trying CPU...")
        try:
            detector = FaceDetector(device="cpu")
        except ModelLoadError as e:
            logger.error(f"Cannot load model: {e}")
            sys.exit(1)

    # Camera: try multiple indices
    camera = None
    for cam_id in [0, 1]:
        try:
            camera = CameraThread(camera_id=cam_id, width=640, height=360)
            camera.start()
            logger.info(f"Camera {cam_id} opened successfully")
            break
        except CameraError as e:
            logger.warning(f"Camera {cam_id} failed: {e}")
    if camera is None:
        logger.error("No camera available")
        sys.exit(1)

    # Database
    try:
        db = FaceDatabase("registry.db")
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        sys.exit(1)

    recognizer = FaceRecognizer(tolerance=0.45)
    tracker = FaceTracker(smooth_frames=5)

    # Load gallery
    encodings, names = db.get_encodings_and_names()
    recognizer.update_cache(encodings, names, db.version)

    return FaceHubPipeline(camera, detector, recognizer, tracker, db)

# Run
try:
    pipeline = create_pipeline()
    pipeline.start()
    while True:
        try:
            result = pipeline.process_frame()
        except InferenceError as e:
            logger.error(f"Inference error: {e} — frame dropped")
            continue
        except RecognitionError as e:
            logger.error(f"Recognition error: {e} — frame dropped")
            continue
        except FaceHubError as e:
            logger.error(f"Unexpected FaceHub error: {e}")
            break

        if result is None:
            continue
        # ... render ...
finally:
    if 'pipeline' in locals():
        pipeline.stop()
```

## Graceful Degradation

A common pattern is to fall back to CPU when GPU inference fails:

```python
from face_hub import FaceDetector
from face_hub.exceptions import InferenceError

try:
    detector = FaceDetector(device="cuda")
except InferenceError:
    print("GPU inference failed, falling back to CPU")
    detector = FaceDetector(device="cpu")
```

## Error Message Propagation

All exceptions include a descriptive message. Access via `str(e)`:

```python
from face_hub.exceptions import CameraError

try:
    camera = CameraThread(camera_id=99)
    camera.start()
except CameraError as e:
    # e.g. "CameraError: Cannot open camera 99: device not found"
    error_message = str(e)
    print(f"UI Alert: {error_message}")
```

## Context Manager for Camera Lifecycle

```python
import cv2
from contextlib import contextmanager
from face_hub import CameraThread
from face_hub.exceptions import CameraError

@contextmanager
def open_camera(camera_id=0, width=640, height=360):
    camera = CameraThread(camera_id, width=width, height=height)
    try:
        camera.start()
        yield camera
    except CameraError as e:
        raise RuntimeError(f"Failed to start camera {camera_id}: {e}")
    finally:
        camera.stop()

# Usage
try:
    with open_camera(0) as cam:
        while True:
            frame = cam.read()
            if frame is not None:
                cv2.imshow("Preview", frame)
            if cv2.waitKey(1) == 27:
                break
except RuntimeError as e:
    print(e)
```

## Notes

- All built-in errors preserve the original exception via `raise ... from e`, so
  `__cause__` is available for debugging.
- `FaceHubError` is the base class; catching it does **not** catch standard
  Python exceptions such as `ValueError` or `KeyError`.
- Library code raises `FaceHubError` subclasses; user code should catch them
  near I/O boundaries (camera open, database load, model init).
- All exceptions include a human-readable message string, suitable for logging
  or displaying in a UI.
- `DatabaseError` covers any SQLite operation failure — disk full, permission
  denied, corrupt file, locked database.
- `InferenceError` is raised when both GPU and CPU inference fail. If only GPU
  fails and CPU succeeds, no exception is raised (silent fallback).