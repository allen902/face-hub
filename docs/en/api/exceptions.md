# Exceptions

All FaceHub exceptions inherit from `FaceHubError`, making it easy to catch
any library error with a single handler.

---

## Exception Hierarchy

```
FaceHubError (base)
 ├── ModelLoadError      — model download / ONNX provider / corrupt model
 │    └── DependencyError — missing Python package (insightface / onnxruntime)
 ├── InferenceError      — ML runtime error (GPU crash + CPU fallback failed)
 ├── CameraError         — camera not connected / in use / unsupported resolution
 ├── DatabaseError       — JSON parse / disk full / permission
 │    └── SerializationError — data format error (JSON parse / corrupt .npy)
 └── RecognitionError    — encoding dimension mismatch / empty cache
```

---

## FaceHubError

Base exception for all FaceHub errors. Catch this to handle any library
error uniformly.

```python
from face_hub.exceptions import FaceHubError

try:
    pipeline.process_frame()
except FaceHubError as e:
    print(f"FaceHub error: {e}")
```

---

## ModelLoadError

Raised when the model file cannot be downloaded, is corrupt, or the requested
ONNX execution provider is unavailable.

**Attributes:**
- `model_name` (`str | None`): Which model failed (e.g. `"buffalo_l"`, `"RetinaFace"`).
- `model_path` (`str | None`): File path that was attempted, if any.

```python
from face_hub import FaceDetector
from face_hub.exceptions import ModelLoadError

try:
    detector = FaceDetector(device="cuda")
except ModelLoadError as e:
    print(f"Model load failed: {e}")
    print(f"  model: {e.model_name}")
    # Possible causes:
    #   1. Network issue downloading buffalo_l (~200 MB)
    #   2. CUDA driver not installed
    #   3. Insufficient disk space
```

---

## DependencyError

Raised when a required Python package is not installed. Subclass of
`ModelLoadError`, so `except ModelLoadError` also catches it.

```python
from face_hub import FaceDetector
from face_hub.exceptions import DependencyError

try:
    detector = FaceDetector()
except DependencyError as e:
    print(f"Missing dependency: {e}")
    print(f"  package: {e.model_name}")
    # Install the missing package:
    #   pip install insightface
    #   pip install onnxruntime
```

---

## InferenceError

Raised during ML inference — e.g. GPU out-of-memory, invalid input tensor,
or ONNX runtime crash.

```python
from face_hub.exceptions import InferenceError

try:
    faces = detector.detect_with_embeddings(frame)
except InferenceError as e:
    print(f"Inference error: {e}")
    # Try reducing det_size or switching to CPU
```

---

## CameraError

Raised when the camera cannot be opened, read from, or configured.

```python
from face_hub import CameraThread
from face_hub.exceptions import CameraError

try:
    camera = CameraThread(camera_id=0)
    camera.start()
except CameraError as e:
    print(f"Camera error: {e}")
```

---

## DatabaseError

Raised on database read/write failures — disk full, permission issues, or file
locking errors.

**Attributes:**
- `db_path` (`str | None`): Path to the database file that failed.

```python
from face_hub import FaceDatabase
from face_hub.exceptions import DatabaseError

try:
    db = FaceDatabase(db_path="/read_only/face_db.json")
except DatabaseError as e:
    print(f"Database error: {e}")
    print(f"  path: {e.db_path}")
```

---

## SerializationError

Raised on data format errors — JSON parse failures, corrupt `.npy` encoding
files, or legacy pickle issues. Subclass of both `DatabaseError` and
`ValueError`, so it can be caught at either level.

```python
from face_hub import FaceDatabase
from face_hub.exceptions import SerializationError

try:
    db = FaceDatabase(db_path="corrupt.json")
except SerializationError as e:
    print(f"Data format error: {e}")
    # Possible causes:
    #   1. Corrupt JSON file
    #   2. Corrupt .npy encoding file
    #   3. Incompatible data format
```

---

## RecognitionError

Raised when recognition matching fails — encoding dimension mismatch or
empty recognition cache.

```python
from face_hub.exceptions import RecognitionError

try:
    result = pipeline.process_frame()
except RecognitionError as e:
    print(f"Recognition error: {e}")
```

---

## Unified Error Handling

In production, wrap each `process_frame()` call to prevent a single bad
frame from crashing the loop:

```python
from face_hub.exceptions import FaceHubError, InferenceError

while True:
    try:
        result = pipeline.process_frame()
        if result is None:
            continue
        # ... render / business logic ...

    except InferenceError as e:
        print(f"Inference error (skip frame): {e}")
        continue
    except FaceHubError as e:
        print(f"Runtime error (retry): {e}")
        continue
```

---

## GPU Fallback Pattern

```python
from face_hub import FaceDetector
from face_hub.exceptions import ModelLoadError

def create_detector():
    for device in ["cuda", "cpu"]:
        try:
            detector = FaceDetector(device=device)
            print(f"Using device: {device}")
            return detector
        except ModelLoadError:
            continue
    raise SystemExit("No available device")
```

---

## Camera Auto-Discovery

```python
from face_hub import CameraThread
from face_hub.exceptions import CameraError

def open_first_camera():
    for cam_id in range(5):
        try:
            camera = CameraThread(camera_id=cam_id)
            camera.start()
            return camera
        except CameraError:
            continue
    raise SystemExit("No camera found")
```

---

## Notes

- All exceptions can be caught via `from face_hub.exceptions import FaceHubError`.
- Subclass exceptions provide finer-grained handling.
- `ModelLoadError` and `InferenceError` occur during model/inference phases;
  others are related to application logic.
- In production, wrap every `process_frame()` with `try/except FaceHubError`
  to prevent a single error from stopping the entire pipeline.
