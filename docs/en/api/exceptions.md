# Exceptions

All FaceHub exceptions inherit from `FaceHubError`, making it easy to catch
any library error with a single handler.

---

## Exception Hierarchy

```
FaceHubError (base)
 ├── ModelLoadError      — model download / ONNX provider / corrupt model
 ├── InferenceError      — ML runtime error (GPU crash + CPU fallback failed)
 ├── CameraError         — camera not connected / in use / unsupported resolution
 ├── DatabaseError       — JSON parse / pickle corrupt / disk full / permission
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

```python
from face_hub import FaceDetector
from face_hub.exceptions import ModelLoadError

try:
    detector = FaceDetector(device="cuda")
except ModelLoadError as e:
    print(f"Model load failed: {e}")
    # Possible causes:
    #   1. Network issue downloading buffalo_l (~200 MB)
    #   2. CUDA driver not installed
    #   3. Insufficient disk space
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

Raised on database read/write failures — JSON parse errors, corrupt pickle
files, disk full, or permission issues.

```python
from face_hub import FaceDatabase
from face_hub.exceptions import DatabaseError

try:
    db = FaceDatabase(db_path="/read_only/face_db.json")
except DatabaseError as e:
    print(f"Database error: {e}")
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
