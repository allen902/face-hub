# Exceptions

FaceVision uses a custom exception hierarchy. All exceptions inherit from
`FaceVisionError`.

## Hierarchy

```
FaceVisionError
├── ModelLoadError
├── InferenceError
├── CameraError
├── DatabaseError
└── RecognitionError
```

## Reference

| Exception | Trigger |
|-----------|---------|
| `ModelLoadError` | insightface not installed, no ONNX provider, corrupt model |
| `InferenceError` | GPU inference failed and CPU fallback also failed |
| `CameraError` | Cannot open camera, camera in use, unsupported resolution |
| `DatabaseError` | JSON parse error, corrupt pickle, disk/permission error |
| `RecognitionError` | Encoding dimension mismatch, empty cache, etc. |

## Catching Errors

```python
from face_vision.exceptions import FaceVisionError

try:
    detector = FaceDetector(device="cpu")
except FaceVisionError as e:
    print(f"FaceVision error: {e}")
```
