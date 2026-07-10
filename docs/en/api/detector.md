# FaceDetector

Face detector based on [insightface](https://github.com/deepinsight/insightface)
RetinaFace. It detects faces, extracts 512-dim ArcFace embeddings, supports GPU
auto-detection (CUDA / DirectML) with CPU fallback, and includes a built-in blur
and size quality filter.

---

## Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `confidence` | `float` | `0.50` | Detection confidence threshold; faces below this are discarded |
| `device` | `str` | `"auto"` | Inference device: `"cpu"`, `"cuda"`, or `"auto"` (auto-detect best GPU) |
| `det_size` | `int` | `640` | Model input size: 320 (fast), 480 (balanced), 640 (accurate) |
| `quality_filter` | `bool` | `True` | Enable blur-based quality filtering |
| `min_face_size` | `int` | `80` | Minimum face size in pixels |

## Methods

### detect(frame)

Detect faces without extracting embeddings.

**Parameters:**
- `frame` (`np.ndarray`): BGR image `(H, W, 3)`.

**Returns:**
- `List[DetectionResult]`

### detect_with_embeddings(frame)

Detect faces and extract 512-dim embeddings.

**Parameters:**
- `frame` (`np.ndarray`): BGR image `(H, W, 3)`.

**Returns:**
- `List[DetectionWithEmbedding]` sorted by confidence descending.

### reload_model(det_size=None)

Reload the model at runtime with a new input size.

**Parameters:**
- `det_size` (`int | None`): New size, or `None` to keep current.

---

## Basic Detection

```python
import cv2
from face_hub import FaceDetector

detector = FaceDetector(device="auto", det_size=640)
frame = cv2.imread("photo.jpg")

results = detector.detect(frame)
print(f"Found {len(results)} face(s)")

for face in results:
    x1, y1, x2, y2 = face.bbox.to_tuple()
    print(f"  box=({x1},{y1},{x2},{y2}), confidence={face.confidence:.2f}")

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

## Detection + Embeddings

```python
faces = detector.detect_with_embeddings(frame)
for face in faces:
    if face.quality_pass and face.has_embedding:
        print(f"Embedding shape: {face.embedding.shape}")  # (512,)
        print(f"Embedding dtype: {face.embedding.dtype}")   # float32
        print(f"Is L2-normalized: {np.linalg.norm(face.embedding):.4f} ≈ 1.0")
```

## Register a New Person from a Photo

```python
import cv2
from face_hub import FaceDetector, FaceDatabase

detector = FaceDetector(device="cpu")
db = FaceDatabase()

def register_from_photo(name, photo_path):
    """Register a person from a single photo"""
    frame = cv2.imread(photo_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read: {photo_path}")

    faces = detector.detect_with_embeddings(frame)

    if not faces:
        raise ValueError(f"No face detected in {photo_path}")

    # Pick the most confident face
    face = faces[0]
    if not face.has_embedding:
        raise ValueError(f"Failed to extract embedding from {photo_path}")

    ok, msg = db.add_person(name, photo_path, face.embedding)
    if not ok:
        raise ValueError(f"Database error: {msg}")

    print(f"✅ {name} registered (confidence: {face.confidence:.0%})")
    return True

# Register multiple people
for name, path in [("Alice", "photos/alice.jpg"), ("Bob", "photos/bob.jpg")]:
    try:
        register_from_photo(name, path)
    except Exception as e:
        print(f"❌ Failed to register {name}: {e}")
```

## Switch Detection Resolution at Runtime

```python
# Start with fast detection for live preview
detector = FaceDetector(det_size=320)

# Switch to high accuracy for a critical frame
detector.reload_model(det_size=640)
results = detector.detect_with_embeddings(important_frame)

# Switch back to fast mode
detector.reload_model(det_size=320)
```

## Device Selection

```python
from face_hub import FaceDetector

# Auto-detect best available: CUDA > DirectML > CPU
detector = FaceDetector(device="auto")

# Force CPU (useful for reproducible benchmarks or GPU-free servers)
detector_cpu = FaceDetector(device="cpu")

# Try GPU with explicit fallback
try:
    detector = FaceDetector(device="cuda")
except Exception as e:
    print(f"CUDA unavailable, falling back to CPU: {e}")
    detector = FaceDetector(device="cpu")
```

## Quality Filtering

```python
# Stricter quality: skip faces smaller than 100 px, disable blur filter
detector = FaceDetector(
    confidence=0.60,
    min_face_size=100,
    quality_filter=False,
)

for face in detector.detect_with_embeddings(frame):
    if face.quality_pass:
        print("✅ Good quality face")
    else:
        print("❌ Face too small or blurry — skipped")
```

## Recommended Settings by Scenario

| Scenario | `det_size` | `confidence` | `min_face_size` |
|----------|------------|--------------|-----------------|
| Real-time video (speed) | 320 | 0.50 | 80 |
| Balanced | 480 | 0.50 | 80 |
| Photo registration (accuracy) | 640 | 0.60 | 100 |
| High-security access | 640 | 0.65 | 100 |

```python
# Real-time camera preview
live_detector = FaceDetector(device="auto", det_size=320)

# Photo registration / ID verification
photo_detector = FaceDetector(device="auto", det_size=640, confidence=0.60)
```

## Performance Optimization

```python
# If GPU memory is limited, reduce input size
low_memory_detector = FaceDetector(device="cuda", det_size=320)

# Disable quality filter for maximum speed
fast_detector = FaceDetector(device="auto", det_size=320, quality_filter=False)

# High quality for registration
quality_detector = FaceDetector(
    device="auto",
    det_size=640,
    confidence=0.55,
    min_face_size=100,
    quality_filter=True,
)
```

## Custom Detector

Implement `DetectorProtocol` to use your own model:

```python
from typing import List
import numpy as np
from face_hub import DetectorProtocol, DetectionWithEmbedding, DetectionResult, BBox

class MyYoloDetector:
    def detect_with_embeddings(self, frame: np.ndarray) -> List[DetectionWithEmbedding]:
        boxes = self.yolo_model(frame)  # your detector
        results = []
        for box in boxes:
            roi = frame[box.y1:box.y2, box.x1:box.x2]
            embedding = self.embedder(roi)  # your embedder
            results.append(DetectionWithEmbedding(
                bbox=BBox(box.x1, box.y1, box.x2, box.y2),
                confidence=box.conf,
                embedding=embedding,
                quality_pass=True,
            ))
        return results

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        return [
            DetectionResult(
                bbox=BBox(box.x1, box.y1, box.x2, box.y2),
                confidence=box.conf,
            )
            for box in self.yolo_model(frame)
        ]
```

## Notes

- The first call to `FaceDetector()` downloads the `buffalo_l` insightface model (~200 MB) if it is not already cached.
- The model is cached locally; subsequent instantiations load from disk.
- Embeddings are L2-normalized 512-dim `float32` vectors, suitable for cosine similarity matching.
- `detect()` and `detect_with_embeddings()` both return results sorted by confidence descending.
- When using `device="auto"` on Windows, DirectML is preferred over CUDA for broader compatibility.
- `reload_model()` is useful for switching resolution without recreating the detector instance.