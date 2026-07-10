# Types

All public API return types are dataclasses for type safety and IDE autocompletion.
They are intentionally simple, immutable where possible, and easy to serialize or
print for debugging.

---

## BBox

Immutable bounding box `(x1, y1, x2, y2)`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `x1` | `int` | Top-left x |
| `y1` | `int` | Top-left y |
| `x2` | `int` | Bottom-right x |
| `y2` | `int` | Bottom-right y |
| `width` | `int` | `x2 - x1` |
| `height` | `int` | `y2 - y1` |

### Methods

#### `to_tuple()` → `(x1, y1, x2, y2)`

Convert to a plain tuple for OpenCV or other drawing functions.

### Example

```python
from face_hub.types import BBox

box = BBox(x1=100, y1=50, x2=300, y2=250)
print(box.width, box.height)   # 200 200
print(box.to_tuple())          # (100, 50, 300, 250)

# BBox is frozen and hashable
boxes = {box: "alice"}
```

---

## DetectionResult

Single result from `FaceDetector.detect()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `bbox` | `BBox` | Bounding box |
| `confidence` | `float` | Detection confidence 0.0~1.0 |

### Example

```python
import cv2
from face_hub import FaceDetector

detector = FaceDetector(device="cpu", det_size=320)
frame = cv2.imread("group.jpg")

for det in detector.detect(frame):
    x1, y1, x2, y2 = det.bbox.to_tuple()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(f"confidence={det.confidence:.2f}")
```

---

## DetectionWithEmbedding

Single result from `FaceDetector.detect_with_embeddings()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `bbox` | `BBox` | Bounding box |
| `confidence` | `float` | Detection confidence |
| `embedding` | `np.ndarray \| None` | 512-dim L2-normalized feature vector |
| `quality_pass` | `bool` | Whether quality filter passed |
| `has_embedding` | `bool` | Whether embedding was extracted |

### Example

```python
from face_hub import FaceDetector
import cv2

detector = FaceDetector(device="cpu")
frame = cv2.imread("photo.jpg")

faces = detector.detect_with_embeddings(frame)
for face in faces:
    if not face.quality_pass:
        print("Face too blurry or too small, skipped")
        continue
    if face.has_embedding:
        print(face.embedding.shape)   # (512,)
```

---

## TrackedFace

Single tracking result from `FaceTracker.update()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `track_id` | `int` | Track ID (stable across frames) |
| `bbox` | `BBox` | Current-frame bounding box |
| `name` | `str` | Recognized name, or `UNKNOWN_SENTINEL` |
| `confidence` | `float` | Cosine similarity 0.0~1.0 |
| `det_confidence` | `float` | Detection confidence |
| `is_confirmed` | `bool` | Identity confirmed by majority vote |
| `quality_pass` | `bool` | Quality filter passed |
| `is_known` | `bool` | Whether the face matches a registered person |

### Example

```python
from face_hub import UNKNOWN_SENTINEL

for face in result.tracked_faces:
    if face.is_known:
        label = f"{face.name} {face.confidence:.0%}"
    else:
        label = "unknown"

    x1, y1, x2, y2 = face.bbox.to_tuple()
    color = (0, 255, 0) if face.is_confirmed else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
```

---

## PipelineResult

Full result from `FaceHubPipeline.process_frame()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `frame` | `np.ndarray` | Current frame |
| `raw_detections` | `List[DetectionWithEmbedding]` | Raw detections |
| `tracked_faces` | `List[TrackedFace]` | Tracked recognition results |
| `fps` | `float` | Current processing FPS |
| `known_faces` | `List[TrackedFace]` | Filtered known faces (property) |
| `unknown_count` | `int` | Number of unknown faces (property) |
| `total_faces` | `int` | Total number of faces (property) |

### Example

```python
result = pipeline.process_frame()
if result is None:
    return

print(f"FPS={result.fps:.1f}, faces={result.total_faces}")
print(f"Known: {len(result.known_faces)}, Unknown: {result.unknown_count}")

for face in result.known_faces:
    print(face.name, face.confidence)
```

---

## Global Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `UNKNOWN_SENTINEL` | `"unknown"` | Default name when a face is not matched |

### Example

```python
from face_hub.types import UNKNOWN_SENTINEL

name, confidence = recognizer.recover(query_encoding)
if name == UNKNOWN_SENTINEL:
    print("Not recognized")
```
