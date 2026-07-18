# Types (Data Structures)

All public data structures used by the FaceHub API. These are pure data
containers (mostly `NamedTuple` subclasses) returned by detectors,
recognizers, trackers, and the pipeline.

---

## BBox

Bounding box with four integer coordinates.

```python
from face_hub import BBox

bbox = BBox(x1=100, y1=50, x2=300, y2=250)
print(bbox.to_tuple())  # (100, 50, 300, 250)
print(bbox.width)       # 200
print(bbox.height)      # 200
print(bbox.area)        # 40000
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `x1` | `int` | Left |
| `y1` | `int` | Top |
| `x2` | `int` | Right |
| `y2` | `int` | Bottom |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `to_tuple()` | `(int, int, int, int)` | `(x1, y1, x2, y2)` |

### Computed Properties

| Property | Description |
|----------|-------------|
| `width` | `x2 - x1` |
| `height` | `y2 - y1` |
| `area` | `width * height` |

---

## DetectionResult

Bare detection result — bounding box + confidence, no embedding.

```python
from face_hub import DetectionResult, BBox

det = DetectionResult(
    bbox=BBox(100, 50, 300, 250),
    confidence=0.92,
)
print(det)
```

Returned by `FaceDetector.detect()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `bbox` | `BBox` | Bounding box |
| `confidence` | `float` | Detection confidence `[0.0, 1.0]` |

---

## DetectionWithEmbedding

Detection result with an optional 512-dim embedding and quality flag.

Returned by `FaceDetector.detect_with_embeddings()`.

```python
from face_hub import DetectionWithEmbedding, BBox
import numpy as np

det = DetectionWithEmbedding(
    bbox=BBox(100, 50, 300, 250),
    confidence=0.92,
    embedding=np.array([...]) if embedding_available else None,
    quality_pass=True,
)

if det.has_embedding:
    print(det.embedding.shape)  # (512,)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `bbox` | `BBox` | Bounding box |
| `confidence` | `float` | Detection confidence |
| `embedding` | `np.ndarray \| None` | 512-dim L2-normalized embedding, or `None` |
| `quality_pass` | `bool` | Whether the face passes quality filter |

### Properties

| Property | Description |
|----------|-------------|
| `has_embedding` | `True` if `embedding is not None` |

---

## TrackedFace

A tracked face with temporal identity smoothing.

Returned by `FaceTracker.update()`.

```python
from face_hub import TrackedFace, BBox

face = TrackedFace(
    track_id=3,
    bbox=BBox(100, 50, 300, 250),
    name="Alice",
    age=15,
    is_confirmed=True,
    confidence=0.78,
)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `track_id` | `int` | Stable ID assigned by the tracker |
| `bbox` | `BBox` | Current bounding box |
| `name` | `str` | Recognized name or `"Unknown"` |
| `age` | `int` | Frames since this track was created |
| `is_confirmed` | `bool` | Identity passed majority-vote smoothing |
| `confidence` | `float` | Average cosine similarity for confirmed tracks |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_known` | `bool` | `True` if the name is not `"Unknown"` (also `"Unknown"` constant) |

### TrackedFace States

| State | `is_confirmed` | `is_known` | Meaning |
|-------|---------------|-----------|---------|
| Unknown | `False` | `False` | Not recognized; unstable |
| Pending | `False` | `True` | Recognizer matched someone, but not enough frames yet |
| Confirmed | `True` | `True` | Identity confirmed by majority vote across `smooth_frames` frames |

---

## PipelineResult

The result returned by `FaceHubPipeline.process_frame()`.

```python
from face_hub.pipeline import PipelineResult

result = pipeline.process_frame()
if result is not None:
    print(f"FPS: {result.fps}")
    print(f"Faces detected: {result.total_faces}")
    for face in result.tracked_faces:
        print(face.track_id, face.name, face.is_confirmed)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `frame` | `np.ndarray` | The BGR frame that was processed |
| `tracked_faces` | `List[TrackedFace]` | All tracked faces in this frame |
| `fps` | `float` | Running average FPS |
| `total_faces` | `int` | Number of faces in this frame (same as `len(tracked_faces)`) |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `known_faces` | `List[TrackedFace]` | Faces where `is_known is True` |
| `confirmed_faces` | `List[TrackedFace]` | Faces where `is_confirmed is True` |
| `unknown_faces` | `List[TrackedFace]` | Faces where `is_known is False` |

---

## PhotoFace

One face found in one photo. Stored in `PhotoClassificationResult.faces`.

```python
from face_hub import PhotoFace, BBox

face = PhotoFace(
    photo_id="party1.jpg",
    bbox=BBox(100, 50, 300, 250),
    det_confidence=0.92,
    label="person_001",
    similarity=0.71,
)
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `photo_id` | `str` | Id of the photo the face was found in |
| `bbox` | `BBox` | Bounding box of the face |
| `det_confidence` | `float` | Detection confidence `[0.0, 1.0]` |
| `label` | `str` | Person name, cluster label (`person_001`, ...), or `UNKNOWN_SENTINEL` |
| `similarity` | `float` | Cosine similarity to the matched person / cluster centroid |

---

## PhotoGroup

A group of photos that contain the same person.

| Attribute | Type | Description |
|-----------|------|-------------|
| `label` | `str` | Person name or cluster label |
| `photo_ids` | `List[str]` | Unique photo ids, in first-appearance order |
| `face_count` | `int` | Total faces of this person across the group |

### Computed Properties

| Property | Description |
|----------|-------------|
| `photo_count` | `len(photo_ids)` |

!!! note
    A photo containing several people appears in several groups, so
    `photo_count` values across groups do **not** sum to `total_photos`.

---

## PhotoClassificationResult

Returned by `PhotoClassifier.classify_photos()` and the `classify_photos()`
helper. See [Photo Classifier](classifier.md) for usage.

```python
result = classifier.classify_photos(photos)

print(result.groups)            # {"person_001": PhotoGroup(...), ...}
print(result.no_face_photos)    # photos with no usable face
print(result.summary())         # {"person_001": 3, "person_002": 1}
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `groups` | `Dict[str, PhotoGroup]` | Label → group, in first-appearance order |
| `faces` | `List[PhotoFace]` | Every usable face across all photos |
| `no_face_photos` | `List[str]` | Photos with no usable face |
| `unreadable_photos` | `List[str]` | Photos that failed to decode |
| `total_photos` | `int` | Always `len(images)` |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `labels` | `List[str]` | All group labels in first-appearance order |
| `total_faces` | `int` | `len(faces)` |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `photos_of(label)` | `List[str]` | Photo ids of a person / cluster (empty if unknown label) |
| `labels_of(photo_id)` | `List[str]` | All labels found in one photo |
| `summary()` | `Dict[str, int]` | Label → photo count, plus `__no_face__` / `__unreadable__` entries |

---

## Constants

### UNKNOWN_SENTINEL

```python
from face_hub import UNKNOWN_SENTINEL

print(UNKNOWN_SENTINEL)  # "Unknown"
```

The string `"Unknown"`. Compare recognition results against this:

```python
name, conf = recognizer.recognize(embedding)
if name == UNKNOWN_SENTINEL:
    print("Not recognized")
```

---

## Usage Patterns

### Filter Faces by State

```python
result = pipeline.process_frame()
if result is None:
    return

# Known (recognized by name)
for face in result.known_faces:
    print(f"✓ {face.name} ({face.confidence:.0%})")

# Confirmed (stable identity)
for face in result.confirmed_faces:
    print(f"[Confirmed] {face.name} track={face.track_id}")

# Unknown
for face in result.unknown_faces:
    print(f"? Unknown face at {face.bbox.to_tuple()}")
```

### Working with BBox

```python
from face_hub import BBox

# Create from coordinates
bbox = BBox(10, 20, 200, 220)

# Unpack
x1, y1, x2, y2 = bbox.to_tuple()

# Center point
cx = (bbox.x1 + bbox.x2) // 2
cy = (bbox.y1 + bbox.y2) // 2

# Size
print(f"Size: {bbox.width}×{bbox.height}, Area: {bbox.area}px²")

# Crop face from frame
import cv2
face_roi = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
```

### Check if Embedding is Present

```python
results = detector.detect_with_embeddings(frame)
for det in results:
    if det.has_embedding:
        # Safe to access det.embedding
        print(det.embedding.shape)   # (512,)
        print(det.embedding.dtype)   # float32
    else:
        print("No embedding — face was filtered out by quality check")
```

### Convert to JSON-Serializable

```python
import json

def tracked_face_to_dict(face):
    return {
        "track_id": face.track_id,
        "bbox": list(face.bbox.to_tuple()),
        "name": face.name,
        "is_confirmed": face.is_confirmed,
        "confidence": round(face.confidence, 4),
        "age": face.age,
    }

result = pipeline.process_frame()
if result:
    faces_data = [tracked_face_to_dict(f) for f in result.tracked_faces]
    print(json.dumps(faces_data, indent=2))
```

### Accumulate All Faces Over Time

```python
all_faces = {}

while pipeline.is_running:
    result = pipeline.process_frame()
    if result is None:
        continue

    for face in result.tracked_faces:
        if face.is_confirmed and face.is_known:
            all_faces[face.track_id] = {
                "name": face.name,
                "confidence": face.confidence,
                "age": face.age,
            }

    # Show everyone who has appeared so far
    for tid, info in all_faces.items():
        print(f"Track {tid}: {info['name']} (appeared {info['age']} frames ago)")
```

## Notes

- All data types in `face_hub` are `NamedTuple` subclasses, making them
  lightweight, immutable, and iterable.
- `DetectionWithEmbedding.embedding` is `None` when quality filtering rejects
  the face. Always check `has_embedding` before accessing it.
- `TrackedFace.name` equals `"Unknown"` (`UNKNOWN_SENTINEL`) when no match is found
  or the track is not yet confirmed.
- BBox coordinates are zero-indexed and in image pixel space; they are always
  integers (clamped to frame boundaries).
