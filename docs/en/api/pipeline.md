# FaceHubPipeline

The `FaceHubPipeline` wires together all components — camera, detector,
recognizer, tracker, and database — into a single, easy-to-use processing loop.

It handles:
- Starting and stopping the camera
- Syncing the recognizer cache with the database
- Running detection + recognition + tracking
- FPS measurement
- Thread-safe frame processing

---

## Constructor Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `camera` | `CameraThread` | Camera capture thread |
| `detector` | `DetectorProtocol` | Detector (built-in `FaceDetector` or custom) |
| `recognizer` | `FaceRecognizer` | Recognizer |
| `tracker` | `FaceTracker` | Tracker |
| `db` | `FaceDatabase` | Face database |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_running` | `bool` | Whether the pipeline is started |

## Methods

### start()

Start the pipeline. The camera is started automatically if it is not already
running.

### stop()

Stop the pipeline and release the camera.

### process_frame(frame=None)

Process one frame through the full pipeline (camera → detection → recognition → tracking).

**Parameters:**
- `frame` (`np.ndarray | None`): Explicit BGR frame; if `None`, fetched from camera.

**Returns:**
- `PipelineResult | None`

### update_database_cache()

Sync the recognizer cache with the database. Returns whether the cache was rebuilt.

**Returns:**
- `bool`: `True` if the cache was rebuilt, `False` if it was already current.

### detect_only(frame)

Run detection only — no embedding extraction, no tracking.

**Parameters:**
- `frame` (`np.ndarray`): BGR image.

**Returns:**
- `List[DetectionResult]`

### extract_embeddings(frame)

Run detection + embedding extraction without tracking.

**Parameters:**
- `frame` (`np.ndarray`): BGR image.

**Returns:**
- `List[DetectionWithEmbedding]`

### reset_tracker()

Reset the tracker. Clears all active tracks and identity history. Use after
changing recognizer tolerance or switching databases.

---

## Minimal Live Camera Example

```python
from face_hub import (
    FaceHubPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread,
)

camera = CameraThread(camera_id=0, width=640, height=360)
detector = FaceDetector(device="auto", det_size=640)
recognizer = FaceRecognizer(tolerance=0.45)
tracker = FaceTracker(smooth_frames=5)
db = FaceDatabase()

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

## Process a Static Image

```python
import cv2

pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
pipeline.start()

frame = cv2.imread("group.jpg")
result = pipeline.process_frame(frame=frame)

if result:
    print(f"Found {result.total_faces} face(s)")
    for face in result.tracked_faces:
        print(face.name, face.bbox.to_tuple())

pipeline.stop()
```

## Draw Results with OpenCV

```python
import cv2

result = pipeline.process_frame()
if result is None:
    return

for face in result.tracked_faces:
    x1, y1, x2, y2 = face.bbox.to_tuple()

    # Color by status
    if face.is_known:
        color = (0, 255, 0)        # green = recognized
    elif face.is_confirmed:
        color = (0, 165, 255)      # orange = confirmed but unknown
    else:
        color = (0, 0, 255)        # red = unconfirmed

    label = f"{face.name} {face.confidence:.0%}"

    cv2.rectangle(result.frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(result.frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imshow("FaceHub", result.frame)
```

## Full Video Processing Loop

```python
import cv2
from face_hub import (
    FaceHubPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread,
)
from face_hub.types import UNKNOWN_SENTINEL

camera = CameraThread(0, width=640, height=360, fps=30)
detector = FaceDetector(device="auto", det_size=640, confidence=0.50)
recognizer = FaceRecognizer(tolerance=0.45)
tracker = FaceTracker(smooth_frames=5, iou_threshold=0.30, max_missed=10)
db = FaceDatabase()

pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
pipeline.start()

try:
    while True:
        result = pipeline.process_frame()
        if result is None:
            continue

        frame = result.frame

        for face in result.tracked_faces:
            x1, y1, x2, y2 = face.bbox.to_tuple()

            if face.is_known:
                color = (0, 255, 0)
                label = f"[{face.track_id}] {face.name} {face.confidence:.0%}"
            elif face.is_confirmed:
                color = (0, 165, 255)
                label = f"[{face.track_id}] Unknown"
            else:
                color = (0, 0, 255)
                label = f"[{face.track_id}] ..."

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + 180, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display FPS and track count
        cv2.putText(frame, f"FPS: {result.fps:.1f}  Tracks: {result.total_faces}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("FaceHub", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
```

## Detect Only (No Recognition / No Tracking)

```python
import cv2

pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
pipeline.start()

frame = cv2.imread("crowd.jpg")
detections = pipeline.detect_only(frame)
print(f"Detected {len(detections)} face(s)")

for det in detections:
    x1, y1, x2, y2 = det.bbox.to_tuple()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

## Extract Embeddings Without Tracking

Useful for batch processing photos for registration:

```python
frame = cv2.imread("photo.jpg")
faces = pipeline.extract_embeddings(frame)

for face in faces:
    if face.has_embedding:
        print(face.embedding.shape)  # (512,)
        print(f"Quality pass: {face.quality_pass}")
```

## Add a Person and Refresh Cache

```python
import cv2

# Register a new person
frame = cv2.imread("alice.jpg")
faces = pipeline.extract_embeddings(frame)
if faces and faces[0].has_embedding:
    db.add_person("Alice", "alice.jpg", faces[0].embedding)
    pipeline.update_database_cache()  # make recognizer aware of the new person

# Now Alice is recognized in live feed
```

## Reset Tracker After Settings Change

```python
# User changed recognition tolerance in settings UI
pipeline.recognizer.tolerance = 0.40
pipeline.reset_tracker()

# Or after modifying the database
db.add_person("David", "david.jpg", david_encoding)
pipeline.update_database_cache()
pipeline.reset_tracker()
```

## Batch Photo Processing

Process all images in a directory without a camera:

```python
import cv2
import glob
from face_hub import FaceDetector, FaceRecognizer, FaceTracker, FaceDatabase

detector = FaceDetector(device="auto")
db = FaceDatabase()

for path in glob.glob("photos/*.jpg"):
    frame = cv2.imread(path)
    if frame is None:
        continue

    faces = pipeline.extract_embeddings(frame)
    for face in faces:
        if face.has_embedding and face.quality_pass:
            # Register with filename as name
            name = os.path.splitext(os.path.basename(path))[0]
            db.add_person(name, path, face.embedding)
            print(f"Registered: {name}")

# Sync after batch registration
pipeline.update_database_cache()
```

## Lifecycle Best Practices

```python
pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)

try:
    pipeline.start()
    while running:
        result = pipeline.process_frame()
        # ... render / business logic ...
finally:
    pipeline.stop()  # always stop to release the camera
```

## Custom Detector Integration

You can use any detector that implements the `DetectorProtocol`:

```python
from face_hub import DetectorProtocol

class MyCustomDetector:
    # Must implement these two methods:

    def detect_with_embeddings(self, frame):
        """Return List[DetectionWithEmbedding]"""
        ...

    def detect(self, frame):
        """Return List[DetectionResult]"""
        ...

custom_detector = MyCustomDetector()
pipeline = FaceHubPipeline(camera, custom_detector, recognizer, tracker, db)
```

## Notes

- `process_frame()` is **thread-safe**. Internally it uses a lock, so you can call it from the main thread while another thread calls `detect_only()`.
- If `frame` is `None` and the camera has no new frame, `process_frame()` returns `None`. Simply skip the frame and call again in the next loop iteration.
- The recognizer cache is updated automatically every frame when the database version changes. You rarely need to call `update_database_cache()` manually.
- Always call `pipeline.stop()` when done to release camera resources and stop background threads.
- `process_frame(frame=...)` can process an explicit frame without using the camera, useful for image batch processing.