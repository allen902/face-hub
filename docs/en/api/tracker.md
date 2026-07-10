# FaceTracker

Lightweight IoU-based multi-face tracker with majority-vote identity smoothing.

The tracker solves two problems at once:
1. **Temporal consistency**: it keeps the same `track_id` for the same person
   across frames, even when detection flickers.
2. **Identity smoothing**: it confirms a name only after the recognizer agrees
   across several consecutive frames, suppressing single-frame misrecognition.

---

## Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `smooth_frames` | `int` | `5` | Frames required to confirm an identity |
| `iou_threshold` | `float` | `0.30` | IoU threshold for matching detections to tracks |
| `max_missed` | `int` | `10` | Frames before a track is removed |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `track_count` | `int` | Number of active tracks |

## Methods

### update(detections, recognizer=None)

Update tracks with current-frame detections.

**Parameters:**
- `detections` (`List[DetectionWithEmbedding]`): Detections from this frame.
- `recognizer` (`FaceRecognizer | None`): Recognizer to identify each detection. Pass `None` for track-only mode (no recognition).

**Returns:**
- `List[TrackedFace]`

### reset()

Clear all active tracks and identity history.

---

## Algorithm

1. Detections are matched to existing tracks via IoU.
2. Each track keeps the last `smooth_frames` recognition results.
3. An identity is confirmed when it wins the majority **and** the average
   similarity is ≥ 0.30.
4. Unconfirmed tracks display the latest name with implicit uncertainty.
5. Tracks that miss detections for `max_missed` frames are removed.

---

## Basic Usage

```python
from face_hub import FaceDetector, FaceRecognizer, FaceTracker

detector = FaceDetector(device="cpu")
recognizer = FaceRecognizer()
tracker = FaceTracker(smooth_frames=5)

# Seed recognizer with registered persons
recognizer.update_cache([alice_emb, bob_emb], ["Alice", "Bob"], db_version=1)

for frame in frames:
    detections = detector.detect_with_embeddings(frame)
    tracked = tracker.update(detections, recognizer=recognizer)

    for face in tracked:
        print(face.track_id, face.name, face.is_confirmed)
```

## Understanding TrackedFace

Each frame returns a list of `TrackedFace` objects:

```python
for face in tracked:
    print(f"ID:       {face.track_id}")
    print(f"Name:     {face.name}")
    print(f"BBox:     {face.bbox.to_tuple()}")
    print(f"Confirmed:{face.is_confirmed}")       # majority vote passed
    print(f"Confidence:{face.confidence:.2f}")     # cosine similarity
    print(f"Age:      {face.age}")                 # frames this track has existed
```

### TrackedFace States

| State | `is_confirmed` | `name` | Confidence | Meaning |
|-------|---------------|--------|------------|---------|
| **Unknown** | `False` | `"Unknown"` | 0.0 | No match found yet (or all matches were UNKNOWN_SENTINEL) |
| **Pending** | `False` | `"Alice"` | e.g. 0.72 | Recognizer says Alice, but not enough frames yet — may flicker |
| **Confirmed** | `True` | `"Alice"` | e.g. 0.78 | Recognizer agrees across `smooth_frames` frames — stable ID |

## Draw Tracks

```python
import cv2

for face in tracked:
    x1, y1, x2, y2 = face.bbox.to_tuple()

    # Color: green for confirmed, orange for pending, red for unknown
    if face.is_known:
        color = (0, 255, 0)
        label = f"{face.name} ({face.confidence:.0%})"
    elif face.is_confirmed:
        color = (0, 165, 255)  # confirmed but name is "Unknown"
        label = "Unknown"
    else:
        color = (0, 0, 255)
        label = face.name if face.name != "Unknown" else "..."

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
```

## Track-Only Mode (No Recognition)

```python
# Pass recognizer=None to only track bounding boxes
for frame in frames:
    detections = detector.detect_with_embeddings(frame)
    tracked = tracker.update(detections, recognizer=None)
    # Every face will be UNKNOWN_SENTINEL, but track_id is still stable
    for face in tracked:
        print(f"Track {face.track_id}: age={face.age}, bbox={face.bbox}")
```

## Reset After Configuration Change

```python
# Clear all tracks after changing recognizer tolerance
tracker.recognizer.tolerance = 0.40
tracker.reset()

# Or after adding new people to the database
db.add_person("Charlie", "charlie.jpg", charlie_embedding)
recognizer.update_cache(...)
tracker.reset()
```

## Tuning Parameters

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `smooth_frames` | Faster identity confirmation, more flicker | Slower confirmation, more stable |
| `iou_threshold` | Stricter matching, more track switches | Looser matching, fewer switches |
| `max_missed` | Short memory, removes occluded faces quickly | Long memory, keeps faces through brief occlusion |

```python
# Fast-changing scene with many people (e.g. crowded entrance)
fast_tracker = FaceTracker(smooth_frames=3, iou_threshold=0.25, max_missed=5)

# Stable, high-quality stream (e.g. video conference)
stable_tracker = FaceTracker(smooth_frames=8, iou_threshold=0.35, max_missed=15)
```

## Scenario-Based Configuration

### High-Security Access Control

```python
# Require more frames for confirmation, stricter IoU matching
secure_tracker = FaceTracker(
    smooth_frames=10,
    iou_threshold=0.35,
    max_missed=5,  # drop quickly if face leaves
)
```

### Retail / Foot Traffic Counting

```python
# Fast confirmation, many tracks simultaneously, loose matching
retail_tracker = FaceTracker(
    smooth_frames=3,
    iou_threshold=0.20,
    max_missed=30,  # keep tracks alive through long occlusions
)
```

### Video Conference / Stream Overlay

```python
# Balanced for stable labels
stream_tracker = FaceTracker(
    smooth_frames=6,
    iou_threshold=0.30,
    max_missed=12,
)
```

## Working with Track Statistics

```python
# Count unique faces that have appeared
confirmed_names = set()
for face in tracked:
    if face.is_confirmed and face.is_known:
        confirmed_names.add(face.name)

print(f"Confirmed people in frame: {confirmed_names}")
print(f"Total active tracks: {tracker.track_count}")
```

## Full Rendering Example

```python
import cv2

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
]

for i, face in enumerate(tracked):
    x1, y1, x2, y2 = face.bbox.to_tuple()

    # Assign a color per track_id for visual consistency
    color = COLORS[face.track_id % len(COLORS)]
    label = f"[{face.track_id}] {face.name}"

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw label background
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw confidence bar
    bar_w = int(face.confidence * 100)
    cv2.rectangle(frame, (x1, y2), (x1 + bar_w, y2 + 6), color, -1)
```

## Debug Logging

```python
import logging

for face in tracked:
    status = "✓" if face.is_confirmed else "~"
    logging.debug(
        f"[{status}] track={face.track_id} name={face.name} "
        f"conf={face.confidence:.2f} age={face.age} bbox={face.bbox}"
    )
```

## Notes

- `track_id` is assigned incrementally and is stable only within a continuous
  tracking session. After `reset()` the counter restarts from 0.
- A track is "confirmed" only when the majority name is not `UNKNOWN_SENTINEL`.
  This prevents noise from immediately affecting UI state.
- `age` starts at 1 for the first frame a track is created.
- The IoU matcher uses Hungarian algorithm for optimal one-to-one assignment.
- Tracks that lose their detection (e.g. person exits frame) stay alive for
  `max_missed` frames, then are permanently removed.