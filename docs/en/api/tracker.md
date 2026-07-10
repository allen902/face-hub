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
- `detections` (`List[DetectionWithEmbedding]`): Detections
- `recognizer` (`FaceRecognizer \| None`): Recognizer used to identify each detection

**Returns:**
- `List[TrackedFace]`

### reset()

Clear all tracks.

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

## Draw Tracks

```python
import cv2

for face in tracked:
    x1, y1, x2, y2 = face.bbox.to_tuple()
    color = (0, 255, 0) if face.is_confirmed else (0, 165, 255)
    label = f"{face.name} {face.confidence:.0%}"
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
```

## Reset After Configuration Change

```python
tracker.reset()  # clear all tracks, e.g. after changing recognizer tolerance
```

## Tuning Parameters

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `smooth_frames` | Faster identity confirmation, more flicker | Slower confirmation, more stable |
| `iou_threshold` | Stricter matching, more track switches | Looser matching, fewer switches |
| `max_missed` | Short memory, removes occluded faces quickly | Long memory, keeps faces through brief occlusion |

```python
# Fast-changing scene with many people
fast_tracker = FaceTracker(smooth_frames=3, iou_threshold=0.25)

# Stable, high-quality stream
stable_tracker = FaceTracker(smooth_frames=8, max_missed=15)
```

## Notes

- `track_id` is assigned incrementally and is stable only within a continuous
  tracking session. After `reset()` the counter restarts from 0.
- A track is "confirmed" only when the majority name is not `UNKNOWN_SENTINEL`.
  This prevents noise from immediately affecting UI state.
