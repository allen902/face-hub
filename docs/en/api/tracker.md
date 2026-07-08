# FaceTracker

Lightweight IoU-based multi-face tracker with majority-vote identity smoothing.

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
- `recognizer` (`FaceRecognizer | None`): Recognizer used to identify each detection

**Returns:**
- `List[TrackedFace]`

### reset()

Clear all tracks.

## Algorithm

1. Detections are matched to existing tracks via IoU.
2. Each track keeps the last `smooth_frames` recognition results.
3. An identity is confirmed when it wins the majority and average similarity ≥ 0.30.
4. Unconfirmed tracks display the latest name (with implicit uncertainty).
