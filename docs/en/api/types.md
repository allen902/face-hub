# Types

All public API return types are dataclasses for type safety.

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

Method: `to_tuple()` → `(x1, y1, x2, y2)`

## DetectionResult

Single result from `detect()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `bbox` | `BBox` | Bounding box |
| `confidence` | `float` | Detection confidence 0.0~1.0 |

## DetectionWithEmbedding

Single result from `detect_with_embeddings()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `bbox` | `BBox` | Bounding box |
| `confidence` | `float` | Detection confidence |
| `embedding` | `np.ndarray \| None` | 512-dim L2-normalized feature vector |
| `quality_pass` | `bool` | Whether quality filter passed |
| `has_embedding` | `bool` | Whether embedding was extracted |

## TrackedFace

Single tracking result from `FaceTracker.update()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `track_id` | `int` | Track ID |
| `bbox` | `BBox` | Current-frame bounding box |
| `name` | `str` | Recognized name, or `UNKNOWN_SENTINEL` |
| `confidence` | `float` | Cosine similarity 0.0~1.0 |
| `det_confidence` | `float` | Detection confidence |
| `is_confirmed` | `bool` | Identity confirmed by majority vote |
| `quality_pass` | `bool` | Quality filter passed |
| `is_known` | `bool` | Whether the face matches a registered person |

## PipelineResult

Full result from `FaceHubPipeline.process_frame()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `frame` | `np.ndarray` | Current frame |
| `raw_detections` | `List[DetectionWithEmbedding]` | Raw detections |
| `tracked_faces` | `List[TrackedFace]` | Tracked recognition results |
| `fps` | `float` | Current processing FPS |
| `known_faces` | `List[TrackedFace]` | Filtered known faces |
| `unknown_count` | `int` | Number of unknown faces |
| `total_faces` | `int` | Total number of faces |

## Global Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `UNKNOWN_SENTINEL` | `"unknown"` | Default name when not matched |
