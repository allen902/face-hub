# FaceDetector

Face detector based on insightface RetinaFace with automatic GPU detection and CPU fallback.

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
- `List[DetectionWithEmbedding]`

### reload_model(det_size=None)

Reload the model at runtime with a new input size.

**Parameters:**
- `det_size` (`int | None`): New size, or `None` to keep current.

## Custom Detector

Implement `DetectorProtocol` to use your own model:

```python
from face_hub import DetectorProtocol, DetectionWithEmbedding, BBox

class MyDetector:
    def detect_with_embeddings(self, frame):
        ...  # your detection + embedding logic
        return [DetectionWithEmbedding(...)]

pipeline = FaceHubPipeline(camera, MyDetector(), ...)
```
