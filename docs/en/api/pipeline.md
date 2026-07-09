# FaceHubPipeline

Full pipeline integrating camera, detector, recognizer, tracker, and database.

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

Start the pipeline (starts the camera if needed).

### stop()

Stop the pipeline (stops the camera).

### process_frame(frame=None)

Process one frame and return a `PipelineResult`, or `None` if no frame is available.

**Parameters:**
- `frame` (`np.ndarray | None`): Explicit BGR frame; if `None`, fetched from camera.

**Returns:**
- `PipelineResult | None`

### update_database_cache()

Sync the recognizer cache with the database. Returns whether the cache was rebuilt.

### detect_only(frame)

Run detection only.

### extract_embeddings(frame)

Run detection + embedding extraction without tracking.

### reset_tracker()

Reset the tracker.
