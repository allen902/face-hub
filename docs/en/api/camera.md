# CameraThread

Background thread for camera frame capture. Fetches frames from a USB or
built-in camera in a dedicated thread to avoid blocking the main loop.

Key features:
- Configurable resolution and FPS
- Thread-safe frame access
- Optional horizontal flip (mirror)
- Automatic camera release

---

## Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `camera_id` | `int` | `0` | Camera device index (0 = default camera) |
| `width` | `int` | `640` | Frame width |
| `height` | `int` | `360` | Frame height |
| `fps` | `int` | `30` | Target capture rate |
| `flip` | `int` | `0` | Flip mode: 0=none, 1=horizontal, -1=vertical, 0=no flip |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_running` | `bool` | Whether the capture thread is active |
| `width` | `int` | Current frame width |
| `height` | `int` | Current frame height |

## Methods

### start()

Start the capture thread. Opens the camera and begins fetching frames.

### stop()

Stop the capture thread and release the camera.

### read()

Get the latest frame.

**Returns:**
- `np.ndarray | None`: BGR frame `(height, width, 3)`, or `None` if no frame is available yet.

---

## Basic Usage

```python
from face_hub import CameraThread

camera = CameraThread(camera_id=0, width=640, height=360, fps=30)
camera.start()

while True:
    frame = camera.read()
    if frame is None:
        continue
    # process frame...

camera.stop()
```

## With the Pipeline (Recommended)

```python
from face_hub import (
    CameraThread, FaceHubPipeline,
    FaceDetector, FaceRecognizer, FaceTracker, FaceDatabase,
)

camera = CameraThread(camera_id=0, width=640, height=360)
pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)

pipeline.start()  # starts camera automatically
try:
    while True:
        result = pipeline.process_frame()
        if result is not None:
            cv2.imshow("FaceHub", result.frame)
finally:
    pipeline.stop()  # stops camera automatically
```

## Auto-Start / Auto-Stop

`CameraThread` can be started manually or automatically. The pipeline
handles this for you:

```python
camera = CameraThread(0)
print(camera.is_running)  # False (camera not opened yet)

# Option 1: Start manually
camera.start()
print(camera.is_running)  # True

# Option 2: Let pipeline start it
pipeline = FaceHubPipeline(camera, ...)
pipeline.start()
print(camera.is_running)  # True (started by pipeline)
```

## Mirror / Flip

```python
# Mirror mode for selfie / kiosk
mirror_camera = CameraThread(camera_id=0, flip=1)

# No flip
normal_camera = CameraThread(camera_id=0, flip=0)
```

## Multiple Cameras

```python
cam1 = CameraThread(camera_id=0, width=640, height=360)
cam2 = CameraThread(camera_id=1, width=320, height=240)

cam1.start()
cam2.start()

frame1 = cam1.read()
frame2 = cam2.read()
```

## FPS Target

```python
# 60 FPS for smooth preview
smooth_cam = CameraThread(camera_id=0, fps=60)

# 15 FPS for low CPU usage
low_usage_cam = CameraThread(camera_id=0, fps=15)
```

**Note:** The actual FPS may be lower than the target depending on camera hardware
capabilities and system load.

## Resolution Presets

```python
# Common resolutions

# 480p (SD) — low CPU usage
cam_sd = CameraThread(camera_id=0, width=640, height=480)

# 720p (HD) — balanced
cam_hd = CameraThread(camera_id=0, width=1280, height=720)

# 1080p (Full HD) — high quality, more CPU
cam_fhd = CameraThread(camera_id=0, width=1920, height=1080)
```

## Lifecycle Best Practices

```python
camera = CameraThread(0)

try:
    camera.start()
    while True:
        frame = camera.read()
        if frame is not None:
            cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    camera.stop()       # always release
    cv2.destroyAllWindows()
```

## Notes

- The camera is opened in a background thread, so `start()` returns immediately.
- The capture thread is created as a daemon thread — if the main process exits,
  the thread stops automatically (but it is better to call `stop()` explicitly).
- `read()` returns `None` when the camera hasn't produced a frame yet. Always
  check for `None` before processing.
- Changing resolution requires stopping and re-creating the `CameraThread`.
- On Windows, DirectShow backend is used automatically via OpenCV.