# CameraThread

Camera capture thread that runs in a dedicated background thread, providing
thread-safe access to the latest frame.

This class is **capture-only** — all ML inference happens in other components
(usually orchestrated by `FaceHubPipeline`).

---

## Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `camera_id` | `int` | `0` | Camera index (0 is the default camera) |
| `width` | `int` | `640` | Requested capture width (pixels) |
| `height` | `int` | `360` | Requested capture height (pixels) |
| `fps` | `int` | `30` | Requested capture frame rate |

> **Note:** The OpenCV backend is auto-detected per platform (Windows: DirectShow, macOS: AVFoundation, Linux: V4L2). No manual configuration is needed.

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `running` | `bool` | Whether the capture thread is active |
| `actual_fps` | `float` | Measured capture FPS (updated once per second) |

## Methods

### start()

Open the camera and start the background capture thread. Raises `CameraError`
if the camera cannot be opened.

### stop()

Stop the capture thread and release camera resources.

### get_frame(timeout=0.05, copy=True)

Get the latest captured frame.

**Parameters:**
- `timeout` (`float`): Max seconds to wait for a new frame. Returns `None` if no frame arrives within the deadline.
- `copy` (`bool`): If `True` (default), returns a safe copy. Set `False` for zero-copy mode — caller must NOT mutate the returned frame.

**Returns:**
- `np.ndarray | None`: BGR frame `(H, W, 3)`, or `None` if timeout expired.

### list_cameras(max_test=5)

Static method. List available camera indices on the system.

**Parameters:**
- `max_test` (`int`): Maximum number of indices to test (0 to max_test-1).

**Returns:**
- `List[int]`: Available camera indices.

---

## Platform Backends

| OS | OpenCV Backend | Notes |
|----|----------------|-------|
| Windows | `cv2.CAP_DSHOW` | DirectShow, best compatibility |
| macOS | `cv2.CAP_AVFOUNDATION` | AVFoundation |
| Linux | `cv2.CAP_V4L2` | Video4Linux2 |

---

## Basic Usage

```python
import cv2
from face_hub import CameraThread

camera = CameraThread(camera_id=0, width=640, height=360, fps=30)
camera.start()

try:
    while True:
        frame = camera.get_frame()
        if frame is not None:
            cv2.imshow("Preview", frame)
            if cv2.waitKey(1) == 27:  # ESC to quit
                break
finally:
    camera.stop()
    cv2.destroyAllWindows()
```

## Context Manager

`CameraThread` supports the `with` statement for automatic cleanup:

```python
from face_hub import CameraThread

with CameraThread(camera_id=0) as cam:
    while True:
        frame = cam.get_frame()
        if frame is None:
            continue
        # process frame...
# camera is automatically stopped when exiting the block
```

## Enumerate Available Cameras

```python
from face_hub import CameraThread

available = CameraThread.list_cameras(max_test=10)
print(f"Available cameras: {available}")

if not available:
    print("No cameras detected")
elif 0 in available:
    camera = CameraThread(camera_id=0)
else:
    camera = CameraThread(camera_id=available[0])
```

## Monitor Actual FPS

```python
import time
from face_hub import CameraThread

camera = CameraThread(camera_id=0, width=640, height=360, fps=30)
camera.start()

# Wait for FPS statistics to stabilize (at least 1 second)
time.sleep(2)

print(f"Requested: 30 FPS")
print(f"Actual: {camera.actual_fps:.1f} FPS")

camera.stop()
```

## Zero-Copy Mode

Zero-copy mode avoids memory duplication for high-frame-rate scenarios.
**Caution:** you must finish processing before the next frame arrives:

```python
camera = CameraThread(camera_id=0, width=1280, height=720, fps=60)
camera.start()

while True:
    frame = camera.get_frame(copy=False)
    if frame is None:
        continue

    # ⚠️ Must finish processing before the next frame arrives
    processed = cv2.resize(frame, (320, 180))
    cv2.imshow("Preview", processed)

    if cv2.waitKey(1) == 27:
        break

camera.stop()
```

## Resolution Presets

```python
# Low resolution: high FPS, good for real-time monitoring
low_res = CameraThread(camera_id=0, width=320, height=180, fps=60)

# Medium resolution: recommended for face recognition
mid_res = CameraThread(camera_id=0, width=640, height=360, fps=30)

# High resolution: good for photo capture / registration
high_res = CameraThread(camera_id=0, width=1280, height=720, fps=15)
```

## Error Handling

```python
from face_hub import CameraThread
from face_hub.exceptions import CameraError

try:
    camera = CameraThread(camera_id=0)
    camera.start()
except CameraError as e:
    print(f"Cannot open camera: {e}")
    print(f"  camera_id: {e.camera_id}")
    print("Check:")
    print("  1. Camera is connected")
    print("  2. Camera is not in use by another app")
    print("  3. camera_id is correct")
```

## Notes

- **Always call `stop()`** when done, or use the context manager. Otherwise the background thread and camera handle may leak.
- **`get_frame()` returns `None`** when no new frame is available within the timeout. In a real-time loop, simply skip and retry.
- **Resolution/FPS are requests** — the actual values are the closest supported by the camera hardware. Use `actual_fps` to check.
- **Thread-safe** — `get_frame()` can be called from multiple threads concurrently; each receives an independent copy.
- The capture thread is a daemon thread — it stops automatically when the main process exits, but explicit `stop()` is recommended.
