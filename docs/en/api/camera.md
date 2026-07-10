# CameraThread

Camera capture thread that runs in a dedicated thread and provides thread-safe,
non-blocking access to the latest frame.

This class is intentionally minimal: it only captures frames. All ML processing
happens elsewhere (typically inside `FaceHubPipeline`).

---

## Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `camera_id` | `int` | `0` | Camera index |
| `width` | `int` | `640` | Requested width |
| `height` | `int` | `360` | Requested height |
| `fps` | `int` | `30` | Requested FPS |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `running` | `bool` | Whether capture is active |
| `actual_fps` | `float` | Actual capture frame rate (updated every second) |

## Methods

### start()

Open the camera and begin capture in a background thread.

### stop()

Stop capture and release the camera.

### get_frame(timeout=0.05, copy=True)

Get the latest frame.

**Parameters:**
- `timeout` (`float`): Max seconds to wait for a new frame
- `copy` (`bool`): Return a copy; `False` enables zero-copy mode

**Returns:**
- `np.ndarray | None`: BGR frame `(H, W, 3)`, or `None` if no new frame arrived.

### list_cameras(max_test=5)

Static method listing available camera indices.

**Returns:**
- `List[int]`: Indices of working cameras.

---

## Platform Backends

| OS | OpenCV backend | Notes |
|----|----------------|-------|
| Windows | `cv2.CAP_DSHOW` | DirectShow |
| macOS | `cv2.CAP_AVFOUNDATION` | AVFoundation |
| Linux | `cv2.CAP_V4L2` | Video4Linux2 |

---

## Basic Example

```python
import cv2
from face_hub import CameraThread

camera = CameraThread(camera_id=0, width=640, height=360, fps=30)
camera.start()

for _ in range(100):
    frame = camera.get_frame()
    if frame is not None:
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == 27:
            break

camera.stop()
cv2.destroyAllWindows()
```

## List Available Cameras

```python
from face_hub import CameraThread

print(CameraThread.list_cameras(max_test=5))  # e.g. [0, 2]
```

## Monitor Actual FPS

```python
import time
from face_hub import CameraThread

camera = CameraThread(camera_id=0)
camera.start()

time.sleep(2)
print(f"Requested 30 FPS, actual {camera.actual_fps:.1f} FPS")

camera.stop()
```

## Tips

- Always call `stop()` when done, otherwise the thread and camera handle may leak.
- `get_frame()` returns `None` if the camera has not produced a new frame since
the last call. In a real-time loop, just skip the frame and try again.
- The requested resolution/FPS are hints; use `actual_fps` and OpenCV properties
if you need to know the real values.
