# CameraThread

Camera capture thread that runs in a dedicated thread and provides thread-safe
access to the latest frame.

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
| `actual_fps` | `float` | Actual capture frame rate |

## Methods

### start()

Start the camera.

### stop()

Stop the camera.

### get_frame(timeout=0.05, copy=True)

Get the latest frame.

**Parameters:**
- `timeout` (`float`): Max seconds to wait for a new frame
- `copy` (`bool`): Return a copy; `False` enables zero-copy mode

**Returns:**
- `np.ndarray | None`

### list_cameras(max_test=5)

Static method listing available camera indices.

## Platform Backends

- Windows: `cv2.CAP_DSHOW`
- macOS: `cv2.CAP_AVFOUNDATION`
- Linux: `cv2.CAP_V4L2`
