# CameraThread

摄像头采集线程，在独立线程中持续采集帧，并通过线程安全的方式在主线程中获取最新帧。

这个类**只负责采集**，所有 ML 推理都在其他组件中完成（通常由 `FaceHubPipeline` 统一调度）。

---

## 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `camera_id` | `int` | `0` | 摄像头索引（0 为默认摄像头） |
| `width` | `int` | `640` | 请求的采集宽度（像素） |
| `height` | `int` | `360` | 请求的采集高度（像素） |
| `fps` | `int` | `30` | 请求的采集帧率 |
| `backend` | `int` | `None` | OpenCV 后端。`None` 为自动选择。可手动指定如 `cv2.CAP_DSHOW`（Windows DirectShow） |

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `running` | `bool` | 采集线程是否正在运行 |
| `actual_fps` | `float` | 实际采集帧率（每秒更新一次） |

## 方法

### start()

打开摄像头并在后台线程开始采集。摄像头打开失败时抛出 `CameraError`。

### stop()

停止采集线程并释放摄像头资源。

### get_frame(timeout=0.05, copy=True)

获取最新采集的帧。

**参数:**
- `timeout` (`float`): 等待新帧的最大秒数。如果在超时时间内没有新帧，返回 `None`。
- `copy` (`bool`): 是否返回帧的拷贝。`True`（默认）安全但稍慢；`False` 为零拷贝模式，效率更高但需要确保在获取下一帧前处理完。

**返回:**
- `np.ndarray | None`: BGR 格式的帧 `(H, W, 3)`，无新帧时返回 `None`。

### list_cameras(max_test=5)

静态方法。列出系统中可用的摄像头索引。

**参数:**
- `max_test` (`int`): 最多探测多少个索引（0 到 max_test-1）。

**返回:**
- `List[int]`: 可用的摄像头索引列表。

---

## 平台后端支持

| 操作系统 | OpenCV 后端 | 说明 |
|----------|-------------|------|
| Windows | `cv2.CAP_DSHOW` | DirectShow，兼容性最好 |
| macOS | `cv2.CAP_AVFOUNDATION` | AVFoundation |
| Linux | `cv2.CAP_V4L2` | Video4Linux2 |

---

## 基础用法

```python
import cv2
from face_hub import CameraThread

camera = CameraThread(camera_id=0, width=640, height=360, fps=30)
camera.start()

try:
    for _ in range(100):
        frame = camera.get_frame()
        if frame is not None:
            cv2.imshow("预览", frame)
            if cv2.waitKey(1) == 27:  # 按 ESC 退出
                break
finally:
    camera.stop()
    cv2.destroyAllWindows()
```

## 与上下文管理器配合使用

```python
from contextlib import contextmanager

@contextmanager
def open_camera(camera_id=0):
    camera = CameraThread(camera_id=camera_id)
    camera.start()
    try:
        yield camera
    finally:
        camera.stop()

# 使用
with open_camera(0) as cam:
    while True:
        frame = cam.get_frame()
        if frame is None:
            continue
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == 27:
            break
```

## 枚举可用摄像头

```python
from face_hub import CameraThread

available = CameraThread.list_cameras(max_test=10)
print(f"可用摄像头: {available}")

if not available:
    print("未检测到摄像头，请检查设备连接")
elif 0 in available:
    camera = CameraThread(camera_id=0)
else:
    camera = CameraThread(camera_id=available[0])  # 使用第一个可用摄像头
```

## 监控实际帧率

```python
import time
from face_hub import CameraThread

camera = CameraThread(camera_id=0, width=640, height=360, fps=30)
camera.start()

# 等待 FPS 统计稳定（至少 1 秒采样）
time.sleep(2)

print(f"请求: 30 FPS")
print(f"实际: {camera.actual_fps:.1f} FPS")

camera.stop()
```

## 零拷贝模式

零拷贝模式避免内存复制，适合高帧率场景。**注意**：必须在下一帧到来之前完成处理：

```python
camera = CameraThread(camera_id=0, width=1280, height=720, fps=60)
camera.start()

while True:
    # copy=False：零拷贝模式，速度快但帧内容可能会被覆盖
    frame = camera.get_frame(copy=False)
    if frame is None:
        continue

    # ⚠️ 务必在下一帧到来前完成处理
    # 不要在零拷贝模式下做耗时操作（如模型推理）
    processed = cv2.resize(frame, (320, 180))
    cv2.imshow("Preview", processed)

    if cv2.waitKey(1) == 27:
        break

camera.stop()
```

## 多摄像头支持

```python
from face_hub import CameraThread

# 获取所有可用摄像头
cameras = []
for cam_id in CameraThread.list_cameras(max_test=5):
    try:
        cam = CameraThread(camera_id=cam_id, width=640, height=360)
        cam.start()
        cameras.append((cam_id, cam))
        print(f"✅ 摄像头 {cam_id} 启动成功")
    except Exception as e:
        print(f"❌ 摄像头 {cam_id} 启动失败: {e}")

# 同时显示多个摄像头
import cv2
import numpy as np

try:
    while True:
        frames = []
        for cam_id, cam in cameras:
            frame = cam.get_frame()
            if frame is not None:
                # 缩放到统一大小
                frame = cv2.resize(frame, (320, 240))
                cv2.putText(frame, f"Camera {cam_id}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                frames.append(frame)

        if frames:
            # 水平拼接
            display = np.hstack(frames)
            cv2.imshow("All Cameras", display)

        if cv2.waitKey(1) == 27:
            break

finally:
    for _, cam in cameras:
        cam.stop()
    cv2.destroyAllWindows()
```

## 指定分辨率

不同分辨率对性能和质量的影响：

```python
# 低分辨率：高帧率，适合实时监控
low_res = CameraThread(camera_id=0, width=320, height=180, fps=60)

# 中等分辨率：推荐用于人脸识别
mid_res = CameraThread(camera_id=0, width=640, height=360, fps=30)

# 高分辨率：适合拍照、注册人脸
high_res = CameraThread(camera_id=0, width=1280, height=720, fps=15)
```

## 错误处理

```python
from face_hub import CameraThread
from face_hub.exceptions import CameraError

try:
    camera = CameraThread(camera_id=0)
    camera.start()
    print("摄像头已启动")
except CameraError as e:
    print(f"无法打开摄像头: {e}")
    print("请检查:")
    print("  1. 摄像头是否已连接")
    print("  2. 是否被其他程序占用")
    print("  3. camera_id 是否正确")

# 运行时帧获取失败处理
while camera.running:
    frame = camera.get_frame(timeout=0.1)
    if frame is None:
        # 摄像头可能暂时卡顿，跳过本帧
        continue
    # 处理 frame...
```

## 自定义后端

在某些平台上需要手动指定 OpenCV 后端以提高兼容性：

```python
import cv2

# Windows: 使用 DirectShow 后端
camera = CameraThread(camera_id=0, backend=cv2.CAP_DSHOW)

# macOS: 使用 AVFoundation 后端
camera = CameraThread(camera_id=0, backend=cv2.CAP_AVFOUNDATION)

# Linux: 使用 V4L2 后端
camera = CameraThread(camera_id=0, backend=cv2.CAP_V4L2)
```

## 注意事项

- **必须调用 `stop()`**：使用完毕后务必调用 `stop()`，否则后台线程和摄像头句柄可能泄漏。
- **`get_frame()` 返回 `None`**：如果摄像头在上次调用后没有产生新帧，会返回 `None`。在实时循环中直接跳过该帧即可，下次继续调用。
- **分辨率/帧率是请求值**：实际使用的是摄像头硬件支持的最接近值。可通过 `actual_fps` 属性查看实际帧率，或通过 OpenCV 的 `cv2.CAP_PROP_FRAME_WIDTH` / `cv2.CAP_PROP_FRAME_HEIGHT` 查看实际分辨率。
- **线程安全**：`CameraThread` 内部使用了锁保护帧缓存，多线程访问 `get_frame()` 是安全的。