# CameraThread

摄像头采集线程，在独立线程中采集并在主线程安全地获取最新帧。

## 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `camera_id` | `int` | `0` | 摄像头索引 |
| `width` | `int` | `640` | 请求宽度 |
| `height` | `int` | `360` | 请求高度 |
| `fps` | `int` | `30` | 请求帧率 |

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `running` | `bool` | 是否正在采集 |
| `actual_fps` | `float` | 实际采集帧率 |

## 方法

### start()

启动摄像头。

### stop()

停止摄像头。

### get_frame(timeout=0.05, copy=True)

获取最新帧。

**参数:**
- `timeout` (`float`): 等待新帧的最大秒数
- `copy` (`bool`): 是否返回副本；`False` 为零拷贝模式

**返回:**
- `np.ndarray | None`

### list_cameras(max_test=5)

静态方法，列出可用摄像头索引。

## 平台后端

- Windows: `cv2.CAP_DSHOW`
- macOS: `cv2.CAP_AVFOUNDATION`
- Linux: `cv2.CAP_V4L2`
