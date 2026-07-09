# FaceHubPipeline

整合摄像头、检测器、识别器、追踪器与数据库的完整流水线。

## 构造参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `camera` | `CameraThread` | 摄像头采集线程 |
| `detector` | `DetectorProtocol` | 检测器（可用内置 `FaceDetector`） |
| `recognizer` | `FaceRecognizer` | 识别器 |
| `tracker` | `FaceTracker` | 追踪器 |
| `db` | `FaceDatabase` | 人脸数据库 |

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `is_running` | `bool` | 流水线是否已启动 |

## 方法

### start()

启动流水线（需要时会自动启动摄像头）。

### stop()

停止流水线（停止摄像头）。

### process_frame(frame=None)

处理一帧，返回 `PipelineResult`；无帧时返回 `None`。

**参数:**
- `frame` (`np.ndarray | None`): 显式传入的 BGR 帧；为 `None` 时从摄像头获取。

**返回:**
- `PipelineResult | None`

### update_database_cache()

同步识别器缓存与数据库，返回是否重建缓存。

### detect_only(frame)

仅运行检测。

### extract_embeddings(frame)

运行检测 + 特征提取（不追踪）。

### reset_tracker()

重置追踪器。
