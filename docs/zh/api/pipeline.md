# FaceHubPipeline

`FaceHubPipeline` 是 FaceHub 的核心流水线，它把摄像头、检测器、识别器、追踪器和数据库串联成一个完整的处理循环。

它负责：
- 启动和停止摄像头
- 将识别器缓存与数据库同步
- 完成检测 → 识别 → 追踪的完整流程
- 实时 FPS 统计
- 线程安全的帧处理

---

## 构造参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `camera` | `CameraThread` | 摄像头采集线程 |
| `detector` | `DetectorProtocol` | 检测器（可用内置 `FaceDetector` 或自定义实现） |
| `recognizer` | `FaceRecognizer` | 识别器 |
| `tracker` | `FaceTracker` | 追踪器 |
| `db` | `FaceDatabase` | 人脸数据库 |

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `is_running` | `bool` | 流水线是否已启动 |

## 方法

### start()

启动流水线。如果摄像头尚未运行，会自动启动。

### stop()

停止流水线并释放摄像头。

### process_frame(frame=None)

处理一帧图像，走完整流水线（检测 → 特征提取 → 识别 → 追踪）。

**参数:**
- `frame` (`np.ndarray | None`): 显式传入的 BGR 帧；为 `None` 时自动从摄像头获取。

**返回:**
- `PipelineResult | None`：如果无帧可取则返回 `None`。

### update_database_cache()

同步识别器缓存与数据库。

**返回:**
- `bool`: 缓存是否被重建。

### detect_only(frame)

仅运行检测（不提取特征、不识别、不追踪）。

**参数:**
- `frame` (`np.ndarray`): BGR 图像。

**返回:**
- `List[DetectionResult]`

### extract_embeddings(frame)

运行检测 + 特征提取（不追踪）。

**参数:**
- `frame` (`np.ndarray`): BGR 图像。

**返回:**
- `List[DetectionWithEmbedding]`

### reset_tracker()

重置追踪器，清除所有活跃追踪。

---

## 完整实时摄像头示例

这是 FaceHub 最典型的使用方式——打开摄像头实时识别人脸：

```python
import cv2
from face_hub import (
    FaceHubPipeline,
    FaceDetector,
    FaceRecognizer,
    FaceTracker,
    FaceDatabase,
    CameraThread,
)

# 1. 初始化各组件
camera = CameraThread(camera_id=0, width=640, height=360)
detector = FaceDetector(device="auto", det_size=640)
recognizer = FaceRecognizer(tolerance=0.45)
tracker = FaceTracker(smooth_frames=5)
db = FaceDatabase(db_path="face_db.json")

# 2. 创建并启动流水线
pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
pipeline.start()

try:
    while True:
        result = pipeline.process_frame()
        if result is None:
            continue

        # 遍历已识别的人脸
        for face in result.known_faces:
            print(f"{face.name} 置信度={face.confidence:.0%}")

        # 绘制结果
        for face in result.tracked_faces:
            x1, y1, x2, y2 = face.bbox.to_tuple()
            if face.is_known:
                color = (0, 255, 0)
                label = f"{face.name} {face.confidence:.0%}"
            else:
                color = (0, 165, 255)
                label = "unknown"
            cv2.rectangle(result.frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result.frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示 FPS 和人脸数量
        cv2.putText(result.frame, f"FPS: {result.fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(result.frame, f"Faces: {result.total_faces}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("FaceHub", result.frame)
        if cv2.waitKey(1) == 27:  # 按 ESC 退出
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
```

## 处理静态图片

```python
import cv2
from face_hub import (
    FaceHubPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread,
)

camera = CameraThread()
pipeline = FaceHubPipeline(
    camera,
    FaceDetector(device="cpu"),
    FaceRecognizer(),
    FaceTracker(),
    FaceDatabase(),
)
pipeline.start()

frame = cv2.imread("group_photo.jpg")
result = pipeline.process_frame(frame=frame)

if result:
    print(f"共检测到 {result.total_faces} 张人脸")
    print(f"已识别: {len(result.known_faces)} 人")
    print(f"未识别: {result.unknown_count} 人")

    for face in result.known_faces:
        print(f"  {face.name}: 置信度 {face.confidence:.0%}")

pipeline.stop()
```

## 用 OpenCV 绘制识别结果

```python
result = pipeline.process_frame()
if result is None:
    return

frame = result.frame
for face in result.tracked_faces:
    x1, y1, x2, y2 = face.bbox.to_tuple()

    # 已确认身份用绿色，未确认用橙色
    if face.is_known:
        color = (0, 255, 0)        # 绿色
    elif face.is_confirmed:
        color = (0, 165, 255)      # 橙色（追踪已确认但无匹配）
    else:
        color = (0, 0, 255)        # 红色（未确认）

    label = f"{face.name} {face.confidence:.0%}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + 120, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
```

## 仅检测（不做识别/追踪）

当只需要知道人脸在哪里，不需要知道是谁时：

```python
pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
pipeline.start()

frame = cv2.imread("crowd.jpg")
detections = pipeline.detect_only(frame)

print(f"检测到 {len(detections)} 张人脸")
for det in detections:
    x1, y1, x2, y2 = det.bbox.to_tuple()
    print(f"  人脸框: ({x1},{y1},{x2},{y2}) 置信度={det.confidence:.2f}")
```

## 提取特征但不追踪

适合批量处理照片、建立人脸库：

```python
frame = cv2.imread("photo.jpg")
faces = pipeline.extract_embeddings(frame)

for face in faces:
    if face.has_embedding:
        print(f"特征维度: {face.embedding.shape}")  # (512,)
        print(f"数据类型: {face.embedding.dtype}")    # float32
        print(f"L2 范数: {np.linalg.norm(face.embedding):.4f}")  # ≈ 1.0

# 可将特征存入数据库
if faces and faces[0].has_embedding:
    db.add_person("Alice", "alice.jpg", faces[0].embedding)
    pipeline.update_database_cache()  # 刷新识别器缓存
```

## 动态注册新用户

运行时实时注册新用户：

```python
import cv2

def register_person(pipeline, db, name, photo_path):
    """从照片中提取人脸特征并注册到数据库"""
    frame = cv2.imread(photo_path)
    if frame is None:
        print(f"无法读取图片: {photo_path}")
        return False

    faces = pipeline.extract_embeddings(frame)
    if not faces:
        print("未检测到人脸")
        return False

    face = faces[0]  # 取置信度最高的人脸
    if not face.has_embedding:
        print("特征提取失败")
        return False

    ok, msg = db.add_person(name, photo_path, face.embedding)
    if ok:
        pipeline.update_database_cache()
        print(f"成功注册: {name}")
    else:
        print(f"注册失败: {msg}")
    return ok

# 用法
register_person(pipeline, db, "Alice", "photos/alice.jpg")
register_person(pipeline, db, "Bob", "photos/bob.jpg")
```

## 调整追踪器参数后重置

```python
# 修改追踪器配置后需要重置
pipeline.tracker = FaceTracker(smooth_frames=3)  # 快速响应
pipeline.reset_tracker()
```

## 错误处理

```python
from face_hub.exceptions import FaceHubError
from face_hub import FaceHubPipeline

pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
pipeline.start()

try:
    while True:
        try:
            result = pipeline.process_frame()
        except FaceHubError as e:
            print(f"流水线错误（已跳过本帧）: {e}")
            continue

        if result is None:
            continue

        # 处理 result ...

finally:
    pipeline.stop()
```

## 完整应用模版

将以上所有功能整合到一个完整的应用程序中：

```python
import cv2
from face_hub import (
    FaceHubPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread,
)
from face_hub.exceptions import FaceHubError

def main():
    # === 初始化 ===
    camera = CameraThread(camera_id=0, width=640, height=360, fps=30)
    detector = FaceDetector(device="auto", det_size=640, confidence=0.50)
    recognizer = FaceRecognizer(tolerance=0.45)
    tracker = FaceTracker(smooth_frames=5)
    db = FaceDatabase()

    # 加载已有的人脸库（如果存在）
    try:
        encodings, names = db.get_encodings_and_names()
        recognizer.update_cache(encodings, names, db.version)
        print(f"已加载 {len(names)} 个注册用户: {names}")
    except FaceHubError:
        print("人脸库为空或加载失败")

    # === 启动流水线 ===
    pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
    pipeline.start()

    print("FaceHub 已启动，按 ESC 退出，按 'r' 重新同步数据库...")
    print(f"实际 FPS: {camera.actual_fps:.1f}")

    try:
        while True:
            result = pipeline.process_frame()
            if result is None:
                continue

            # 绘制结果
            draw_results(result)

            # 显示
            cv2.imshow("FaceHub", result.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r'):
                pipeline.update_database_cache()
                print("已刷新识别器缓存")

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("FaceHub 已停止")

def draw_results(result):
    """在帧上绘制识别结果"""
    for face in result.tracked_faces:
        x1, y1, x2, y2 = face.bbox.to_tuple()
        if face.is_known:
            color = (0, 255, 0)
            label = f"{face.name} {face.confidence:.0%}"
        else:
            color = (0, 165, 255)
            label = face.name

        cv2.rectangle(result.frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result.frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(result.frame, f"FPS: {result.fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

if __name__ == "__main__":
    main()
```

## 注意事项

- 首次调用 `FaceDetector()` 时会自动下载 insightface 的 `buffalo_l` 模型（约 200 MB）。
- `process_frame()` 返回 `None` 表示当前没有新帧，可直接跳过继续循环。
- 追踪器会在人脸离开画面 `max_missed` 帧后自动移除该追踪。
- `known_faces` 是 `tracked_faces` 的子集，仅包含 `is_known=True` 的人脸。