# FaceTracker

基于 IoU 的轻量级多目标人脸追踪器，配合多数投票进行身份平滑。

追踪器同时解决两个问题：
1. **时序一致性**：为同一个人在不同帧中保持相同的 `track_id`，即使检测偶尔闪烁。
2. **身份平滑**：只有在识别器跨多帧连续一致时才确认身份，抑制单帧误识别。

---

## 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `smooth_frames` | `int` | `5` | 身份确认所需的滑动窗口帧数 |
| `iou_threshold` | `float` | `0.30` | 检测框与追踪框匹配的 IoU 阈值 |
| `max_missed` | `int` | `10` | 连续未匹配多少帧后删除该追踪 |

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `track_count` | `int` | 当前活跃的追踪数量 |

## 方法

### update(detections, recognizer=None)

用当前帧的检测结果更新追踪器。

**参数:**
- `detections` (`List[DetectionWithEmbedding]`): 检测结果
- `recognizer` (`FaceRecognizer | None`): 识别器，用于对每个检测做 1:N 识别；为 `None` 时不识别

**返回:**
- `List[TrackedFace]`: 追踪结果列表

### reset()

清空所有追踪。追踪 ID 计数器归零。

---

## 算法说明

1. 每帧的检测结果通过 IoU 与已有追踪进行匹配。
2. 每个追踪维护最近 `smooth_frames` 帧的识别结果。
3. 当某个名字在窗口内获得多数票，**且**平均余弦相似度 ≥ 0.30 时，身份被"确认"。
4. 未确认时显示最新的识别名（含不确定性）。
5. 连续 `max_missed` 帧未匹配到检测的追踪会被移除。

---

## 基础用法

```python
from face_hub import FaceDetector, FaceRecognizer, FaceTracker

detector = FaceDetector(device="cpu")
recognizer = FaceRecognizer()
tracker = FaceTracker(smooth_frames=5)

# 初始化识别器（加载已注册人员）
recognizer.update_cache([alice_emb, bob_emb], ["Alice", "Bob"], db_version=1)

for frame in frames:
    detections = detector.detect_with_embeddings(frame)
    tracked = tracker.update(detections, recognizer=recognizer)

    for face in tracked:
        print(f"ID={face.track_id}, 姓名={face.name}, 已确认={face.is_confirmed}")
```

## 绘制追踪结果

```python
import cv2

for face in tracked:
    x1, y1, x2, y2 = face.bbox.to_tuple()

    # 已确认身份用绿色，未确认用橙色
    color = (0, 255, 0) if face.is_confirmed else (0, 165, 255)
    label = f"{face.name} {face.confidence:.0%}"

    # 绘制边框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 绘制姓名标签背景
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + 150, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
```

## 仅追踪（不识别）

不传入识别器时，追踪器只追踪边框位置和 ID，不尝试识别身份：

```python
# recognizer=None：仅追踪边界框
for frame in frames:
    detections = detector.detect_with_embeddings(frame)
    tracked = tracker.update(detections, recognizer=None)

    for face in tracked:
        # 所有 face.name 都是 UNKNOWN_SENTINEL
        # 但 track_id 仍然跨帧稳定
        print(f"ID={face.track_id} @ {face.bbox.to_tuple()}")
```

## 配置变更后重置

修改追踪器参数或切换识别器后，应重置追踪状态：

```python
# 切换识别器
pipeline.recognizer = FaceRecognizer(tolerance=0.50)

# 重置追踪器（清除所有活跃追踪和身份历史）
tracker.reset()
# 或
pipeline.reset_tracker()
```

## 参数调优指南

| 参数 | 减小 | 增大 |
|------|------|------|
| `smooth_frames` | 身份确认更快，可能更多抖动 | 身份确认更慢，更稳定 |
| `iou_threshold` | 匹配更严格，追踪更频繁切换 | 匹配更宽松，追踪更少切换 |
| `max_missed` | 短记忆，遮挡后人脸快速移除 | 长记忆，短暂遮挡后仍能恢复 |

```python
# 人多、移动快的场景
fast_tracker = FaceTracker(
    smooth_frames=3,      # 快速确认身份
    iou_threshold=0.25,   # 宽松的匹配
    max_missed=5,         # 短暂离开即移除
)

# 稳定、高质量的视频流
stable_tracker = FaceTracker(
    smooth_frames=8,      # 缓慢确认，更稳定
    iou_threshold=0.35,   # 严格的匹配
    max_missed=15,        # 遮挡后仍能恢复
)

# 默认推荐
default_tracker = FaceTracker(
    smooth_frames=5,
    iou_threshold=0.30,
    max_missed=10,
)
```

## 完整视频流处理

```python
import cv2
from face_hub import FaceDetector, FaceRecognizer, FaceTracker, FaceDatabase

detector = FaceDetector(device="auto")
recognizer = FaceRecognizer(tolerance=0.45)
tracker = FaceTracker(smooth_frames=5)
db = FaceDatabase()

# 同步数据库
encodings, names = db.get_encodings_and_names()
recognizer.update_cache(encodings, names, db.version)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 检测 + 追踪 + 识别
    detections = detector.detect_with_embeddings(frame)
    tracked_faces = tracker.update(detections, recognizer=recognizer)

    # 绘制结果
    for face in tracked_faces:
        x1, y1, x2, y2 = face.bbox.to_tuple()

        # 按状态着色
        if face.is_known:
            color = (0, 255, 0)        # 绿色 = 已识别
        elif face.is_confirmed:
            color = (0, 165, 255)      # 橙色 = 追踪已确认但无匹配
        else:
            color = (0, 0, 255)        # 红色 = 未确认

        # 标签
        status = "✓" if face.is_confirmed else "?"
        label = f"[{face.track_id}] {face.name} {status}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 显示追踪统计
    cv2.putText(frame, f"Tracks: {tracker.track_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Face Tracker", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

## 追踪事件处理

可以基于追踪状态触发业务逻辑：

```python
from collections import defaultdict
import time

# 记录每个人的出现时间
appear_time = {}
last_seen = {}

for frame in camera_stream:
    detections = detector.detect_with_embeddings(frame)
    tracked = tracker.update(detections, recognizer=recognizer)

    for face in tracked:
        # 新人出现
        if face.track_id not in appear_time and face.is_known:
            appear_time[face.track_id] = time.time()
            print(f"🟢 {face.name} 出现！")

        # 身份确认
        if face.track_id in appear_time and face.is_confirmed:
            duration = time.time() - appear_time[face.track_id]
            if duration > 3.0:  # 停留超过 3 秒
                print(f"✅ {face.name} 已确认，停留 {duration:.1f} 秒")

        last_seen[face.track_id] = time.time()

    # 检查离开的人
    now = time.time()
    gone = [tid for tid, ts in last_seen.items() if now - ts > 5.0]
    for tid in gone:
        if tid in appear_time:
            print(f"🔴 追踪 {tid} 已离开")
            del appear_time[tid]
            del last_seen[tid]
```

## 注意事项

- `track_id` 是按递增分配的，仅在连续追踪会话中保持稳定。`reset()` 后计数器从 0 重新开始。
- 追踪在身份被"确认"之前不会将 `is_confirmed` 设为 `True`。这可以防止噪声立即影响 UI 状态。
- 只有当多数投票的名字不是 `UNKNOWN_SENTINEL` 时，追踪才被视为"已知"（`is_known=True`）。
- 追踪器不会在帧间"补齐"检测，如果人脸被完全遮挡，追踪会在 `max_missed` 帧后被移除。