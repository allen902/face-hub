# FaceTracker

基于 IoU 的轻量级多目标人脸追踪器，配合多数投票进行身份平滑。

## 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `smooth_frames` | `int` | `5` | 身份确认所需的滑动窗口帧数 |
| `iou_threshold` | `float` | `0.30` | 检测与追踪器匹配的 IoU 阈值 |
| `max_missed` | `int` | `10` | 连续未匹配多少帧后删除追踪 |

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `track_count` | `int` | 当前活跃追踪数量 |

## 方法

### update(detections, recognizer=None)

用当前帧检测结果更新追踪器。

**参数:**
- `detections` (`List[DetectionWithEmbedding]`): 检测结果
- `recognizer` (`FaceRecognizer | None`): 识别器，用于对每个检测做 1:N 识别

**返回:**
- `List[TrackedFace]`: 追踪结果

### reset()

清空所有追踪器。

## 算法说明

1. 每帧检测通过 IoU 与现有追踪器匹配。
2. 每个追踪器维护最近 `smooth_frames` 帧的识别结果。
3. 当某名字在窗口内获得多数票且平均相似度 ≥ 0.30 时，身份被确认。
4. 未确认时显示最新识别名（带问号提示）。
