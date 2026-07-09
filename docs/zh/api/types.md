# 数据类型

所有公开 API 返回类型均使用 dataclass，保证类型安全。

## BBox

不可变边界框 (x1, y1, x2, y2)。

| 属性 | 类型 | 说明 |
|------|------|------|
| `x1` | `int` | 左上角 x |
| `y1` | `int` | 左上角 y |
| `x2` | `int` | 右下角 x |
| `y2` | `int` | 右下角 y |
| `width` | `int` | 宽度 (x2 - x1) |
| `height` | `int` | 高度 (y2 - y1) |

方法: `to_tuple()` → `(x1, y1, x2, y2)`

## DetectionResult

`detect()` 返回的单条结果。

| 属性 | 类型 | 说明 |
|------|------|------|
| `bbox` | `BBox` | 边界框 |
| `confidence` | `float` | 检测置信度 0.0~1.0 |

## DetectionWithEmbedding

`detect_with_embeddings()` 返回的单条结果。

| 属性 | 类型 | 说明 |
|------|------|------|
| `bbox` | `BBox` | 边界框 |
| `confidence` | `float` | 检测置信度 |
| `embedding` | `np.ndarray \| None` | 512 维 L2 归一化特征向量 |
| `quality_pass` | `bool` | 质量过滤是否通过 |
| `has_embedding` | `bool` | 特征是否提取成功 |

## TrackedFace

`FaceTracker.update()` 返回的单条追踪结果。

| 属性 | 类型 | 说明 |
|------|------|------|
| `track_id` | `int` | 追踪 ID |
| `bbox` | `BBox` | 当前帧边界框 |
| `name` | `str` | 识别姓名，`UNKNOWN_SENTINEL` 表示未匹配 |
| `confidence` | `float` | 余弦相似度 0.0~1.0 |
| `det_confidence` | `float` | 检测置信度 |
| `is_confirmed` | `bool` | 身份是否经多数投票确认 |
| `quality_pass` | `bool` | 质量过滤是否通过 |
| `is_known` | `bool` | 是否为已注册人员 |

## PipelineResult

`FaceHubPipeline.process_frame()` 的完整返回。

| 属性 | 类型 | 说明 |
|------|------|------|
| `frame` | `np.ndarray` | 当前帧 |
| `raw_detections` | `List[DetectionWithEmbedding]` | 原始检测结果 |
| `tracked_faces` | `List[TrackedFace]` | 追踪后的识别结果 |
| `fps` | `float` | 当前处理帧率 |
| `known_faces` | `List[TrackedFace]` | 筛选已识别人员 |
| `unknown_count` | `int` | 未识别的人数 |
| `total_faces` | `int` | 总人脸数 |

## 全局常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `UNKNOWN_SENTINEL` | `"unknown"` | 未匹配到已注册人员时的默认名称 |
