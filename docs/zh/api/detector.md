# FaceDetector

基于 insightface RetinaFace 的人脸检测器，支持 GPU 自动检测与回退。

## 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `confidence` | `float` | `0.50` | 检测置信度阈值 (0.0~1.0)，低于此值的人脸被丢弃 |
| `device` | `str` | `"auto"` | 推理设备：`"cpu"` / `"cuda"` / `"auto"`。`"auto"` 自动探测最优 GPU |
| `det_size` | `int` | `640` | 检测模型输入尺寸：320(快) / 480(均衡) / 640(精准) |
| `quality_filter` | `bool` | `True` | 是否启用模糊度过滤 |
| `min_face_size` | `int` | `80` | 最小人脸尺寸 (px)，小于此值标记为低质量 |

## 方法

### detect(frame)

仅检测人脸，不提取特征。

**参数:**
- `frame` (`np.ndarray`): BGR 图像 (H, W, 3)

**返回:**
- `List[DetectionResult]`: 检测到的人脸列表

### detect_with_embeddings(frame)

检测人脸并提取 512 维特征向量。

**参数:**
- `frame` (`np.ndarray`): BGR 图像 (H, W, 3)

**返回:**
- `List[DetectionWithEmbedding]`: 人脸列表，含特征向量

### reload_model(det_size=None)

运行时切换检测尺寸。

**参数:**
- `det_size` (`int | None`): 新尺寸，`None` 表示保持当前值

## 自定义检测器

实现 `DetectorProtocol` 接口即可接入自有模型：

```python
from face_hub import DetectorProtocol, DetectionWithEmbedding, BBox

class MyDetector:
    def detect_with_embeddings(self, frame):
        ...  # 你的检测 + 特征提取逻辑
        return [DetectionWithEmbedding(...)]

pipeline = FaceHubPipeline(camera, MyDetector(), ...)
```
