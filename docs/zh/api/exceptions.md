# 异常

FaceVision 使用自定义异常层级，所有异常均继承自 `FaceVisionError`。

## 异常层级

```
FaceVisionError
├── ModelLoadError
├── InferenceError
├── CameraError
├── DatabaseError
└── RecognitionError
```

## 说明

| 异常 | 触发条件 |
|------|----------|
| `ModelLoadError` | insightface 未安装、无可用 ONNX Provider、模型损坏 |
| `InferenceError` | GPU 推理失败且 CPU 回退也失败 |
| `CameraError` | 摄像头无法打开、被占用、不支持的分辨率 |
| `DatabaseError` | JSON 解析失败、pickle 损坏、磁盘/权限错误 |
| `RecognitionError` | 编码维度不匹配、空缓存等识别错误 |

## 捕获示例

```python
from face_vision.exceptions import FaceVisionError

try:
    detector = FaceDetector(device="cpu")
except FaceVisionError as e:
    print(f"FaceVision error: {e}")
```
