# 配置

FaceHub 使用 `FaceHubSettings` TypedDict 定义所有可配置项，配合 `DEFAULT_SETTINGS` 提供合理的默认值。

库本身不读取任何配置文件。调用方应：
1. 读取自己的配置文件（JSON / YAML / TOML 等）
2. 与 `DEFAULT_SETTINGS` 合并作为回退
3. 将最终值传入各组件的构造函数

---

## FaceHubSettings

类型化的配置字典，所有键均为可选（`total=False`）。

```python
from face_hub import FaceHubSettings

# 部分配置 — 未指定的键从 DEFAULT_SETTINGS 补全
my_config: FaceHubSettings = {
    "device": "cuda",
    "confidence": 0.60,
}
```

### 配置项

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `device` | `"auto" \| "cuda" \| "cpu"` | `"auto"` | 推理设备。`"auto"` 自动探测最优 GPU |
| `confidence` | `float` | `0.50` | 检测置信度阈值 (0, 1] |
| `tolerance` | `float` | `0.45` | 余弦相似度容差：0.40=严格，0.45=推荐，0.50=宽松 |
| `cam_width` | `int` | `640` | 摄像头采集宽度 |
| `cam_height` | `int` | `360` | 摄像头采集高度 |
| `cam_fps` | `int` | `30` | 摄像头采集帧率 |
| `proc_fps` | `int` | `30` | ML 处理帧率上限（0=不限） |
| `det_size` | `int` | `640` | 检测模型输入尺寸：320=快，480=均衡，640=精准 |
| `track_smooth` | `int` | `5` | 追踪平滑帧数：3=快，5=推荐，8=稳定 |
| `min_face_size` | `int` | `80` | 最小人脸尺寸 (px) |
| `quality_filter` | `bool` | `True` | 是否启用模糊度质量过滤 |

---

## DEFAULT_SETTINGS

默认配置常量，类型为 `FaceHubSettings`。

```python
from face_hub import DEFAULT_SETTINGS

print(DEFAULT_SETTINGS)
# {'device': 'auto', 'confidence': 0.50, 'tolerance': 0.45, ...}
```

> **注意：** 直接修改 `DEFAULT_SETTINGS` 会影响全局。如需安全修改，请使用 `get_default_settings()`。

---

## get_default_settings()

返回 `DEFAULT_SETTINGS` 的深拷贝，可安全修改而不影响全局默认值。

```python
from face_hub import get_default_settings

config = get_default_settings()
config["device"] = "cuda"
config["confidence"] = 0.60

# 传入组件
detector = FaceDetector(
    device=config["device"],
    confidence=config["confidence"],
)
```

---

## 典型用法

### 从 JSON 配置文件加载

```python
import json
from face_hub import DEFAULT_SETTINGS, FaceDetector, FaceRecognizer

# 读取用户配置
with open("config.json") as f:
    user_config = json.load(f)

# 合并：用户配置优先，缺失项用默认值
config = {**DEFAULT_SETTINGS, **user_config}

# 使用
detector = FaceDetector(
    device=config["device"],
    confidence=config["confidence"],
    det_size=config["det_size"],
)
recognizer = FaceRecognizer(tolerance=config["tolerance"])
```

### 场景化配置模板

```python
from face_hub import get_default_settings

# 实时监控：速度优先
live_config = get_default_settings()
live_config.update({
    "det_size": 320,
    "confidence": 0.45,
    "track_smooth": 3,
    "quality_filter": False,
})

# 人脸注册：精度优先
register_config = get_default_settings()
register_config.update({
    "det_size": 640,
    "confidence": 0.60,
    "min_face_size": 100,
    "quality_filter": True,
})
```

---

## 注意事项

- `DEFAULT_SETTINGS` 是模块级常量，直接修改会影响所有后续使用。建议用 `get_default_settings()` 获取副本。
- `dict.update()` 是浅合并。如果 `FaceHubSettings` 未来增加嵌套结构，需改用递归深合并。
- 配置项仅提供类型提示和默认值，实际校验在各组件的构造函数中完成。
