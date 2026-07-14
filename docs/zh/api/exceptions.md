# 异常体系

FaceHub 的所有异常都继承自 `FaceHubError`，便于统一捕获和处理。

---

## 异常层级

```
FaceHubError (基础异常)
 ├── ModelLoadError      — 模型加载失败（下载失败 / ONNX 提供者不可用 / 模型损坏）
 │    └── DependencyError — 缺少必要的 Python 包（insightface / onnxruntime）
 ├── InferenceError      — 推理执行错误（GPU 崩溃 + CPU 回退也失败）
 ├── CameraError         — 摄像头未连接 / 被占用 / 不支持的分辨率
 ├── DatabaseError       — JSON 解析 / 磁盘满 / 权限不足
 │    └── SerializationError — 数据格式错误（JSON 解析失败 / .npy 损坏）
 └── RecognitionError    — 编码维度不匹配 / 缓存为空
```

---

## FaceHubError

所有 FaceHub 异常的基类。可用于统一捕获任何库错误。

```python
from face_hub.exceptions import FaceHubError

try:
    pipeline.process_frame()
except FaceHubError as e:
    print(f"FaceHub 错误: {e}")
```

---

## ModelLoadError

当模型文件下载失败、损坏或请求的 ONNX 执行提供者不可用时抛出。

**属性:**
- `model_name` (`str | None`): 失败的模型名称（如 `"buffalo_l"`、`"RetinaFace"`）。
- `model_path` (`str | None`): 尝试加载的文件路径（如果有）。

```python
from face_hub import FaceDetector
from face_hub.exceptions import ModelLoadError

try:
    detector = FaceDetector(device="cuda")
except ModelLoadError as e:
    print(f"模型加载失败: {e}")
    print(f"  模型: {e.model_name}")
    # 可能的原因:
    #   1. 网络问题导致 buffalo_l 模型下载失败（约 200MB）
    #   2. CUDA 驱动未安装
    #   3. 磁盘空间不足
```

---

## DependencyError

缺少必要的 Python 包时抛出。是 `ModelLoadError` 的子类，因此 `except ModelLoadError` 也能捕获。

```python
from face_hub import FaceDetector
from face_hub.exceptions import DependencyError

try:
    detector = FaceDetector()
except DependencyError as e:
    print(f"缺少依赖: {e}")
    print(f"  包名: {e.model_name}")
    # 安装缺失的包:
    #   pip install insightface
    #   pip install onnxruntime
```

---

## InferenceError

ML 推理过程中抛出 — 例如 GPU 显存溢出、非法输入张量或 ONNX 运行时崩溃。

```python
from face_hub.exceptions import InferenceError

try:
    faces = detector.detect_with_embeddings(frame)
except InferenceError as e:
    print(f"推理错误: {e}")
    # 尝试降低 det_size 或切换到 CPU
```

---

## CameraError

摄像头无法打开、读取或配置时抛出。

```python
from face_hub import CameraThread
from face_hub.exceptions import CameraError

try:
    camera = CameraThread(camera_id=0)
    camera.start()
except CameraError as e:
    print(f"摄像头错误: {e}")
```

---

## DatabaseError

数据库读写失败时抛出 — 磁盘满、权限问题或文件锁定错误。

**属性:**
- `db_path` (`str | None`): 失败的数据库文件路径。

```python
from face_hub import FaceDatabase
from face_hub.exceptions import DatabaseError

try:
    db = FaceDatabase(db_path="/read_only/face_db.json")
except DatabaseError as e:
    print(f"数据库错误: {e}")
    print(f"  路径: {e.db_path}")
```

---

## SerializationError

数据格式错误时抛出 — JSON 解析失败、`.npy` 编码文件损坏或旧版 pickle 格式问题。
同时是 `DatabaseError` 和 `ValueError` 的子类，可在任一层级捕获。

```python
from face_hub import FaceDatabase
from face_hub.exceptions import SerializationError

try:
    db = FaceDatabase(db_path="corrupt.json")
except SerializationError as e:
    print(f"数据格式错误: {e}")
    # 可能的原因:
    #   1. JSON 文件损坏
    #   2. .npy 编码文件损坏
    #   3. 数据格式不兼容
```

---

## RecognitionError

识别匹配失败时抛出 — 编码维度不匹配或识别缓存为空。

```python
from face_hub.exceptions import RecognitionError

try:
    result = pipeline.process_frame()
except RecognitionError as e:
    print(f"识别错误: {e}")
```

---

## 统一错误处理示例

在生产环境中，包裹每次 `process_frame()` 调用，防止单个错误帧中断整个循环：

```python
from face_hub.exceptions import FaceHubError, InferenceError

while True:
    try:
        result = pipeline.process_frame()
        if result is None:
            continue
        # ... 渲染 / 业务逻辑 ...

    except InferenceError as e:
        print(f"推理错误（跳过本帧）: {e}")
        continue
    except FaceHubError as e:
        print(f"运行时错误（重试）: {e}")
        continue
```

---

## GPU 回退模式

```python
from face_hub import FaceDetector
from face_hub.exceptions import ModelLoadError

def create_detector():
    """按优先级尝试设备"""
    for device in ["cuda", "cpu"]:
        try:
            detector = FaceDetector(device=device)
            print(f"使用设备: {device}")
            return detector
        except ModelLoadError:
            continue
    raise SystemExit("所有设备均不可用")
```

---

## 摄像头自动发现

```python
from face_hub import CameraThread
from face_hub.exceptions import CameraError

def open_first_camera():
    """自动尝试打开第一个可用的摄像头"""
    for cam_id in range(5):
        try:
            camera = CameraThread(camera_id=cam_id)
            camera.start()
            return camera
        except CameraError:
            continue
    raise SystemExit("未找到可用摄像头")
```

---

## 注意事项

- 所有异常均可通过 `from face_hub.exceptions import FaceHubError` 统一捕获。
- 子类异常提供更精细的错误处理能力。
- `ModelLoadError` 和 `InferenceError` 发生在模型/推理阶段；其他异常与业务流程相关。
- 生产环境中，建议用 `try/except FaceHubError` 包裹每次 `process_frame()` 调用，防止单个错误中断整个流水线。
