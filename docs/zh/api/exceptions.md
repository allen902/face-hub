# 异常体系

FaceHub 的所有异常都继承自 `FaceHubError`，便于统一捕获和处理。

---

## 异常层级

```
FaceHubError (基础异常)
 ├── FaceHubConfigError    — 配置错误
 ├── FaceHubRuntimeError   — 运行时错误
 ├── ModelLoadError        — 模型加载失败
 ├── InferenceError        — 推理执行错误
 ├── CameraError           — 摄像头相关错误
 └── DatabaseError         — 数据库操作错误
```

---

## FaceHubError

### FaceHubConfigError

配置相关的错误。例如传入了无效的参数值。

```python
from face_hub import FaceDetector
from face_hub.exceptions import FaceHubConfigError

try:
    detector = FaceDetector(confidence=1.5)  # 置信度必须在 0~1 之间
except FaceHubConfigError as e:
    print(f"配置错误: {e}")
```

---

## 运行时错误

### FaceHubRuntimeError

流水线或组件在执行过程中发生的运行时错误。

```python
from face_hub.exceptions import FaceHubRuntimeError

try:
    pipeline.process_frame()
except FaceHubRuntimeError as e:
    print(f"运行时错误: {e}")
    # 根据错误决定继续或中止
```

---

## 模型错误

### ModelLoadError

当模型文件下载失败、损坏或路径不可访问时抛出。首次使用 `FaceDetector` 时需要下载约 200MB 的 `buffalo_l` 模型。

```python
from face_hub import FaceDetector
from face_hub.exceptions import ModelLoadError

try:
    detector = FaceDetector(device="cuda")
except ModelLoadError as e:
    print(f"模型加载失败: {e}")
    print("可能的原因:")
    print("  1. 网络问题导致模型下载失败")
    print("  2. 磁盘空间不足")
    print("  3. GPU 驱动未安装（device='cuda' 时）")
```

### InferenceError

推理过程中发生的错误，例如显存溢出、非法输入等。

```python
from face_hub.exceptions import InferenceError

try:
    faces = detector.detect_with_embeddings(large_frame)
except InferenceError as e:
    print(f"推理错误: {e}")
    print("可能的原因:")
    print("  1. GPU 显存不足，尝试降低 det_size")
    print("  2. 输入的图像格式不正确")
```

---

## 摄像头错误

### CameraError

摄像头打开、读取或关闭过程中发生的错误。

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

## 数据库错误

### DatabaseError

数据库写入、读取或校验过程中发生的错误。

```python
from face_hub import FaceDatabase
from face_hub.exceptions import DatabaseError

try:
    db = FaceDatabase(db_path="/read_only/face_db.json")
except DatabaseError as e:
    print(f"数据库错误: {e}")
```

---

## 统一错误处理示例

在实际应用中可以统一捕获所有 FaceHub 异常：

```python
from face_hub import (
    FaceHubPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread,
)
from face_hub.exceptions import (
    FaceHubError, FaceHubConfigError, ModelLoadError,
    InferenceError, CameraError, DatabaseError,
)

def initialize_facehub():
    """初始化 FaceHub 各组件的健壮函数"""
    try:
        camera = CameraThread(camera_id=0, width=640, height=360)
    except FaceHubConfigError as e:
        raise SystemExit(f"摄像头配置错误: {e}")
    except CameraError as e:
        raise SystemExit(f"无法打开摄像头: {e}")

    try:
        detector = FaceDetector(device="auto", det_size=640)
    except FaceHubConfigError as e:
        raise SystemExit(f"检测器配置错误: {e}")
    except ModelLoadError as e:
        print(f"GPU 模型加载失败，尝试 CPU: {e}")
        try:
            detector = FaceDetector(device="cpu")
        except ModelLoadError as e2:
            raise SystemExit(f"CPU 模型也加载失败: {e2}")

    recognizer = FaceRecognizer(tolerance=0.45)
    tracker = FaceTracker(smooth_frames=5)

    try:
        db = FaceDatabase()
    except DatabaseError as e:
        raise SystemExit(f"数据库初始化失败: {e}")

    pipeline = FaceHubPipeline(camera, detector, recognizer, tracker, db)
    return pipeline


def main():
    try:
        pipeline = initialize_facehub()
        pipeline.start()
        print("FaceHub 启动成功！")

        while True:
            try:
                result = pipeline.process_frame()
                if result is None:
                    continue

                # 处理结果...
                cv2.imshow("FaceHub", result.frame)
                if cv2.waitKey(1) == 27:
                    break

            except InferenceError as e:
                print(f"推理错误（跳过本帧）: {e}")
                continue
            except FaceHubError as e:
                print(f"运行时错误（重试）: {e}")
                continue

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        try:
            pipeline.stop()
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()
```

## 按场景分类的错误处理

### GPU 回退

```python
from face_hub import FaceDetector
from face_hub.exceptions import ModelLoadError, InferenceError

def create_detector():
    """按优先级尝试设备"""
    for device in ["cuda", "cpu"]:
        try:
            detector = FaceDetector(device=device)
            print(f"✅ 使用设备: {device}")
            return detector
        except ModelLoadError as e:
            print(f"⚠️ {device} 不可用: {e}")
            continue
    raise SystemExit("所有设备均不可用")
```

### 数据库恢复

```python
import os
from face_hub import FaceDatabase
from face_hub.exceptions import DatabaseError
import shutil

def safe_open_database(db_path="face_db.json"):
    """安全打开数据库，损坏时自动恢复"""
    try:
        db = FaceDatabase(db_path=db_path)
        return db
    except DatabaseError as e:
        print(f"数据库加载失败: {e}")

        # 尝试从备份恢复
        backup_path = db_path + ".backup"
        if os.path.exists(backup_path):
            print("正在从备份恢复...")
            shutil.copy2(backup_path, db_path)
            try:
                db = FaceDatabase(db_path=db_path)
                print("恢复成功")
                return db
            except DatabaseError:
                pass

        # 创建新数据库
        print("创建新数据库...")
        os.remove(db_path) if os.path.exists(db_path) else None
        return FaceDatabase(db_path=db_path)
```

### 摄像头自动切换

```python
from face_hub import CameraThread
from face_hub.exceptions import CameraError

def open_first_available_camera():
    """自动尝试打开第一个可用的摄像头"""
    for cam_id in range(5):
        try:
            camera = CameraThread(camera_id=cam_id)
            camera.start()
            print(f"✅ 打开摄像头 {cam_id}")
            return camera
        except CameraError as e:
            print(f"❌ 摄像头 {cam_id} 不可用: {e}")
            continue

    raise SystemExit("未找到可用摄像头")
```

## 注意事项

- 所有 FaceHub 异常都可以通过 `from face_hub.exceptions import FaceHubError` 统一捕获。
- 子类异常提供了更精细的错误信息，方便针对性处理。
- `ModelLoadError` 和 `InferenceError` 通常发生在模型加载/推理阶段；其他异常与业务流程相关。
- 在生产环境中，建议使用 `try/except FaceHubError` 包裹每次 `process_frame()` 调用，避免单个错误帧中断整个流程。