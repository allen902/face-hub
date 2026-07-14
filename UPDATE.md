# v1.0.5 更新日志

> **发布日期：** 2026-07-14

---

## 🔒 安全修复

### 严重 · Pickle 反序列化漏洞修复（CVE 风险）

- **问题：** `FaceDatabase` 使用 `pickle.load()` 加载编码文件，攻击者篡改 `.pkl` 文件可实现远程代码执行 (RCE)。
- **修复：** 编码存储格式从 `.pkl`（pickle）迁移为 `.npy`（numpy 原生安全格式）。`np.load(allow_pickle=False)` 不执行任意代码，彻底消除 RCE 攻击面。
- **兼容性：** 旧版 `.pkl` 文件在首次加载时自动迁移为 `.npy`，无需手动操作。
- **文件：** [face_database.py](face_hub/engine/face_database.py)

### 高危 · 路径穿越漏洞修复

- **问题：** `remove_person()`、`remove_persons()`、`clear()` 直接删除 `image_path` 指向的文件，无路径校验。篡改 JSON 数据库可删除任意文件。
- **修复：** 新增 `_validate_image_path()` 方法，校验路径必须在数据库目录内（`is_relative_to`）。路径越界时跳过删除并记录警告日志。
- **文件：** [face_database.py](face_hub/engine/face_database.py)

---

## 🛡️ 安全加固

### 原子写入

- `save()` 方法改为临时文件 + 原子重命名，防止进程在两次写入之间崩溃导致 JSON 与编码文件不同步。
- **文件：** [face_database.py](face_hub/engine/face_database.py)

### 文件权限收紧

- 数据库文件写入后设置 `0o600` 权限（仅 owner 可读写），防止其他用户读取生物特征数据。

### 输入验证

| 组件 | 校验内容 |
|------|----------|
| `FaceDatabase.add_person()` | `name` 必须为非空字符串；`encoding` shape 必须为 `(512,)` |
| `FaceDetector.__init__()` | `confidence` ∈ (0, 1]；`det_size` ≥ 160；`min_face_size` ≥ 0 |
| `FaceTracker.__init__()` | `smooth_frames` ≥ 1；`iou_threshold` ∈ (0, 1]；`max_missed` ≥ 1 |
| `CameraThread.__init__()` | `camera_id` ≥ 0；`width`/`height` > 0；`fps` > 0 |
| `FaceRecognizer.recognize()` | 编码维度不匹配时安全返回 UNKNOWN，而非 numpy 广播错误 |

### GitHub Actions 安全

- **权限收紧：** `docs.yml` 顶层权限从 `contents: write` 改为 `contents: read`，仅 `deploy` 任务拥有写权限。
- **版本锁定：** 所有 GitHub Actions 引用从标签（`@v4`）锁定到完整 SHA 哈希，防止供应链攻击。
- **文件：** [docs.yml](.github/workflows/docs.yml)、[publish.yml](.github/workflows/publish.yml)

### 依赖版本上界

- 所有依赖添加主版本上界约束（如 `numpy>=1.24.0,<3`），防止大版本不兼容。
- **文件：** [pyproject.toml](pyproject.toml)

---

## ⚡ 性能优化

### Pipeline 缓存同步优化

- `FaceHubPipeline._process_frame_impl()` 现在仅在数据库版本号变化时同步编码缓存，避免每帧执行不必要的函数调用和属性查找（以 30fps 计，每秒减少 ~30 次冗余调用）。
- **文件：** [pipeline.py](face_hub/pipeline.py)

---

## 🔧 代码质量改进

### 异常处理

- `FaceTracker.update()` 中的 `except Exception: pass` 改为 `logger.debug("Recognition failed", exc_info=True)`，不再静默吞没错误。
- **文件：** [face_tracker.py](face_hub/engine/face_tracker.py)

### 环境变量副作用

- `os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"]` 从模块级（import 时执行）移入 `CameraThread.__init__()`，避免污染全局进程环境。
- **文件：** [camera.py](face_hub/engine/camera.py)

### 资源清理

- `CameraThread` 注册 `atexit` 钩子，确保解释器退出时摄像头资源被正确释放。
- **文件：** [camera.py](face_hub/engine/camera.py)

---

## 📖 文档更新

- 中文 API 文档 [database.md](docs/zh/api/database.md)：更新存储格式说明、安全特性、输入验证、路径校验、自动迁移说明。
- 英文 API 文档 [database.md](docs/en/api/database.md)：重写以匹配实际 API（移除不存在的 `update_person`、`list_persons` 等方法）。

---

## ✅ 测试

- 新增 3 个测试用例：
  - `test_add_person_validates_name` — 验证空名字抛出 `ValueError`
  - `test_add_person_validates_encoding_shape` — 验证错误 shape 抛出 `ValueError`
  - `test_encoding_saved_as_npy` — 验证编码以 `.npy` 格式存储
- 更新 `conftest.py`：临时文件后缀从 `.pkl` 改为 `.npy`
- **全部 47 个测试通过**

---

## 📦 升级指南

### 从 v1.0.4 升级

无需手动操作。首次加载旧版 `.pkl` 编码文件时会自动迁移为 `.npy`。

### 破坏性变更

| 变更 | 影响 | 迁移方式 |
|------|------|----------|
| `encoding_path` 默认值从 `encodings.pkl` 改为 `encodings.npy` | 使用默认路径的代码会创建新文件名 | 无需操作，旧文件自动迁移 |
| `add_person()` 现在对 `name` 和 `encoding` 做校验 | 之前传入空 name 或错误 shape 会静默存储 | 无需操作，错误输入会被正确拒绝 |
| 构造函数参数校验更严格 | 传入非法参数（如负数 fps）会立即报错 | 检查传入组件构造函数的参数 |
