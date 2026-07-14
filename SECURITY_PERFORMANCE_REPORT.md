# FaceVision 安全与性能审计报告

> **审计日期：** 2026-07-14
> **项目版本：** face_hub v1.0.5
> **审计范围：** 全部源代码（12 个源文件 + 10 个测试文件）、配置文件、CI/CD 工作流

---

## 📊 问题总览

| 严重程度 | 数量 | 说明 |
|----------|------|------|
| 🔴 **严重 (CRITICAL)** | 1 | Pickle 反序列化导致远程代码执行 |
| 🟠 **高危 (HIGH)** | 1 | 路径穿越导致任意文件删除 |
| 🟡 **中危 (MEDIUM)** | 6 | 数据一致性、输入校验、CI 安全等 |
| 🔵 **低危 (LOW)** | 7 | 性能优化、异常处理、资源管理等 |

---

## 🔴 严重 (CRITICAL)

### C1. Pickle 反序列化 — 远程代码执行 (RCE)

**文件：** [face_database.py:124](face_hub/engine/face_database.py#L124)

```python
# line 124
self.encodings = pickle.load(f)
```

**问题描述：**
Python 的 `pickle.load()` 可以在反序列化时执行任意代码。如果攻击者能够替换或篡改 `encodings.pkl` 文件（通过供应链攻击、共享文件系统、社会工程等），在下次实例化 `FaceDatabase` 时即可实现**远程代码执行**。

**攻击场景：**
1. 攻击者篡改 `encodings.pkl` 文件，植入恶意 payload
2. 用户运行程序，`FaceDatabase.__init__()` → `load()` → `pickle.load(f)`
3. 恶意代码以当前用户权限执行

**修复建议：**

```python
# 方案一（推荐）：使用 numpy 的安全格式
import numpy as np

def save(self):
    # ... 保存 JSON 部分不变 ...
    np.save(self.encoding_path, np.array(self.encodings, dtype=object))

def load(self):
    # ... 加载 JSON 部分不变 ...
    if self.encoding_path.exists():
        data = np.load(self.encoding_path, allow_pickle=False)
        self.encodings = [np.array(e, dtype=np.float32) for e in data]
```

```python
# 方案二：如必须使用 pickle，添加 HMAC 签名验证
import hmac, hashlib, secrets

def _sign_file(self, path: Path) -> str:
    key = secrets.token_bytes(32)  # 应从安全存储获取
    with open(path, "rb") as f:
        return hmac.new(key, f.read(), hashlib.sha256).hexdigest()
```

**优先级：🔴 立即修复**

---

## 🟠 高危 (HIGH)

### H1. 路径穿越 — 任意文件删除

**文件：** [face_database.py:54-56](face_hub/engine/face_database.py#L54-L56)

```python
# line 54-56
img_path = Path(person["image_path"])
if img_path.exists():
    img_path.unlink(missing_ok=True)
```

**问题描述：**
`remove_person()` 和 `remove_persons()` 方法从 JSON 数据库读取 `image_path` 并直接删除该文件，**没有任何路径校验**。如果 `face_db.json` 被篡改，攻击者可以设置 `image_path` 为 `../../important_file` 来删除任意文件。

**同样受影响：**
- [face_database.py:74-76](face_hub/engine/face_database.py#L74-L76) — `remove_persons()`
- [face_database.py:133-134](face_hub/engine/face_database.py#L133-L134) — `clear()`

**修复建议：**

```python
def _validate_image_path(self, image_path: str) -> Path:
    """确保 image_path 在安全目录内"""
    safe_dir = self.db_path.parent.resolve()
    resolved = Path(image_path).resolve()
    if not resolved.is_relative_to(safe_dir):
        raise ValueError(f"Image path {image_path} is outside safe directory")
    return resolved

def remove_person(self, name: str) -> bool:
    # ...
    img_path = self._validate_image_path(person["image_path"])
    if img_path.exists():
        img_path.unlink(missing_ok=True)
```

**优先级：🟠 尽快修复**

---

## 🟡 中危 (MEDIUM)

### M1. 非原子双文件保存 — 数据损坏风险

**文件：** [face_database.py:104-113](face_hub/engine/face_database.py#L104-L113)

**问题描述：**
`save()` 方法分别写入 `face_db.json` 和 `encodings.pkl`，两个操作不是原子的。如果进程在两次写入之间崩溃，JSON（人员列表）和 pickle（编码向量）将不同步，导致下次加载时**人名与编码错位**，产生静默误识别。

**修复建议：**

```python
def save(self):
    import tempfile
    # 写入临时文件，然后原子重命名
    tmp_json = self.db_path.with_suffix(".json.tmp")
    tmp_enc = self.encoding_path.with_suffix(".npy.tmp")

    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(self.persons, f, ensure_ascii=False, indent=2)
    np.save(tmp_enc, np.array(self.encodings, dtype=object))

    tmp_json.replace(self.db_path)      # 原子操作
    tmp_enc.replace(self.encoding_path)  # 原子操作
```

---

### M2. 生物特征数据未加密存储

**文件：** [face_database.py:104-113](face_hub/engine/face_database.py#L104-L113)

**问题描述：**
人脸编码（512 维生物特征向量）以明文形式存储在 `encodings.pkl` 中，`face_db.json` 使用默认文件权限。在 GDPR、BIPA、CCPA 等生物特征隐私法规管辖下，这可能构成合规违规。

**修复建议：**
- 写入文件时设置限制性权限：`os.chmod(path, 0o600)`
- 提供可选的静态加密功能
- 在文档中明确告知用户数据存储方式

---

### M3. 缺少输入验证

**受影响位置：**

| 文件 | 行号 | 问题 |
|------|------|------|
| [face_database.py:37](face_hub/engine/face_database.py#L37) | `add_person()` | 未验证 `encoding` 是否为 `(512,)` 形状的 numpy 数组 |
| [face_database.py:37](face_hub/engine/face_database.py#L37) | `add_person()` | 未验证 `name` 是否为非空字符串 |
| [face_recognizer.py:96-108](face_hub/engine/face_recognizer.py#L96-L108) | `recognize()` | 未验证 `unknown_encoding` 维度是否为 512 |
| [face_detector.py:25](face_hub/engine/face_detector.py#L25) | 构造函数 | `confidence`、`det_size`、`min_face_size` 无范围校验 |
| [face_tracker.py:155](face_hub/engine/face_tracker.py#L155) | 构造函数 | `smooth_frames`、`iou_threshold`、`max_missed` 无范围校验 |
| [camera.py:29](face_hub/engine/camera.py#L29) | 构造函数 | `camera_id`、`width`、`height`、`fps` 无范围校验 |

**修复建议（示例）：**

```python
def add_person(self, name: str, encoding: np.ndarray, image_path: str = ""):
    if not name or not isinstance(name, str):
        raise ValueError("name must be a non-empty string")
    if not isinstance(encoding, np.ndarray) or encoding.shape != (512,):
        raise ValueError(f"encoding must be a numpy array of shape (512,), got {encoding.shape}")
    # ...
```

---

### M4. GitHub Actions 权限过宽

**文件：** [.github/workflows/docs.yml:19-20](.github/workflows/docs.yml#L19-L20)

```yaml
# line 19-20（顶层）
permissions:
  contents: write  # build 任务只需要 read
```

**修复建议：**
将顶层权限改为 `contents: read`，仅在 `deploy` 任务中授予 `contents: write`。

---

### M5. GitHub Actions 未锁定版本 SHA

**文件：** [docs.yml](.github/workflows/docs.yml) 和 [publish.yml](.github/workflows/publish.yml)

所有 Actions 使用标签引用（如 `actions/checkout@v4`），标签可被上游仓库篡改。

**修复建议：**
锁定到完整 SHA 哈希：

```yaml
- uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11  # v4
```

---

### M6. 模块级环境变量修改

**文件：** [camera.py:21](face_hub/engine/camera.py#L21)

```python
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
```

**问题描述：**
在 `import` 时永久修改进程环境变量，可能影响同一进程中其他使用 OpenCV 的库。

**修复建议：**
移入 `CameraThread.__init__()` 或 `start()` 方法中，并在退出时恢复原值。

---

## 🔵 低危 (LOW)

### L1. 每帧同步数据库缓存（性能）

**文件：** [pipeline.py:133-136](face_hub/pipeline.py#L133-L136)

**问题描述：**
`_process_frame_impl()` 在每一帧都调用 `db.get_encodings_and_names()` 和 `recognizer.update_cache()`，即使数据库未变化。以 30fps 运行时，每秒产生 30 次不必要的函数调用开销。

**修复建议：**

```python
# 在 Pipeline 类中缓存版本号
self._last_db_version = -1

def _process_frame_impl(self, frame):
    current_version = self.db.version
    if current_version != self._last_db_version:
        encodings, names = self.db.get_encodings_and_names()
        self.recognizer.update_cache(encodings, names)
        self._last_db_version = current_version
    # ...
```

---

### L2. 每次变更都写磁盘（性能）

**文件：** [face_database.py:47,62,93](face_hub/engine/face_database.py#L47)

**问题描述：**
`add_person()`、`remove_person()`、`remove_persons()` 每次调用都触发完整的 JSON 序列化 + pickle 写入。批量注册人员时性能较差。

**修复建议：**
提供批量方法或上下文管理器：

```python
@contextmanager
def batch_mode(self):
    self._defer_save = True
    try:
        yield self
    finally:
        self._defer_save = False
        self.save()
```

---

### L3. 守护线程无资源清理保证

**文件：** [camera.py:85](face_hub/engine/camera.py#L85)

**问题描述：**
摄像头线程使用 `daemon=True`，主程序退出时线程被强制终止，`cv2.VideoCapture` 可能未正确释放，在某些平台上会锁定摄像头设备。

**修复建议：**
添加 `atexit` 钩子或实现上下文管理器：

```python
import atexit

def start(self):
    # ...
    atexit.register(self.stop)
```

---

### L4. 异常被静默吞没

**文件：** [face_tracker.py:186-187](face_hub/engine/face_tracker.py#L186-L187)

```python
except Exception:
    pass
```

**修复建议：**

```python
except Exception:
    logger.debug("Recognition failed for detection", exc_info=True)
```

---

### L5. 异常消息泄露内部细节

**受影响文件：**
- [pipeline.py:176](face_hub/pipeline.py#L176)
- [face_detector.py:84](face_hub/engine/face_detector.py#L84)
- [face_detector.py:178](face_hub/engine/face_detector.py#L178)
- [face_database.py:113](face_hub/engine/face_database.py#L113)
- [face_database.py:128](face_hub/engine/face_database.py#L128)

**问题描述：**
异常消息中包含原始错误详情（文件路径、内存地址等）。作为库这是可接受的，但调用方如果通过 HTTP 返回这些消息，需要进行脱敏处理。

---

### L6. 依赖版本无上界约束

**文件：** [pyproject.toml:28-34](pyproject.toml#L28-L34)

所有依赖使用 `>=` 而无上界限制，用户可能安装到存在已知 CVE 的版本或不兼容的大版本。

**修复建议：**

```toml
dependencies = [
    "opencv-python>=4.8.0,<5",
    "insightface>=0.7.3,<1",
    "onnxruntime>=1.18.0,<2",
    "numpy>=1.24.0,<3",
    "Pillow>=10.0.0,<11",
]
```

---

### L7. `recognize()` 中的数组拷贝（性能）

**文件：** [face_recognizer.py:85](face_hub/engine/face_recognizer.py#L85)

**问题描述：**
当显式传入 `known_encodings` 时，`np.array(known_encodings, dtype=np.float32)` 每次调用都会创建完整拷贝。缓存路径避免了此问题，但公开 API 仍暴露了这一低效路径。

---

## ✅ 安全亮点

| 项目 | 状态 |
|------|------|
| 硬编码凭证/密钥 | ✅ 未发现 |
| SQL 注入 | ✅ 不适用（使用 JSON + 文件存储） |
| 命令注入 | ✅ 未发现（无 subprocess/eval/exec） |
| SSRF | ✅ 不适用（无网络请求） |
| 弱加密算法 | ✅ 不适用（余弦相似度匹配，符合领域惯例） |
| `.env` 文件泄露 | ✅ 未发现 `.env` 文件 |
| `.gitignore` 配置 | ✅ 正确排除了敏感文件 |

---

## 🛠️ 修复优先级建议

| 优先级 | 编号 | 问题 | 预估工作量 |
|--------|------|------|-----------|
| **P0** | C1 | Pickle 反序列化 RCE | 1-2 小时 |
| **P1** | H1 | 路径穿越文件删除 | 30 分钟 |
| **P1** | M1 | 非原子保存 | 1 小时 |
| **P2** | M3 | 输入验证 | 1-2 小时 |
| **P2** | M6 | 环境变量修改 | 15 分钟 |
| **P3** | M2,M4,M5 | 生物数据加密/CI 安全 | 2-4 小时 |
| **P4** | L1-L7 | 性能与代码质量 | 3-5 小时 |

---

## 📝 总结

FaceVision 项目整体代码质量良好，无硬编码凭证、无命令注入、无 SQL 注入等常见安全问题。**最紧迫的问题是 `pickle.load()` 导致的远程代码执行风险**，建议用 `numpy.save()`/`numpy.load()` 替换，改动量小且彻底消除攻击面。其次是路径穿越漏洞和数据一致性问题，均属于可通过简单输入校验和原子写入修复的问题。

对于一个计算机视觉库而言，性能方面的关注点（每帧缓存同步、磁盘 I/O 频率）在小规模使用场景下影响有限，但在高帧率或批量处理场景下建议优化。
