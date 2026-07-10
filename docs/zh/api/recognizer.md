# FaceRecognizer

1:N 人脸识别器，基于余弦相似度。它将查询人脸的特征向量与已注册的人脸库进行比对，返回最佳匹配。

识别器在内部维护一个缓存。每次人脸库发生变化时，应调用 `update_cache()` 来刷新缓存。缓存仅在 `db_version` 变化时才真正重建，这避免了每帧都重复计算矩阵。

---

## 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tolerance` | `float` | `0.45` | 余弦相似度阈值；0.40 严格，0.45 推荐，0.50 宽松 |

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `cached_names` | `List[str]` | 当前缓存中的姓名列表 |
| `tolerance` | `float` | 当前阈值 |

## 方法

### update_cache(known_encodings, known_names, db_version=0)

更新编码缓存。仅在数据库版本变化时才重建。

**参数:**
- `known_encodings` (`List[np.ndarray]`): 已注册的人脸特征向量列表
- `known_names` (`List[str]`): 对应的姓名列表
- `db_version` (`int`): 数据库版本号，用于判断缓存是否需要重建

**返回:**
- `bool`: 缓存是否被重建

### recognize(unknown_encoding, known_encodings=None, known_names=None)

识别单个人脸特征向量。

**参数:**
- `unknown_encoding` (`np.ndarray`): 512 维查询特征向量
- `known_encodings` (`List[np.ndarray] | None`): 显式指定注册库（可选）
- `known_names` (`List[str] | None`): 显式指定姓名（可选）

**返回:**
- `(str, float)`: `(姓名, 置信度)`。未匹配时返回 `(UNKNOWN_SENTINEL, 0.0)`。

---

## 基础识别

```python
import numpy as np
from face_hub import FaceRecognizer
from face_hub.types import UNKNOWN_SENTINEL

recognizer = FaceRecognizer(tolerance=0.45)

# 实际使用时，query 来自 FaceDetector.detect_with_embeddings()
query = np.random.randn(512).astype(np.float32)

name, confidence = recognizer.recognize(query)
if name == UNKNOWN_SENTINEL:
    print("未识别到此人的身份")
else:
    print(f"识别为 {name}，置信度 {confidence:.0%}")
```

## 从数据库同步缓存

这是实际应用中推荐的方式：从数据库加载人脸库，同步到识别器：

```python
from face_hub import FaceDatabase, FaceRecognizer

db = FaceDatabase()
recognizer = FaceRecognizer()

# 从数据库获取注册信息
encodings, names = db.get_encodings_and_names()
rebuilt = recognizer.update_cache(encodings, names, db.version)

if rebuilt:
    print(f"识别器缓存已更新，共 {len(names)} 人: {names}")

# 之后识别时无需再传注册库
name, conf = recognizer.recognize(query_embedding)
```

## 显式传入注册库（不使用缓存）

适合一次性比对、临时测试等场景：

```python
# 已有的注册库
known_encodings = [alice_emb, bob_emb, charlie_emb]
known_names = ["Alice", "Bob", "Charlie"]

# 传入注册库进行比对
name, conf = recognizer.recognize(
    query_embedding,
    known_encodings=known_encodings,
    known_names=known_names,
)

if conf > 0:
    print(f"最相似: {name} ({conf:.0%})")
```

## 缓存版本管理

版本号机制避免不必要的缓存重建，适合频繁调用的场景：

```python
recognizer = FaceRecognizer()

# 首次调用：版本号 1，缓存被重建
rebuilt = recognizer.update_cache([alice_emb], ["Alice"], db_version=1)
print(f"首次重建: {rebuilt}")  # True

# 相同版本号再次调用：不重建
rebuilt = recognizer.update_cache([alice_emb], ["Alice"], db_version=1)
print(f"版本未变: {rebuilt}")  # False

# 版本号变化：重建
rebuilt = recognizer.update_cache(
    [alice_emb, bob_emb], ["Alice", "Bob"], db_version=2
)
print(f"版本更新重建: {rebuilt}")  # True
```

## 阈值调参指南

| 场景 | 推荐 `tolerance` | 效果 |
|------|-----------------|------|
| 高安全性 / 低误接受 | 0.35 ~ 0.40 | 严格，可能拒绝真人 |
| 推荐默认值 | 0.45 | 平衡 |
| 低误拒 / 高召回率 | 0.50 | 宽松，接受更多候选人 |

```python
# 门禁等高安全场景
strict_recognizer = FaceRecognizer(tolerance=0.40)

# 考勤等便利性场景
loose_recognizer = FaceRecognizer(tolerance=0.50)

# 一般使用
default_recognizer = FaceRecognizer(tolerance=0.45)
```

## 批量识别多张人脸

```python
import cv2
from face_hub import FaceDetector, FaceRecognizer, FaceDatabase

detector = FaceDetector(device="cpu")
recognizer = FaceRecognizer(tolerance=0.45)
db = FaceDatabase()

# 同步数据库
encodings, names = db.get_encodings_and_names()
recognizer.update_cache(encodings, names, db.version)

frame = cv2.imread("group_photo.jpg")
faces = detector.detect_with_embeddings(frame)

results = []
for face in faces:
    if face.has_embedding:
        name, confidence = recognizer.recognize(face.embedding)
        results.append({
            "bbox": face.bbox.to_tuple(),
            "name": name,
            "confidence": confidence,
        })

# 按置信度排序
results.sort(key=lambda x: x["confidence"], reverse=True)
for r in results:
    print(f"{r['name']}: {r['confidence']:.0%} @ {r['bbox']}")
```

## 查看缓存内容

```python
recognizer = FaceRecognizer()

encodings, names = db.get_encodings_and_names()
recognizer.update_cache(encodings, names, db.version)

print(f"已注册 {len(recognizer.cached_names)} 人:")
for name in recognizer.cached_names:
    print(f"  - {name}")
```

## 完整集成：检测 + 识别 + 阈值调优

```python
import cv2
from face_hub import FaceDetector, FaceRecognizer, FaceDatabase
from face_hub.types import UNKNOWN_SENTINEL

class FaceAuth:
    """人脸认证系统：根据场景切换阈值"""

    def __init__(self, db_path="face_db.json"):
        self.detector = FaceDetector(device="auto", det_size=640)
        self.db = FaceDatabase(db_path=db_path)
        self.recognizer = FaceRecognizer(tolerance=0.45)  # 默认阈值
        self._sync_db()

    def _sync_db(self):
        """同步数据库到识别器缓存"""
        encodings, names = self.db.get_encodings_and_names()
        self.recognizer.update_cache(encodings, names, self.db.version)

    def authenticate(self, photo_path, mode="normal"):
        """
        认证函数

        Args:
            photo_path: 照片路径
            mode: "strict" (高安全), "normal" (默认), "loose" (宽松)
        """
        # 根据模式调整阈值
        thresholds = {"strict": 0.40, "normal": 0.45, "loose": 0.50}
        self.recognizer.tolerance = thresholds.get(mode, 0.45)

        # 检测并提取特征
        frame = cv2.imread(photo_path)
        faces = self.detector.detect_with_embeddings(frame)

        if not faces:
            return {"success": False, "reason": "未检测到人脸"}

        face = faces[0]  # 取最高置信度的人脸
        if not face.has_embedding:
            return {"success": False, "reason": "特征提取失败"}

        # 识别
        name, confidence = self.recognizer.recognize(face.embedding)
        if name == UNKNOWN_SENTINEL:
            return {"success": False, "reason": "未匹配到已注册用户"}

        return {
            "success": True,
            "name": name,
            "confidence": confidence,
            "mode": mode,
        }

    def register(self, name, photo_path):
        """注册新用户"""
        frame = cv2.imread(photo_path)
        faces = self.detector.detect_with_embeddings(frame)
        if not faces or not faces[0].has_embedding:
            return False

        ok, msg = self.db.add_person(name, photo_path, faces[0].embedding)
        if ok:
            self._sync_db()
        return ok


# 使用示例
auth = FaceAuth()

# 注册
auth.register("Alice", "alice.jpg")
auth.register("Bob", "bob.jpg")

# 认证（严格模式）
result = auth.authenticate("test.jpg", mode="strict")
if result["success"]:
    print(f"认证成功: {result['name']} ({result['confidence']:.0%})")
else:
    print(f"认证失败: {result['reason']}")
```

## 注意事项

- 使用的特征向量需为 L2 归一化的 512 维向量（ArcFace 输出）。
- 余弦相似度通过点积计算，因为人脸库已做了归一化。
- `confidence` 是原始余弦相似度，取值范围 `[0.0, 1.0]`。
- 同时传入 `known_encodings`/`known_names` 和已缓存的库时，优先使用显式传入的库。
- 建议在每次数据库变更后立即调用 `update_database_cache()` 同步缓存。