# FaceDatabase

人脸数据库，负责持久化人员记录与 512 维人脸特征向量。

数据库存储：
- 人员元数据（`name`、`image_path`）到 JSON 文件。
- 人脸特征向量到 pickle 文件。
- 单调递增的 `version` 版本号，下游组件用它来高效地失效缓存。

---

## 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `db_path` | `str` | `"face_db.json"` | 人员记录 JSON 文件路径 |
| `encoding_path` | `str` | `"encodings.pkl"` | 编码 pickle 文件路径 |

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `version` | `int` | 数据库版本号，每次写操作（增删改）后递增 |
| `db_path` | `str` | 当前数据库 JSON 文件路径 |
| `encoding_path` | `str` | 当前编码 pickle 文件路径 |

## 方法

### add_person(name, image_path, encoding)

添加人员到数据库。要求 `name` 在数据库中唯一。

**参数:**
- `name` (`str`): 唯一姓名
- `image_path` (`str`): 参考照片的路径
- `encoding` (`np.ndarray`): 512 维 L2 归一化的人脸特征向量

**返回:**
- `(bool, str)`: `(是否成功, 消息)`。成功时消息为 `"ok"`，失败时说明原因。

### remove_person(name)

删除单个人员。如果对应的参考照片文件存在，也会一并删除。

**参数:**
- `name` (`str`): 要删除的人员姓名

**返回:**
- `(bool, str)`: `(是否成功, 消息)`

### remove_persons(names)

批量删除人员。

**参数:**
- `names` (`List[str]`): 要删除的人员姓名列表

**返回:**
- `(List[str], List[str])`: `(成功删除列表, 未找到列表)`

### get_names()

获取所有已注册人员的姓名列表。

**返回:**
- `List[str]`

### get_encodings_and_names()

获取所有已注册人员的特征向量和姓名。

**返回:**
- `(List[np.ndarray], List[str])`: `(编码列表, 姓名列表)`，两者顺序对应。

### get_person_info(name)

获取某个人员的详细信息。

**参数:**
- `name` (`str`): 姓名

**返回:**
- `dict | None`: 包含 `name` 和 `image_path` 的字典，不存在则返回 `None`。

### clear()

清空数据库并删除持久化文件与所有参考照片。

### save()

显式持久化到磁盘。通常由修改方法（`add_person`、`remove_person` 等）自动调用。

### load()

从磁盘加载数据库。构造时自动调用，也可在数据库文件被外部修改后手动调用。

---

## 基础 CRUD 操作

```python
import numpy as np
from face_hub import FaceDatabase

db = FaceDatabase(db_path="face_db.json", encoding_path="encodings.pkl")

# 实际使用时，编码来自 FaceDetector.detect_with_embeddings()
encoding = np.random.randn(512).astype(np.float32)

# 添加人员
ok, msg = db.add_person("Alice", "photos/alice.jpg", encoding)
print(ok, msg)  # True, "ok"

# 查询所有人员
print(db.get_names())  # ['Alice']

# 查询详细信息
info = db.get_person_info("Alice")
print(info)  # {'name': 'Alice', 'image_path': 'photos/alice.jpg'}

# 删除人员
ok, msg = db.remove_person("Alice")
print(ok, msg)  # True, "deleted"
```

## 重复添加检查

每个 `name` 必须唯一，重复添加会失败：

```python
ok1, msg1 = db.add_person("Alice", "alice.jpg", encoding)
print(ok1, msg1)  # True, "ok"

ok2, msg2 = db.add_person("Alice", "alice2.jpg", another_encoding)
print(ok2, msg2)  # False, "name already exists"
```

## 从真实照片注册用户

完整的注册流程：读取照片 → 检测人脸 → 提取特征 → 存入数据库：

```python
import cv2
from face_hub import FaceDetector, FaceDatabase

detector = FaceDetector(device="cpu")
db = FaceDatabase()

def register_from_photo(name, photo_path):
    """从照片注册新用户"""
    frame = cv2.imread(photo_path)
    if frame is None:
        raise FileNotFoundError(f"无法读取图片: {photo_path}")

    faces = detector.detect_with_embeddings(frame)

    if not faces:
        raise ValueError(f"在 {photo_path} 中未检测到人脸")

    # 使用置信度最高的人脸
    face = faces[0]
    if not face.has_embedding:
        raise ValueError(f"未能从 {photo_path} 提取特征向量")

    ok, msg = db.add_person(name, photo_path, face.embedding)
    if not ok:
        raise ValueError(f"数据库写入失败: {msg}")

    print(f"✅ {name} 注册成功（置信度 {face.confidence:.0%}）")
    return True

# 批量注册
for name, path in [("Alice", "photos/alice.jpg"), ("Bob", "photos/bob.jpg")]:
    try:
        register_from_photo(name, path)
    except Exception as e:
        print(f"❌ 注册失败: {e}")
```

## 批量操作

```python
# 批量删除
removed, not_found = db.remove_persons(["Alice", "Bob", "Charlie"])
print(f"已删除: {removed}")       # ['Alice', 'Bob']
print(f"未找到: {not_found}")     # ['Charlie']（不存在）
```

## 与识别器同步

这是数据库与识别器配合使用的标准模式：

```python
from face_hub import FaceRecognizer

recognizer = FaceRecognizer()

# 从数据库加载并同步到识别器
encodings, names = db.get_encodings_and_names()

if len(names) == 0:
    print("数据库为空，请先注册用户")
else:
    recognizer.update_cache(encodings, names, db.version)
    print(f"已加载 {len(names)} 人: {names}")

# 添加新人后重新同步
db.add_person("David", "david.jpg", david_encoding)
# 版本号已变化，重新同步
encodings, names = db.get_encodings_and_names()
recognizer.update_cache(encodings, names, db.version)
```

## 版本号与增量更新

版本号机制是实现高效缓存同步的关键：

```python
db = FaceDatabase()

v1 = db.version  # 初始版本

# 写操作后版本号自增
db.add_person("Alice", "alice.jpg", alice_encoding)
v2 = db.version
print(v2 > v1)  # True

db.add_person("Bob", "bob.jpg", bob_encoding)
v3 = db.version
print(v3 > v2)  # True

db.remove_person("Alice")
v4 = db.version
print(v4 > v3)  # True

# 查询操作不改变版本号
db.get_names()
print(db.version == v4)  # True
```

## 清空数据库

```python
db = FaceDatabase()

# 添加一些数据
db.add_person("Alice", "alice.jpg", alice_encoding)
db.add_person("Bob", "bob.jpg", bob_encoding)

print(db.get_names())  # ['Alice', 'Bob']

# 清空所有数据（包括文件）
db.clear()

print(db.get_names())  # []

# 重新开始注册
db.add_person("Charlie", "charlie.jpg", charlie_encoding)
print(db.version)  # 版本号从 1 重新开始
```

## 完整封装示例

将数据库操作封装为服务类：

```python
import cv2
from face_hub import FaceDetector, FaceDatabase, FaceRecognizer

class FaceRegistry:
    """人脸注册服务"""

    def __init__(self, db_path="face_db.json"):
        self.detector = FaceDetector(device="auto", det_size=640)
        self.db = FaceDatabase(db_path=db_path)
        self.recognizer = FaceRecognizer()
        self._sync()

    def _sync(self):
        """同步数据库到识别器"""
        encodings, names = self.db.get_encodings_and_names()
        if names:
            self.recognizer.update_cache(encodings, names, self.db.version)

    def register(self, name, photo_path):
        """注册新用户"""
        frame = cv2.imread(photo_path)
        if frame is None:
            return False, f"无法读取图片: {photo_path}"

        faces = self.detector.detect_with_embeddings(frame)
        if not faces:
            return False, "未检测到人脸"

        face = faces[0]
        if not face.has_embedding:
            return False, "特征提取失败"

        ok, msg = self.db.add_person(name, photo_path, face.embedding)
        if ok:
            self._sync()
        return ok, msg

    def unregister(self, name):
        """注销用户"""
        ok, msg = self.db.remove_person(name)
        if ok:
            self._sync()
        return ok, msg

    def list_users(self):
        """列出所有用户"""
        return self.db.get_names()

    def get_user_count(self):
        """获取用户数量"""
        return len(self.db.get_names())


# 使用示例
registry = FaceRegistry()

registry.register("Alice", "photos/alice.jpg")
registry.register("Bob", "photos/bob.jpg")

print(f"已注册 {registry.get_user_count()} 人")
print(registry.list_users())  # ['Alice', 'Bob']

registry.unregister("Bob")
print(registry.list_users())  # ['Alice']
```

## 数据备份与恢复

```python
import shutil
from datetime import datetime

def backup_database(db, backup_dir="backups"):
    """备份数据库文件"""
    import os
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_backup = f"{backup_dir}/face_db_{timestamp}.json"
    pkl_backup = f"{backup_dir}/encodings_{timestamp}.pkl"

    shutil.copy2(db.db_path, json_backup)
    shutil.copy2(db.encoding_path, pkl_backup)

    print(f"数据库已备份到 {backup_dir}/")
    return json_backup, pkl_backup

def restore_database(db, json_path, pkl_path):
    """从备份恢复数据库"""
    import os
    db.save()  # 先保存当前状态

    # 复制备份文件
    shutil.copy2(json_path, db.db_path)
    shutil.copy2(pkl_path, db.encoding_path)

    # 重新加载
    db.load()
    print(f"数据库已从备份恢复，当前有 {len(db.get_names())} 人")

# 使用
backup_database(db)
```

## 注意事项

- 数据库文件默认保存在当前工作目录。指定 `db_path` 参数可以控制位置。
- `encoding` 必须是 512 维 `float32` 的 numpy 数组（ArcFace 输出）。
- 删除人员时，对应的参考照片文件也会被删除（如果存在）。
- `clear()` 会删除持久化文件和所有照片，操作不可逆。
- 版本号从 1 开始，每次写操作自增 1。`clear()` 后版本号重置为 1。
- 数据库在构造时自动调用 `load()`，如果文件不存在则初始化为空。