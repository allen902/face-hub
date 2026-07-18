# PhotoClassifier 照片分类器

按人脸对照片集进行分类 —— 把照片按其中的人物分组。支持两种模式：

- **自动聚类模式**（无人脸库）：按特征向量余弦相似度把人脸聚类为匿名分组
  （`person_001`、`person_002`……）。
- **人脸库模式**（传入 `FaceRecognizer`）：匹配到已注册人员的人脸直接归入其
  姓名分组；匹配不上的人脸会落入聚类流程，陌生人依然能被自动分组。

!!! note "分类器不会做什么"
    `PhotoClassifier` 只**在内存中**计算分组结果 —— 不会移动、复制或重命名
    任何文件。包含多人的照片会**同时出现在多个分组**中（如需把结果落地为按人
    分类的文件夹，见[导出到文件夹](#_6)）。

---

## 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `detector` | `DetectorProtocol` | *（必填）* | `FaceDetector` 或任意自定义检测器 |
| `recognizer` | `FaceRecognizer \| None` | `None` | 缓存中持有注册人脸库的识别器；不传则为自动聚类模式 |
| `cluster_threshold` | `float` | `0.45` | 聚类余弦相似度阈值；0.40 严格，0.45 推荐，0.50 宽松 |
| `skip_low_quality` | `bool` | `True` | 忽略 `quality_pass` 为 `False` 的检测结果 |

## 方法

### classify_photos(images, photo_ids=None, progress_callback=None)

按人脸对一批照片进行分类。

**参数：**
- `images`（`Sequence[str | Path | np.ndarray]`）：图片文件路径或 BGR numpy 数组。
- `photo_ids`（`Sequence[str] | None`）：可选，为每张图片指定显式 id。默认文件用路径字符串、数组用 `image_0001` 风格 id。必须唯一且与 `len(images)` 一致。
- `progress_callback`（`Callable[[int, int, str], None] | None`）：可选，每处理完一张照片调用 `fn(done, total, photo_id)`。回调内抛出的异常只会记录日志，不会中断批处理。

**返回：**
- `PhotoClassificationResult`：按人分组、逐人脸明细及簿记列表，见[类型文档](types.md#photoface)。

**抛出：**
- `ValueError`：`photo_ids` 长度与 `images` 不一致，或存在重复。
- `FaceHubError`：某张照片检测失败。

---

## classify_photos() 一行式便捷函数

```python
from face_hub import classify_photos

result = classify_photos(["party1.jpg", "party2.jpg", "party3.jpg"])
```

未提供 `detector` 时自动构建默认 `FaceDetector`（多余的关键字参数会转发给
其构造函数），一次调用完成分类：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `images` | `Sequence` | *（必填）* | 路径或 BGR 数组 |
| `detector` | `DetectorProtocol \| None` | `None` | 已有检测器；省略时自动构建 |
| `recognizer` | `FaceRecognizer \| None` | `None` | 人脸库模式的识别器 |
| `cluster_threshold` | `float` | `0.45` | 聚类阈值 |
| `photo_ids` | `Sequence[str] \| None` | `None` | 显式照片 id |
| `progress_callback` | `Callable \| None` | `None` | 进度回调 |
| `**detector_kwargs` | | | 自动构建时转发给 `FaceDetector(...)` |

---

## 自动聚类模式（无人脸库）

```python
from face_hub import classify_photos

result = classify_photos(["a.jpg", "b.jpg", "c.jpg", "d.jpg"])

for label, group in result.groups.items():
    print(f"{label}: {group.photo_ids} ({group.face_count} 张人脸)")
# person_001: ['a.jpg', 'c.jpg'] (2 张人脸)
# person_002: ['b.jpg'] (1 张人脸)

print(result.no_face_photos)      # ['d.jpg'] —— 未检测到可用人脸
print(result.unreadable_photos)   # [] —— 解码失败的文件
print(result.summary())           # {'person_001': 2, 'person_002': 1}
```

## 人脸库模式（已注册人员）

```python
from face_hub import FaceDetector, FaceRecognizer, FaceDatabase, PhotoClassifier

detector = FaceDetector(device="auto")
recognizer = FaceRecognizer(tolerance=0.45)

db = FaceDatabase(db_path="face_db.json")
encodings, names = db.get_encodings_and_names()
recognizer.update_cache(encodings, names, db.version)

classifier = PhotoClassifier(detector, recognizer=recognizer)
result = classifier.classify_photos(photos)

for label, group in result.groups.items():
    print(label, "→", group.photo_ids)
# alice → ['img1.jpg', 'img7.jpg']      ← 匹配到人脸库
# person_001 → ['img2.jpg', 'img5.jpg'] ← 陌生人，自动聚类
```

## 包含多人的照片

包含多人的照片会同时出现在**每个人**的分组中：

```python
result = classify_photos(["合影.jpg"])

print(result.labels_of("合影.jpg"))
# ['person_001', 'person_002']

print(result.groups["person_001"].photo_ids)  # ['合影.jpg']
print(result.groups["person_002"].photo_ids)  # ['合影.jpg']
```

逐人脸明细（边界框、检测置信度、分组标签、相似度）见 `result.faces`：

```python
for face in result.faces:
    print(face.photo_id, face.label, face.bbox.to_tuple(), f"{face.similarity:.2f}")
```

## 导出到文件夹

分类器只在内存中返回分组结果。如需落地为按人分类的文件夹，请自行复制
文件 —— 注意多人合影会被复制到**每一个**相关人物的文件夹：

```python
import shutil
from pathlib import Path
from face_hub import classify_photos

photos = list(Path("incoming").glob("*.jpg"))
result = classify_photos(photos)

out_root = Path("sorted")
for label, group in result.groups.items():
    folder = out_root / label
    folder.mkdir(parents=True, exist_ok=True)
    for photo_id in group.photo_ids:
        shutil.copy2(photo_id, folder / Path(photo_id).name)

# 未检测到可用人脸的照片
no_face = out_root / "_no_face"
no_face.mkdir(exist_ok=True)
for photo_id in result.no_face_photos:
    shutil.copy2(photo_id, no_face / Path(photo_id).name)
```

## 进度回调

```python
def on_progress(done, total, photo_id):
    print(f"\r{done}/{total}: {photo_id}", end="", flush=True)

result = classify_photos(photos, progress_callback=on_progress)
```

## 阈值调优

| 现象 | 调整 | 效果 |
|------|------|------|
| 同一个人被拆成多个 `person_xxx` 组 | 调低 `cluster_threshold`（如 0.40） | 合并更激进 |
| 不同的人被合并到同一组 | 调高 `cluster_threshold`（如 0.50） | 拆分更保守 |

```python
classifier = PhotoClassifier(detector, cluster_threshold=0.40)  # 更宽松的分组
```

侧脸和极端光照下的特征向量与正脸差异较大；0.45 是合适的默认值，大批量
处理前建议先用自己的一小批照片试跑调参。

## 自定义检测器

与流水线一样，任何 `DetectorProtocol` 实现都可以直接接入：

```python
classifier = PhotoClassifier(MyYoloDetector(), cluster_threshold=0.45)
result = classifier.classify_photos(photos)
```

## 备注

- 聚类前特征向量会做 L2 归一化，因此点积即余弦相似度。
- 聚类为贪心算法，在输入顺序下是确定性的：每张人脸加入质心最相似的
  分组，否则新建分组。
- 人脸库模式下，`FaceRecognizer.tolerance` 决定人脸库匹配，
  `cluster_threshold` 决定陌生人如何分组 —— 两个阈值相互独立。
- 未通过质量过滤的人脸默认被跳过（`skip_low_quality=True`）；过滤后没有
  可用人脸的照片会进入 `no_face_photos`。
- `result.total_photos` 恒等于 `len(images)`；`result.total_faces` 统计
  所有照片中可用人脸的总数。
