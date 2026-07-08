# FaceRecognizer

1:N 人脸识别器，基于余弦相似度。

## 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tolerance` | `float` | `0.45` | 余弦相似度阈值；0.40 严格，0.45 推荐，0.50 宽松 |

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `cached_names` | `List[str]` | 当前缓存中的姓名列表 |
| `tolerance` | `float` | 阈值 |

## 方法

### update_cache(known_encodings, known_names, db_version=0)

更新编码缓存。仅在数据库版本变化时重建。

**参数:**
- `known_encodings` (`List[np.ndarray]`): 注册编码列表
- `known_names` (`List[str]`): 对应姓名列表
- `db_version` (`int`): 数据库版本号

**返回:**
- `bool`: 是否实际重建了缓存

### recognize(unknown_encoding, known_encodings=None, known_names=None)

识别单个人脸编码。

**参数:**
- `unknown_encoding` (`np.ndarray`): 512 维查询编码
- `known_encodings` (`List[np.ndarray] | None`): 显式指定注册库（可选）
- `known_names` (`List[str] | None`): 显式指定姓名（可选）

**返回:**
- `(str, float)`: `(name, confidence)`，未匹配时返回 `(UNKNOWN_SENTINEL, 0.0)`

## 阈值调参

- 高安全场景：0.35 ~ 0.40
- 推荐默认值：0.45
- 低误拒场景：0.50
