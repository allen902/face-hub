# FaceDatabase

人脸数据库，负责持久化人员记录与编码。

## 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `db_path` | `str` | `"face_db.json"` | 人员记录 JSON 文件路径 |
| `encoding_path` | `str` | `"encodings.pkl"` | 编码 pickle 文件路径 |

## 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `version` | `int` | 数据库版本号，每次写操作递增 |

## 方法

### add_person(name, image_path, encoding)

添加人员。

**返回:**
- `(bool, str)`: `(是否成功, 消息)`

### remove_person(name)

删除单个人员。

### remove_persons(names)

批量删除。

**返回:**
- `(List[str], List[str])`: `(已删除列表, 未找到列表)`

### get_names()

返回所有注册姓名。

### get_encodings_and_names()

返回 `(encodings, names)`。

### save()

持久化到磁盘。

### load()

从磁盘加载。

### clear()

清空数据库并删除文件。
