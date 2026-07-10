# FaceDatabase

Face database for persisting person records and their 512-dim face embeddings.

The database stores:
- Person metadata (`name`, `image_path`) in a JSON file.
- Face embeddings in a pickle file.
- A monotonically increasing `version` that downstream components use to
  invalidate caches efficiently.

---

## Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `db_path` | `str` | `"face_db.json"` | JSON file path for person records |
| `encoding_path` | `str` | `"encodings.pkl"` | Pickle file path for encodings |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `version` | `int` | Database version, incremented on every write |

## Methods

### add_person(name, image_path, encoding)

Add a person.

**Parameters:**
- `name` (`str`): Unique person name
- `image_path` (`str`): Path to the reference photo
- `encoding` (`np.ndarray`): 512-dim face embedding

**Returns:**
- `(bool, str)`: `(success, message)`

### remove_person(name)

Remove a single person and delete their photo file if it exists.

### remove_persons(names)

Remove multiple persons in one call.

**Returns:**
- `(List[str], List[str])`: `(removed, not_found)`

### get_names()

Return all registered names.

### get_encodings_and_names()

Return `(encodings, names)`.

### save()

Persist to disk explicitly. Normally called automatically by mutating methods.

### load()

Load from disk. Called automatically on construction.

### clear()

Clear the database and delete persisted files and photos.

---

## Basic CRUD Example

```python
import numpy as np
from face_hub import FaceDatabase

db = FaceDatabase(db_path="face_db.json", encoding_path="encodings.pkl")

# In practice the encoding comes from FaceDetector.detect_with_embeddings()
encoding = np.random.randn(512).astype(np.float32)

ok, msg = db.add_person("Alice", "photos/alice.jpg", encoding)
print(ok, msg)

print(db.get_names())  # ['Alice']

ok, msg = db.remove_person("Alice")
print(ok, msg)
```

## Register from a Real Photo

```python
import cv2
import numpy as np
from face_hub import FaceDetector, FaceDatabase

detector = FaceDetector(device="cpu")
db = FaceDatabase()

image_path = "alice.jpg"
frame = cv2.imread(image_path)
faces = detector.detect_with_embeddings(frame)

if not faces:
    raise ValueError("No face found")

# Use the highest-confidence detection
face = faces[0]
if not face.has_embedding:
    raise ValueError("Failed to extract embedding")

ok, msg = db.add_person("Alice", image_path, face.embedding)
print(msg)
```

## Batch Delete

```python
removed, not_found = db.remove_persons(["Alice", "Bob", "Charlie"])
print("Removed:", removed)
print("Not found:", not_found)
```

## Sync Cache with Recognizer

```python
from face_hub import FaceRecognizer

recognizer = FaceRecognizer()
encodings, names = db.get_encodings_and_names()
recognizer.update_cache(encodings, names, db.version)
```

## Version Bumping

```python
v1 = db.version
db.add_person("Bob", "bob.jpg", encoding)
v2 = db.version
print(v2 > v1)  # True
```

## Clear Everything

```python
db.clear()
print(db.get_names())  # []
```
