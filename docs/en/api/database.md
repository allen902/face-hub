# FaceDatabase

Face database that stores person names, photo paths, and 512-dim
embeddings. It provides versioned access for cache-aware recognizer
synchronization.

Key features:
- Add, remove persons
- List all registered persons
- Export encodings + names for recognizer cache
- Automatic database version tracking (integer, incremented on every mutation)
- Secure storage: `.npy` format (no pickle), atomic writes, path validation

---

## Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `db_path` | `str` | `"face_db.json"` | Path to the JSON database file |
| `encoding_path` | `str` | `"encodings.npy"` | Path to the numpy encoding file |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `version` | `int` | Current database version; incremented on every add/remove |

## Methods

### add_person(name, image_path, encoding)

Register a new person.

**Parameters:**
- `name` (`str`): Person name (must be unique, non-empty).
- `image_path` (`str`): Path to the person's photo.
- `encoding` (`np.ndarray`): 512-dim face embedding, shape `(512,)`.

**Returns:**
- `(bool, str)`: `(success, message)`. `False` if the name already exists.

**Raises:**
- `ValueError`: If `name` is empty or `encoding` shape is not `(512,)`.

### remove_person(name)

Delete a person from the registry.

**Parameters:**
- `name` (`str`): Person to remove.

**Returns:**
- `(bool, str)`: `(success, message)`. `False` if the person does not exist.

### remove_persons(names)

Delete multiple persons at once.

**Parameters:**
- `names` (`List[str]`): Person names to remove.

**Returns:**
- `(List[str], List[str])`: `(removed, not_found)`.

### get_names()

List all registered person names.

**Returns:**
- `List[str]`

### get_encodings_and_names()

Get all encodings and names for recognizer usage.

**Returns:**
- `(List[np.ndarray], List[str])`: `(encodings, names)`.

### clear()

Delete all records and persisted files.

---

## Basic CRUD

```python
import numpy as np
from face_hub import FaceDatabase

db = FaceDatabase("face_db.json")

# Add
ok, msg = db.add_person("Alice", "photos/alice.jpg", alice_embedding)
print(msg)  # "Added: Alice"

# List
print(db.get_names())  # ['Alice']

# Remove
ok, msg = db.remove_person("Alice")

# Clear all
db.clear()
```

## Register from Detection

```python
import cv2
from face_hub import FaceDetector, FaceDatabase

detector = FaceDetector(device="cpu")
db = FaceDatabase()

def register_person(name, photo_path):
    frame = cv2.imread(photo_path)
    faces = detector.detect_with_embeddings(frame)

    if not faces:
        return False, "No face detected"

    face = faces[0]
    if not face.has_embedding:
        return False, "Failed to extract embedding"

    return db.add_person(name, photo_path, face.embedding)

ok, msg = register_person("Bob", "bob.jpg")
print(msg)
```

## Versioned Cache Sync

```python
from face_hub import FaceRecognizer

recognizer = FaceRecognizer()
encodings, names = db.get_encodings_and_names()

# The recognizer only rebuilds its internal matrix when version changes
rebuilt = recognizer.update_cache(encodings, names, db.version)

if rebuilt:
    print(f"Cache rebuilt — {len(names)} person(s) loaded")

# Subsequent calls with same version are no-ops
rebuilt = recognizer.update_cache(encodings, names, db.version)
assert not rebuilt  # no-op
```

## Input Validation

```python
# Encoding must be shape (512,)
try:
    db.add_person("Bad", "photo.jpg", np.zeros(256))
except ValueError as e:
    print(e)  # "encoding must be a numpy array of shape (512,), got (256,)"

# Name must be non-empty
try:
    db.add_person("", "photo.jpg", np.zeros(512))
except ValueError as e:
    print(e)  # "name must be a non-empty string"
```

## Notes

- The database file is created automatically on first use.
- `version` is a monotonically increasing integer. It starts at 0 (empty database) and increments by 1 on every mutation.
- Encodings are stored in `.npy` format (numpy native), **not** pickle, for security.
- Legacy `.pkl` files are automatically migrated to `.npy` on first load.
- `add_person()` checks for duplicate names and returns `(False, message)` rather than raising an exception.
- `clear()` deletes both the JSON and encoding files.
- Database files are written atomically (temp-file + rename) to prevent corruption.
- File permissions are set to owner-only (`0o600`) on platforms that support it.