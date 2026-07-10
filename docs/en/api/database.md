# FaceDatabase

SQLite-backed face registry that stores person names, photo paths, and
512-dim embeddings. It provides versioned access for cache-aware recognizer
synchronization.

Key features:
- Add, update, remove persons
- List all registered persons
- Export encodings + names for recognizer cache
- Automatic database version tracking (integer, incremented on every mutation)

---

## Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `db_path` | `str` | `"registry.db"` | Path to the SQLite database file |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `version` | `int` | Current database version; incremented on every add/update/remove |

## Methods

### init_db()

Initialize the database tables. Called automatically on first use. Safe to
call multiple times — it uses `CREATE TABLE IF NOT EXISTS`.

### add_person(name, photo_path, encoding)

Register a new person.

**Parameters:**
- `name` (`str`): Person name (must be unique).
- `photo_path` (`str`): Path or URI to the person's photo.
- `encoding` (`np.ndarray`): 512-dim face embedding.

**Returns:**
- `(bool, str)`: `(success, message)`. `False` if the name already exists.

### update_person(name, **kwargs)

Update an existing person's record.

**Parameters:**
- `name` (`str`): Person to update.
- `**kwargs`: Keyword arguments to update. Supported: `photo_path` (`str`), `encoding` (`np.ndarray`), `new_name` (`str`).

**Returns:**
- `(bool, str)`: `(success, message)`. `False` if the person does not exist.

### remove_person(name)

Delete a person from the registry.

**Parameters:**
- `name` (`str`): Person to remove.

**Returns:**
- `(bool, str)`: `(success, message)`. `False` if the person does not exist.

### list_persons()

List all registered persons.

**Returns:**
- `List[Tuple[str, str]]`: List of `(name, photo_path)` tuples.

### count_persons()

Count registered persons.

**Returns:**
- `int`

### get_encodings_and_names()

Get all encodings and names for recognizer usage.

**Returns:**
- `(List[np.ndarray], List[str])`: `(encodings, names)`.

### clear()

Delete all records from the database.

---

## Basic CRUD

```python
import numpy as np
from face_hub import FaceDatabase

db = FaceDatabase("my_people.db")

# Add
ok, msg = db.add_person("Alice", "photos/alice.jpg", alice_embedding)
print(msg)  # "Person 'Alice' added successfully"

# List
for name, photo_path in db.list_persons():
    print(f"{name} ← {photo_path}")

# Update
ok, msg = db.update_person("Alice", photo_path="photos/alice_new.jpg")
db.update_person("Alice", encoding=new_encoding)

# Count
print(f"Total registered: {db.count_persons()}")

# Remove
ok, msg = db.remove_person("Alice")

# Clear all
db.clear()
```

## Rename a Person

```python
ok, msg = db.update_person("Alice", new_name="Alice Smith")
if ok:
    print("Renamed successfully")
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

## Duplicate Name Protection

```python
ok, msg = db.add_person("Alice", "alice.jpg", alice_emb)
print(ok, msg)  # True, "Person 'Alice' added successfully"

ok, msg = db.add_person("Alice", "alice2.jpg", alice_emb2)
print(ok, msg)  # False, "Person 'Alice' already exists"

# Must remove or rename first
db.remove_person("Alice")
ok, msg = db.add_person("Alice", "alice2.jpg", alice_emb2)
print(ok, msg)  # True
```

## Update Non-Existent Person

```python
ok, msg = db.update_person("Nobody", photo_path="nobody.jpg")
print(msg)  # "Person 'Nobody' does not exist"
```

## Working with Multiple Databases

```python
# Separate registries for different applications
work_db = FaceDatabase("work_employees.db")
family_db = FaceDatabase("family_members.db")

# Each has independent versioning
work_db.add_person("Boss", "boss.jpg", boss_emb)     # work_db.version = 1
family_db.add_person("Mom", "mom.jpg", mom_emb)       # family_db.version = 1
```

## Database Inspection

```python
from face_hub import FaceDatabase

db = FaceDatabase("registry.db")

print(f"Version:      {db.version}")
print(f"Total persons:{db.count_persons()}")

for name, photo in db.list_persons():
    print(f"  {name:20s} ← {photo}")

encodings, names = db.get_encodings_and_names()
print(f"Encodings:    {len(encodings)} vectors of shape {encodings[0].shape}")
```

## Error Handling

```python
# Check return value instead of catching exceptions
ok, msg = db.add_person("Alice", "alice.jpg", emb)
if not ok:
    # msg contains a human-readable reason
    if "already exists" in msg:
        print("Alice is already registered — use update_person() to modify")
    else:
        print(f"Unexpected error: {msg}")

# Remove is idempotent (succeeds if person exists, fails gracefully if not)
ok, msg = db.remove_person("Ghost")
if not ok:
    print(msg)  # "Person 'Ghost' does not exist"
```

## Complete Registration Workflow

```python
import cv2
import numpy as np
from face_hub import FaceDetector, FaceDatabase

def build_registry_from_folder(photo_dir: str, db_path: str = "registry.db"):
    """Register all photos in a folder. Each filename becomes the person name."""
    import os
    import glob

    detector = FaceDetector(device="auto", det_size=640, confidence=0.60)
    db = FaceDatabase(db_path)

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    photos = []
    for pat in patterns:
        photos.extend(glob.glob(os.path.join(photo_dir, pat)))

    registered = 0
    skipped = 0

    for photo_path in photos:
        name = os.path.splitext(os.path.basename(photo_path))[0]
        frame = cv2.imread(photo_path)
        if frame is None:
            print(f"Skipping {photo_path}: cannot read")
            skipped += 1
            continue

        faces = detector.detect_with_embeddings(frame)
        if not faces or not faces[0].has_embedding:
            print(f"Skipping {photo_path}: no face detected")
            skipped += 1
            continue

        face = faces[0]
        if not face.quality_pass:
            print(f"Skipping {photo_path}: low quality (blurry or too small)")
            skipped += 1
            continue

        ok, msg = db.add_person(name, photo_path, face.embedding)
        if ok:
            print(f"✅ {name}")
            registered += 1
        else:
            print(f"❌ {name}: {msg}")
            skipped += 1

    print(f"\nDone. Registered: {registered}, Skipped: {skipped}")
    print(f"Database version: {db.version}, Total: {db.count_persons()}")

# Usage
build_registry_from_folder("photos/employees")
```

## Notes

- The database file is created automatically on first use. No need to call `init_db()` manually.
- `version` is a monotonically increasing integer. It starts at 0 (empty database) and increments by 1 on every mutation.
- Embeddings are stored as binary blobs in SQLite. Each is 512 × 4 = 2048 bytes.
- `add_person()` checks for duplicate names and returns `(False, message)` rather than raising an exception.
- `update_person()` does nothing if no keyword arguments are provided.
- `clear()` resets the database but does not delete the file — the file remains with empty tables.
- The database is thread-safe at the SQLite level (WAL mode is used internally).