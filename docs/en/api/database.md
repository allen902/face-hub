# FaceDatabase

Face database for persisting person records and encodings.

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

**Returns:**
- `(bool, str)`: `(success, message)`

### remove_person(name)

Remove a single person.

### remove_persons(names)

Remove multiple persons.

**Returns:**
- `(List[str], List[str])`: `(removed, not_found)`

### get_names()

Return all registered names.

### get_encodings_and_names()

Return `(encodings, names)`.

### save()

Persist to disk.

### load()

Load from disk.

### clear()

Clear the database and delete persisted files.
