# FaceRecognizer

1:N face recognizer based on cosine similarity. It compares a query face
embedding against a gallery of registered embeddings and returns the best match.

The recognizer maintains an internal cache. Call `update_cache()` whenever the
registered gallery changes; the cache is only rebuilt when the provided
`db_version` changes, which avoids expensive matrix recomputation every frame.

---

## Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `tolerance` | `float` | `0.45` | Cosine-similarity threshold; 0.40 strict, 0.45 recommended, 0.50 loose |

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `cached_names` | `List[str]` | Names currently in the cache |
| `tolerance` | `float` | Threshold |

## Methods

### update_cache(known_encodings, known_names, db_version=0)

Update the encoding cache. Rebuilds only when the database version changes.

**Parameters:**
- `known_encodings` (`List[np.ndarray]`): Registered encodings
- `known_names` (`List[str]`): Corresponding names
- `db_version` (`int`): Database version number

**Returns:**
- `bool`: Whether the cache was rebuilt

### recognize(unknown_encoding, known_encodings=None, known_names=None)

Recognize a single face embedding.

**Parameters:**
- `unknown_encoding` (`np.ndarray`): 512-dim query encoding
- `known_encodings` (`List[np.ndarray] \| None`): Optional explicit gallery
- `known_names` (`List[str] \| None`): Optional explicit names

**Returns:**
- `(str, float)`: `(name, confidence)`. Returns `(UNKNOWN_SENTINEL, 0.0)` on no match.

---

## Basic Recognition

```python
import numpy as np
from face_hub import FaceRecognizer, UNKNOWN_SENTINEL

recognizer = FaceRecognizer(tolerance=0.45)

# In practice, query comes from FaceDetector.detect_with_embeddings()
query = np.random.randn(512).astype(np.float32)

name, confidence = recognizer.recognize(query)
if name == UNKNOWN_SENTINEL:
    print("Not recognized")
else:
    print(f"{name} ({confidence:.2%})")
```

## Update Cache from Database

```python
from face_hub import FaceDatabase, FaceRecognizer

db = FaceDatabase()
recognizer = FaceRecognizer()

encodings, names = db.get_encodings_and_names()
recognizer.update_cache(encodings, names, db.version)

# Now recognize without passing the gallery every time
name, conf = recognizer.recognize(query_embedding)
```

## Explicit Gallery (No Cache)

```python
known_encodings = [alice_emb, bob_emb]
known_names = ["Alice", "Bob"]

name, conf = recognizer.recognize(
    query_embedding,
    known_encodings=known_encodings,
    known_names=known_names,
)
```

## Cache Versioning

```python
recognizer = FaceRecognizer()

# First call builds the cache
rebuilt = recognizer.update_cache([alice_emb], ["Alice"], db_version=1)
print(rebuilt)  # True

# Same version: no-op
rebuilt = recognizer.update_cache([alice_emb], ["Alice"], db_version=1)
print(rebuilt)  # False

# New version: rebuild
rebuilt = recognizer.update_cache([alice_emb, bob_emb], ["Alice", "Bob"], db_version=2)
print(rebuilt)  # True
```

## Threshold Tuning

| Scenario | Suggested `tolerance` | Effect |
|----------|----------------------|--------|
| High security / low false-accept | 0.35 ~ 0.40 | Stricter, may reject real users |
| Recommended default | 0.45 | Balanced |
| Low false-rejection / high recall | 0.50 | More permissive |

```python
# High-security entrance
strict_recognizer = FaceRecognizer(tolerance=0.40)

# General attendance / convenience
loose_recognizer = FaceRecognizer(tolerance=0.50)
```

## Inspect Cached Gallery

```python
print(recognizer.cached_names)
print(len(recognizer.cached_names))
```

## Notes

- Embeddings are expected to be L2-normalized 512-dim vectors (ArcFace output).
- Cosine similarity is computed as the dot product because the gallery is
  already normalized.
- `confidence` is the raw cosine similarity in `[0.0, 1.0]`.
