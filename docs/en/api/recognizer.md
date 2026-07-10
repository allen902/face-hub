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
| `tolerance` | `float` | Current threshold value |

## Methods

### update_cache(known_encodings, known_names, db_version=0)

Update the encoding cache. Rebuilds only when the database version changes.

**Parameters:**
- `known_encodings` (`List[np.ndarray]`): Registered encodings (512-dim L2-normalized).
- `known_names` (`List[str]`): Corresponding person names.
- `db_version` (`int`): Database version number. If unchanged from last call, no rebuild occurs.

**Returns:**
- `bool`: Whether the cache was rebuilt.

### recognize(unknown_encoding, known_encodings=None, known_names=None)

Recognize a single face embedding against the cached or explicit gallery.

**Parameters:**
- `unknown_encoding` (`np.ndarray`): 512-dim query encoding (L2-normalized).
- `known_encodings` (`List[np.ndarray] | None`): Optional explicit gallery. Overrides cache if provided.
- `known_names` (`List[str] | None`): Optional explicit names.

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
# Note: real embeddings from insightface are already L2-normalized

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

# Load all registered persons into recognizer
encodings, names = db.get_encodings_and_names()
rebuilt = recognizer.update_cache(encodings, names, db.version)

if rebuilt:
    print(f"Cache loaded: {len(names)} person(s)")

# Now recognize without passing the gallery every time
name, conf = recognizer.recognize(query_embedding)
print(f"Best match: {name} ({conf:.2f})")
```

## Explicit Gallery (No Cache)

Pass the gallery directly to `recognize()` without using the cache:

```python
known_encodings = [alice_emb, bob_emb, charlie_emb]
known_names = ["Alice", "Bob", "Charlie"]

name, conf = recognizer.recognize(
    query_embedding,
    known_encodings=known_encodings,
    known_names=known_names,
)
print(f"Matched: {name} with {conf:.2f} similarity")
```

## Cache Versioning

The version-based cache avoids expensive matrix rebuilds every frame:

```python
recognizer = FaceRecognizer()

# First call builds the cache (expensive)
rebuilt = recognizer.update_cache([alice_emb], ["Alice"], db_version=1)
print(rebuilt)  # True — cache was built

# Same version: no-op (fast)
rebuilt = recognizer.update_cache([alice_emb], ["Alice"], db_version=1)
print(rebuilt)  # False — cache is current, skipped

# New version: rebuild (Alice + Bob)
rebuilt = recognizer.update_cache(
    [alice_emb, bob_emb], ["Alice", "Bob"], db_version=2
)
print(rebuilt)  # True — cache rebuilt with new data
```

## Threshold Tuning

| Scenario | Suggested `tolerance` | Effect |
|----------|----------------------|--------|
| High security / low false-accept | 0.35 ~ 0.40 | Stricter, may reject real users |
| Recommended default | 0.45 | Balanced precision and recall |
| Low false-rejection / high recall | 0.50 ~ 0.55 | More permissive |

```python
# High-security entrance
strict_recognizer = FaceRecognizer(tolerance=0.40)

# General attendance / convenience
loose_recognizer = FaceRecognizer(tolerance=0.50)
```

## Changing Tolerance at Runtime

```python
recognizer = FaceRecognizer(tolerance=0.45)
name1, conf1 = recognizer.recognize(embedding)

# Make it stricter
recognizer.tolerance = 0.40
name2, conf2 = recognizer.recognize(embedding)
# Same embedding may now return UNKNOWN_SENTINEL
```

## Full Recognition Loop with All Components

```python
import cv2
from face_hub import (
    FaceDetector, FaceRecognizer, FaceTracker,
    FaceDatabase, UNKNOWN_SENTINEL,
)

detector = FaceDetector(device="auto", det_size=640)
recognizer = FaceRecognizer(tolerance=0.45)
tracker = FaceTracker(smooth_frames=5)
db = FaceDatabase()

# Load gallery from database
encodings, names = db.get_encodings_and_names()
recognizer.update_cache(encodings, names, db.version)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect_with_embeddings(frame)
    tracked = tracker.update(detections, recognizer=recognizer)

    for face in tracked:
        if face.is_confirmed and face.name != UNKNOWN_SENTINEL:
            color = (0, 255, 0)
            label = f"{face.name} ({face.confidence:.0%})"
        else:
            color = (0, 0, 255)
            label = "Unknown"

        x1, y1, x2, y2 = face.bbox.to_tuple()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

## Incremental Updates (Adding People While Running)

```python
# Register a new person
db.add_person("David", "david.jpg", david_embedding)

# Reload cache with new version
encodings, names = db.get_encodings_and_names()
recognizer.update_cache(encodings, names, db.version)

# Reset tracker to clear old identity history
tracker.reset()

# David is now recognized immediately
```

## Inspect Cached Gallery

```python
recognizer = FaceRecognizer()
encodings, names = db.get_encodings_and_names()
recognizer.update_cache(encodings, names, db.version)

print(f"Gallery size: {len(recognizer.cached_names)}")
for name in recognizer.cached_names:
    print(f"  - {name}")
```

## Confidence Interpretation

The returned `confidence` is the raw cosine similarity between the query
embedding and the best-matching gallery embedding:

```python
name, confidence = recognizer.recognize(query_embedding)

if confidence >= 0.80:
    level = "Very high — near-identical"
elif confidence >= 0.60:
    level = "High — same person"
elif confidence >= 0.50:
    level = "Moderate — likely same person"
elif confidence >= 0.45:
    level = "Marginal — verify manually"
else:
    level = "Low — unknown or different person"

print(f"Match confidence: {confidence:.2f} — {level}")
```

## Performance Considerations

```python
# Cache rebuild complexity
# Time: O(n) per rebuild (n = registered persons)
# Memory: O(n × 512 × 4) bytes (each embedding = 2048 bytes)

# Per-frame recognition complexity
# Time: O(n) — dot product with all gallery embeddings
#       (computed as one matrix-vector multiply, very fast on CPU)

# With 10000 registered persons:
# — Cache: ~20 MB RAM
# — Per-frame: ~0.1 ms on modern CPU
```

## Notes

- Embeddings are expected to be L2-normalized 512-dim `float32` vectors (ArcFace
  output from insightface). If your embeddings are not normalized, normalize
  them first: `emb = emb / np.linalg.norm(emb)`.
- Cosine similarity is computed as the dot product because the gallery is
  already normalized, making it equivalent to cosine similarity.
- `confidence` is the raw cosine similarity in `[0.0, 1.0]`, not a
  probability. Values ≥ 0.50 generally indicate a match.
- The cache is stored as a contiguous `(n, 512)` `float32` NumPy array for
  efficient matrix-vector multiplication.
- When an explicit gallery is passed to `recognize()`, it is used directly
  without touching the cache.