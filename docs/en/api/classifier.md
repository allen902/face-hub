# PhotoClassifier

Classify a collection of photos by the faces they contain — group photos
by person. Two modes:

- **Discovery mode** (no gallery): faces are clustered by embedding cosine
  similarity into anonymous groups (`person_001`, `person_002`, ...).
- **Gallery mode** (with a `FaceRecognizer`): faces matching a registered
  person are filed under that person's name; faces that don't match anyone
  fall through to clustering, so strangers are still grouped together.

!!! note "What the classifier does NOT do"
    `PhotoClassifier` only computes groups **in memory** — it never moves,
    copies, or renames files. A photo containing several people appears in
    **several groups** (see [Exporting to folders](#exporting-to-folders)
    for turning the result into per-person directories).

---

## Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `detector` | `DetectorProtocol` | *(required)* | `FaceDetector` or any custom detector |
| `recognizer` | `FaceRecognizer \| None` | `None` | Recognizer whose cache holds the registered gallery; omit for discovery mode |
| `cluster_threshold` | `float` | `0.45` | Cosine-similarity threshold for clustering; 0.40 strict, 0.45 recommended, 0.50 loose |
| `skip_low_quality` | `bool` | `True` | Ignore detections whose `quality_pass` is `False` |

## Methods

### classify_photos(images, photo_ids=None, progress_callback=None)

Classify a batch of photos by face.

**Parameters:**
- `images` (`Sequence[str | Path | np.ndarray]`): Image file paths or BGR numpy arrays.
- `photo_ids` (`Sequence[str] | None`): Optional explicit id per image. Defaults to the path string for files, or `image_0001`-style ids for arrays. Must be unique and match `len(images)`.
- `progress_callback` (`Callable[[int, int, str], None] | None`): Optional `fn(done, total, photo_id)` called after each photo. Exceptions inside the callback are logged and never abort the batch.

**Returns:**
- `PhotoClassificationResult`: groups per person, per-face records, and bookkeeping lists. See [Types](types.md#photoface).

**Raises:**
- `ValueError`: If `photo_ids` length differs from `images`, or contains duplicates.
- `FaceHubError`: If detection fails on a photo.

---

## classify_photos() One-Shot Helper

```python
from face_hub import classify_photos

result = classify_photos(["party1.jpg", "party2.jpg", "party3.jpg"])
```

Builds a default `FaceDetector` when `detector` is not given (extra keyword
arguments are forwarded to its constructor) and classifies in one call:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `images` | `Sequence` | *(required)* | Paths or BGR arrays |
| `detector` | `DetectorProtocol \| None` | `None` | Existing detector; built automatically if omitted |
| `recognizer` | `FaceRecognizer \| None` | `None` | Gallery for gallery mode |
| `cluster_threshold` | `float` | `0.45` | Clustering threshold |
| `photo_ids` | `Sequence[str] \| None` | `None` | Explicit photo ids |
| `progress_callback` | `Callable \| None` | `None` | Progress hook |
| `**detector_kwargs` | | | Forwarded to `FaceDetector(...)` when auto-built |

---

## Discovery Mode (No Gallery)

```python
from face_hub import classify_photos

result = classify_photos(["a.jpg", "b.jpg", "c.jpg", "d.jpg"])

for label, group in result.groups.items():
    print(f"{label}: {group.photo_ids} ({group.face_count} faces)")
# person_001: ['a.jpg', 'c.jpg'] (2 faces)
# person_002: ['b.jpg'] (1 faces)

print(result.no_face_photos)      # ['d.jpg'] — no usable face found
print(result.unreadable_photos)   # [] — files that failed to decode
print(result.summary())           # {'person_001': 2, 'person_002': 1}
```

## Gallery Mode (Registered People)

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
# alice → ['img1.jpg', 'img7.jpg']      ← matched the gallery
# person_001 → ['img2.jpg', 'img5.jpg'] ← stranger, clustered
```

## Photos with Multiple People

A photo containing several people appears in **each** of their groups:

```python
result = classify_photos(["group_photo.jpg"])

print(result.labels_of("group_photo.jpg"))
# ['person_001', 'person_002']

print(result.groups["person_001"].photo_ids)  # ['group_photo.jpg']
print(result.groups["person_002"].photo_ids)  # ['group_photo.jpg']
```

Use `result.faces` for per-face details (bounding box, detection confidence,
label, similarity):

```python
for face in result.faces:
    print(face.photo_id, face.label, face.bbox.to_tuple(), f"{face.similarity:.2f}")
```

## Exporting to Folders

The classifier returns groups in memory only. To materialize per-person
folders, copy the files yourself — note that a multi-person photo is copied
into **every** folder it belongs to:

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

# Photos with no usable face
no_face = out_root / "_no_face"
no_face.mkdir(exist_ok=True)
for photo_id in result.no_face_photos:
    shutil.copy2(photo_id, no_face / Path(photo_id).name)
```

## Progress Reporting

```python
def on_progress(done, total, photo_id):
    print(f"\r{done}/{total}: {photo_id}", end="", flush=True)

result = classify_photos(photos, progress_callback=on_progress)
```

## Threshold Tuning

| Symptom | Adjustment | Effect |
|---------|-----------|--------|
| Same person split into several `person_xxx` groups | Lower `cluster_threshold` (e.g. 0.40) | Merges more aggressively |
| Different people merged into one group | Raise `cluster_threshold` (e.g. 0.50) | Splits more conservatively |

```python
classifier = PhotoClassifier(detector, cluster_threshold=0.40)  # looser grouping
```

Side profiles and extreme lighting produce embeddings further apart than
frontal shots; 0.45 is a good default, tune with a small sample of your own
photos before running a large batch.

## Custom Detector

Any `DetectorProtocol` implementation works, exactly like the pipeline:

```python
classifier = PhotoClassifier(MyYoloDetector(), cluster_threshold=0.45)
result = classifier.classify_photos(photos)
```

## Notes

- Embeddings are L2-normalized before clustering, so the dot product equals
  cosine similarity.
- Clustering is greedy and deterministic in input order: each face joins the
  cluster with the most similar centroid, or starts a new one.
- In gallery mode, `FaceRecognizer.tolerance` decides gallery matches while
  `cluster_threshold` decides how strangers are grouped — the two thresholds
  are independent.
- Faces failing the quality filter are skipped by default
  (`skip_low_quality=True`); photos left with no usable face land in
  `no_face_photos`.
- `result.total_photos` always equals `len(images)`; `result.total_faces`
  counts every usable face across all photos.
