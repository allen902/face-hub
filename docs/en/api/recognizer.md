# FaceRecognizer

1:N face recognizer based on cosine similarity.

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
- `known_encodings` (`List[np.ndarray] | None`): Optional explicit gallery
- `known_names` (`List[str] | None`): Optional explicit names

**Returns:**
- `(str, float)`: `(name, confidence)`. Returns `(UNKNOWN_SENTINEL, 0.0)` on no match.

## Tuning the Threshold

- High security: 0.35 ~ 0.40
- Recommended default: 0.45
- Lower false-rejection: 0.50
