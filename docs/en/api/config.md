# Configuration

FaceHub uses a `FaceHubSettings` TypedDict for all configuration options,
paired with `DEFAULT_SETTINGS` for sensible defaults.

The library never reads any config file. Callers should:
1. Read their own config file (JSON / YAML / TOML etc.)
2. Merge with `DEFAULT_SETTINGS` as fallback
3. Pass final values to each component's constructor

---

## FaceHubSettings

Typed configuration dictionary. All keys are optional (`total=False`).

```python
from face_hub import FaceHubSettings

# Partial config — missing keys are filled from DEFAULT_SETTINGS
my_config: FaceHubSettings = {
    "device": "cuda",
    "confidence": 0.60,
}
```

### Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `device` | `"auto" \| "cuda" \| "cpu"` | `"auto"` | Inference device. `"auto"` detects the best GPU |
| `confidence` | `float` | `0.50` | Detection confidence threshold (0, 1] |
| `tolerance` | `float` | `0.45` | Cosine similarity tolerance: 0.40=strict, 0.45=recommended, 0.50=loose |
| `cam_width` | `int` | `640` | Camera capture width |
| `cam_height` | `int` | `360` | Camera capture height |
| `cam_fps` | `int` | `30` | Camera capture FPS |
| `proc_fps` | `int` | `30` | ML processing FPS cap (0=unlimited) |
| `det_size` | `int` | `640` | Detection model input size: 320=fast, 480=balanced, 640=accurate |
| `track_smooth` | `int` | `5` | Tracking smooth frames: 3=fast, 5=recommended, 8=stable |
| `min_face_size` | `int` | `80` | Minimum face size in pixels |
| `quality_filter` | `bool` | `True` | Enable blur-based quality filtering |

---

## DEFAULT_SETTINGS

Default configuration constant of type `FaceHubSettings`.

```python
from face_hub import DEFAULT_SETTINGS

print(DEFAULT_SETTINGS)
# {'device': 'auto', 'confidence': 0.50, 'tolerance': 0.45, ...}
```

> **Note:** Modifying `DEFAULT_SETTINGS` directly affects the global default. Use `get_default_settings()` for safe mutation.

---

## get_default_settings()

Returns a deep copy of `DEFAULT_SETTINGS` that is safe to mutate without
affecting the global default.

```python
from face_hub import get_default_settings

config = get_default_settings()
config["device"] = "cuda"
config["confidence"] = 0.60

# Pass to components
detector = FaceDetector(
    device=config["device"],
    confidence=config["confidence"],
)
```

---

## Typical Usage

### Load from a JSON config file

```python
import json
from face_hub import DEFAULT_SETTINGS, FaceDetector, FaceRecognizer

# Read user config
with open("config.json") as f:
    user_config = json.load(f)

# Merge: user config takes priority, missing keys use defaults
config = {**DEFAULT_SETTINGS, **user_config}

# Use
detector = FaceDetector(
    device=config["device"],
    confidence=config["confidence"],
    det_size=config["det_size"],
)
recognizer = FaceRecognizer(tolerance=config["tolerance"])
```

### Scenario-based config templates

```python
from face_hub import get_default_settings

# Live monitoring: speed priority
live_config = get_default_settings()
live_config.update({
    "det_size": 320,
    "confidence": 0.45,
    "track_smooth": 3,
    "quality_filter": False,
})

# Photo registration: accuracy priority
register_config = get_default_settings()
register_config.update({
    "det_size": 640,
    "confidence": 0.60,
    "min_face_size": 100,
    "quality_filter": True,
})
```

---

## Notes

- `DEFAULT_SETTINGS` is a module-level constant. Mutating it directly affects all subsequent uses. Always use `get_default_settings()` for a safe copy.
- `dict.update()` is a shallow merge. If `FaceHubSettings` gains nested structures in the future, callers must switch to a recursive deep merge.
- Configuration keys provide type hints and defaults only. Actual validation happens in each component's constructor.
