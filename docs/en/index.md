<div class="fh-hero">
<canvas id="fh-particles"></canvas>
<div class="fh-hero-content">
<span class="fh-hero-badge">REAL-TIME FACE RECOGNITION · v1.3.0</span>
<h1 class="fh-hero-title">Face<span class="fh-glow">Hub</span></h1>
<p class="fh-hero-typing"><span id="fh-typer" data-phrases='["Detection · Embedding · Tracking · Matching", "Group your entire photo library by face", "Plug in any detector via DetectorProtocol", "CUDA / DirectML / CPU — auto-selected"]'></span><span class="fh-cursor"></span></p>
<div class="fh-hero-actions">
<a class="fh-btn fh-btn--primary" href="installation/">Get Started</a>
<a class="fh-btn fh-btn--ghost" href="api/pipeline/">API Reference</a>
</div>
<div class="fh-stats">
<div class="fh-stat"><span class="fh-stat-num" data-target="512">0</span><span class="fh-stat-label">dim embeddings</span></div>
<div class="fh-stat"><span class="fh-stat-num" data-target="30" data-suffix=" FPS">0</span><span class="fh-stat-label">real-time pipeline</span></div>
<div class="fh-stat"><span class="fh-stat-num" data-target="3" data-suffix=" OS">0</span><span class="fh-stat-label">cross-platform</span></div>
<div class="fh-stat"><span class="fh-stat-num" data-target="100" data-suffix="%">0</span><span class="fh-stat-label">open source</span></div>
</div>
</div>
</div>

FaceHub is a real-time face recognition Python library. It provides a clean,
GUI-free package for detection, embedding extraction, recognition, tracking,
camera capture — and grouping whole photo collections by face.

## Features { .fh-section }

<div class="grid">
<div class="fh-card">
<span class="fh-card-icon">◎</span>
<h3>Detection</h3>
<p>insightface RetinaFace with GPU auto-detection (CUDA / DirectML) and CPU fallback.</p>
</div>
<div class="fh-card">
<span class="fh-card-icon">⬡</span>
<h3>Embedding</h3>
<p>ArcFace 512-dim L2-normalized features, ready for cosine-similarity matching.</p>
</div>
<div class="fh-card">
<span class="fh-card-icon">⇄</span>
<h3>Recognition</h3>
<p>1:N matching with a versioned encoding cache — rebuilt only when the gallery changes.</p>
</div>
<div class="fh-card">
<span class="fh-card-icon">⋈</span>
<h3>Tracking</h3>
<p>IoU multi-face tracker with majority-vote identity smoothing.</p>
</div>
<div class="fh-card">
<span class="fh-card-icon">▦</span>
<h3>Photo Classification</h3>
<p>Group photo collections by face — gallery matching or automatic clustering, with per-person folder export.</p>
</div>
<div class="fh-card">
<span class="fh-card-icon">⌘</span>
<h3>Extensible</h3>
<p><code>DetectorProtocol</code> lets you plug in your own detector (YOLO, MediaPipe, …).</p>
</div>
</div>

## 60-Second Quick Start { .fh-section }

<div class="fh-terminal" markdown>
<div class="fh-terminal-bar"><span class="fh-terminal-dot"></span><span class="fh-terminal-dot"></span><span class="fh-terminal-dot"></span><span class="fh-terminal-title">zsh — pip</span></div>

```bash
pip install face-hub
```

</div>

```python
from face_hub import classify_photos, export_to_folders

# Group a folder of photos by the people in them
result = classify_photos(["party1.jpg", "party2.jpg", "party3.jpg"])

for label, group in result.groups.items():
    print(label, "→", group.photo_ids)
# person_001 → ['party1.jpg', 'party3.jpg']
# person_002 → ['party2.jpg']

# Export into per-person folders
export_to_folders(result, "sorted/")
```

## Platform Support { .fh-section }

| Platform | Inference Backend | Notes |
|----------|-------------------|-------|
| Windows | DirectML / CUDA / CPU | Auto-detect, prefers GPU |
| Linux | CUDA / CPU | Auto-detect, prefers GPU |
| macOS | CPU | CPU only; CoreML planned |

## Next Steps { .fh-section }

[Installation](installation.md){ .fh-link-card }
[Quick Start](quickstart.md){ .fh-link-card }
[Photo Classifier](api/classifier.md){ .fh-link-card }
[API Reference](api/pipeline.md){ .fh-link-card }
