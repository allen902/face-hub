# FaceHub

FaceHub is a real-time face recognition Python library. It provides a clean,
GUI-free package for detection, embedding extraction, recognition, tracking,
and camera capture.

## Features

- High-accuracy face detection with insightface RetinaFace
- ArcFace 512-dim L2-normalized embeddings
- 1:N cosine-similarity matching with a versioned encoding cache
- IoU multi-object tracking + majority-vote identity smoothing
- Cross-platform camera capture thread
- `DetectorProtocol` for plugging in custom detectors

## Platform Support

| Platform | Inference Backend | Notes |
|----------|-------------------|-------|
| Windows | DirectML / CUDA / CPU | Auto-detect, prefers GPU |
| Linux | CUDA / CPU | Auto-detect, prefers GPU |
| macOS | CPU | CPU only in v1.0; CoreML planned for v1.1 |

## Next Steps

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [API Reference](api/pipeline.md)
