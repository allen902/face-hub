# Installation

## Base Install

```bash
pip install face-hub
```

## GPU Acceleration (Optional)

### Windows DirectML

```bash
pip uninstall -y onnxruntime
pip install face-hub[gpu-win]
```

### Linux NVIDIA CUDA

```bash
pip uninstall -y onnxruntime
pip install face-hub[gpu-linux]
```

## Development Dependencies

```bash
pip install face-hub[dev]
```

## Platform Notes

- **Windows**: Uses the DirectShow backend for cameras.
- **macOS**: CPU inference in v1.0; camera backend is AVFoundation.
- **Linux**: Camera backend is V4L2.

## Model License

The FaceHub source code is MIT-licensed. The `buffalo_l` model downloaded
automatically by insightface is subject to insightface's own non-commercial
model license; commercial use requires separate authorization.
