# FaceVision v1.0 — 打包发布计划

> **目标：** 定义 `facevision` 的 PyPI 打包配置、CI/CD 流水线、README 内容和发版流程。
> **关联：** 代码结构见 [REFACTOR_PLAN.md](REFACTOR_PLAN.md)，开发者指南见 [DEVELOPER.md](DEVELOPER.md)

---

## 一、打包配置

### 1.1 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=75.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "facevision"
version = "1.0.0"
description = "Real-time face recognition library — RetinaFace detection + ArcFace recognition + IoU tracking"
readme = "README.md"
license = {text = "MIT"}
authors = [{name = "AllenDeng"}]
keywords = ["face-recognition", "insightface", "retinaface", "arcface", "computer-vision"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Recognition",
]
requires-python = ">=3.10"
dependencies = [
    "opencv-python>=4.8.0",
    "insightface>=0.7.3",
    "onnxruntime>=1.18.0",
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
# ⚠️ GPU extras require MANUALLY uninstalling onnxruntime first:
#   pip uninstall onnxruntime
#   pip install facevision[gpu-win]   # Windows
#   pip install facevision[gpu-linux] # Linux
# This is because onnxruntime / onnxruntime-directml / onnxruntime-gpu
# install to the same directory and conflict.

# Windows GPU: DirectML (AMD / NVIDIA / Intel GPU)
gpu-win = ["onnxruntime-directml>=1.24.0"]
# Linux GPU: NVIDIA CUDA
gpu-linux = ["onnxruntime-gpu>=1.18.0"]
# Development / testing
dev = ["pytest", "build", "twine"]
docs = ["mkdocs", "mkdocs-material"]

[project.urls]
Repository = "https://github.com/xxx/FaceVision"

[tool.setuptools.packages.find]
where = ["."]
include = ["face_vision", "face_vision.*"]
```

---

## 二、CI/CD 流水线

### 2.1 `.github/workflows/publish.yml`

```yaml
name: Test & Publish

on:
  push:
    tags:
      - 'v*.*.*'
  pull_request:
    branches: [main]

jobs:
  test:
    name: Test (${{ matrix.os }})
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.10', '3.11', '3.12']
        exclude:
          # Reduce matrix: skip older Python on non-Linux
          - os: macos-latest
            python-version: '3.10'
          - os: windows-latest
            python-version: '3.10'
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest

      - name: Run tests
        run: pytest tests/ -v --ignore=tests/test_face_detector.py

  publish:
    needs: test
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install build tools
        run: pip install build

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

> **说明：**
> - **test** job 使用矩阵策略：`ubuntu-latest` / `macos-latest` / `windows-latest` × Python `3.10` / `3.11` / `3.12`
> - `test_face_detector.py` 在 CI 中跳过（需 insightface 模型下载，CI 无 GPU）；本地测试完整运行
> - **publish** job 仅在 tag push 时触发，且需要 test 全部通过
> - 使用 PyPI trusted publishing（无需 token），首次需在 PyPI 项目设置中关联 GitHub 仓库

---

## 三、发版操作

```bash
# 1. 确保全部测试通过
pytest tests/ -v

# 2. 更新版本号（两处）
#    - pyproject.toml: version = "1.0.0"
#    - face_vision/__init__.py: __version__ = "1.0.0"

# 3. 提交并打 tag
git add -A && git commit -m "Release v1.0.0"
git tag v1.0.0
git push origin main --tags

# 4. GitHub Actions 自动触发 → 测试 → 构建 → 发布到 PyPI
```

---

## 四、README.md

```markdown
# FaceVision

Real-time face recognition library for Python.

- **Detection:** insightface RetinaFace (buffalo_l)
- **Recognition:** ArcFace 512d embedding + cosine similarity 1:N matching
- **Tracking:** IoU-based multi-object tracking with identity voting
- **Backend:** ONNX Runtime (CUDA / DirectML / CPU auto-detection)

## Installation

```bash
# CPU only (default, all platforms, zero-config)
pip install facevision

# --- GPU acceleration ---
# Windows (DirectML):
pip uninstall onnxruntime
pip install facevision[gpu-win]

# Linux (NVIDIA CUDA):
pip uninstall onnxruntime
pip install facevision[gpu-linux]

# macOS: no extra package needed
# Apple Silicon AMX coprocessor + CPU is already fast
```

> **Why the uninstall step?** `onnxruntime` (CPU), `onnxruntime-directml` (GPU),
> and `onnxruntime-gpu` (GPU) are mutually exclusive — they all install to the same
> `onnxruntime/` package directory. You must remove the CPU version before
> installing a GPU variant.

## Quick Start

```python
from face_vision import (
    FaceVisionPipeline,
    FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase,
    CameraThread,
)

# 1. Init components
db = FaceDatabase(db_path="face_db.json")
detector = FaceDetector(device="auto", det_size=640)
recognizer = FaceRecognizer(tolerance=0.45)
tracker = FaceTracker(smooth_frames=5)
camera = CameraThread(camera_id=0, width=640, height=360)

# 2. Build pipeline
pipeline = FaceVisionPipeline(camera, detector, recognizer, tracker, db)
pipeline.start()

# 3. Process frames
while True:
    result = pipeline.process_frame()
    if result is None:
        continue
    for face in result.known_faces:
        print(f"{face.name} ({face.confidence:.2%})")

pipeline.stop()
```

## Platform Support

| Platform | GPU Backend | Camera Backend |
|----------|------------|----------------|
| Windows  | DirectML / CUDA | DShow |
| Linux    | CUDA | V4L2 |
| macOS    | CPU (CoreML planned v1.1) | AVFoundation |
| All      | CPU fallback | — |

## License

MIT © AllenDeng

> **Note:** The FaceVision library code is MIT-licensed. However, the pretrained
> face recognition models (`buffalo_l`) auto-downloaded by `insightface` at runtime
> are **non-commercial research only**. Contact [insightface](https://github.com/deepinsight/insightface)
> for commercial model licensing.
```
