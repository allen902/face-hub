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
Repository = "https://github.com/allen902/FaceVision"

[tool.setuptools.packages.find]
where = ["."]
include = ["face_vision", "face_vision.*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

### 1.2 `MANIFEST.in`

确保非 Python 文件被打包进 sdist：

```
include README.md
include LICENSE
include pyproject.toml
recursive-include docs *.md *.yml *.yaml *.css *.js *.png *.svg
prune docs/.cache
prune docs/site
```

> **说明：** `MANIFEST.in` 控制 `sdist` (`.tar.gz`) 中包含的文件。`wheel` 默认只包含 `.py` 文件，
> 所以文档等非代码文件需通过 `MANIFEST.in` 声明。`docs/` 目录会被包含以支持离线文档浏览。

---

## 二、CI/CD 流水线

### 2.1 `.github/workflows/ci.yml`（PR 测试）

```yaml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    name: Test (${{ matrix.os }} / ${{ matrix.python-version }})
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.10', '3.11', '3.12']
        exclude:
          # Reduce matrix: skip oldest Python on non-Linux
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
          pip install -e ".[dev]"

      - name: Run tests (skip GPU-dependent detector tests)
        run: pytest tests/ -v --ignore=tests/test_face_detector.py

      - name: Verify package imports
        run: |
          python -c "from face_vision import (
            FaceVisionPipeline, FaceDetector, FaceRecognizer,
            FaceTracker, FaceDatabase, CameraThread, UNKNOWN_SENTINEL,
            DEFAULT_SETTINGS, BBox, DetectionResult, DetectionWithEmbedding,
            TrackedFace, PipelineResult,
          )"
          python -c "from face_vision.exceptions import (
            FaceVisionError, ModelLoadError, InferenceError,
            CameraError, DatabaseError,
          )"

  lint:
    name: Lint check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Check import structure
        run: |
          python -c "import ast, sys, pathlib
          # Verify face_vision/__init__.py exports match __all__
          init = pathlib.Path('face_vision/__init__.py').read_text()
          tree = ast.parse(init)
          print('face_vision package structure valid')
          "
```

### 2.2 `.github/workflows/publish.yml`（打 tag 自动发布）

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  test:
    name: Pre-publish test (${{ matrix.os }})
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.11', '3.12']
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install and test
        run: |
          pip install -e ".[dev]"
          pytest tests/ -v --ignore=tests/test_face_detector.py

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install build tools
        run: pip install build twine

      - name: Build package
        run: python -m build

      - name: Check package metadata
        run: twine check dist/*

      - name: Store build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4

      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1

  docs:
    needs: publish
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install docs dependencies
        run: pip install mkdocs mkdocs-material

      - name: Build and deploy docs
        run: |
          cd docs
          mkdocs build --clean
          mkdocs gh-deploy --force
```

> **说明：**
> - **CI** workflow：每次 PR 和 push 到 main 触发，跑全矩阵测试 + import 完整性检查
> - **Publish** workflow：仅 tag push (`v*.*.*`) 触发
>   - `test` job：跑精简矩阵 (3.11 / 3.12)
>   - `build` job：构建 sdist + wheel，twine check 校验元数据，upload artifact
>   - `publish` job：使用 PyPI trusted publishing（无需 token），首次需在 PyPI 项目设置 → "Trusted Publisher Management" 中关联 GitHub 仓库
>   - `docs` job：发布成功后自动部署 MkDocs 到 GitHub Pages
> - CI 中 `test_face_detector.py` 跳过（需 insightface 模型下载，CI 无 GPU）；本地测试完整运行

### 2.3 PyPI Trusted Publishing 首次配置

1. 登录 [pypi.org](https://pypi.org) → 进入 `facevision` 项目 → **Settings** → **Trusted Publisher Management**
2. 添加：
   - **Owner:** `allen902`
   - **Repository:** `FaceVision`
   - **Workflow:** `publish.yml`
   - **Environment:** (留空)
3. 保存后即可免 token 发布（OIDC 认证）

---

## 三、版本号规范 (SemVer)

```
MAJOR.MINOR.PATCH    (e.g. 1.0.0)

MAJOR: 不兼容的 API 变更（公开类/方法名变化、参数签名变化、返回值类型变化）
MINOR: 向后兼容的功能新增（新组件、新方法、新可选参数）
PATCH: 向后兼容的 bug 修复（性能优化、文档修正、日志调整）
```

### 版本号出现位置（全部需同步更新）

| 位置 | 字段 | 示例 |
|------|------|------|
| `pyproject.toml` | `project.version` | `version = "1.0.0"` |
| `face_vision/__init__.py` | `__version__` | `__version__ = "1.0.0"` |

> **v1.0 路线图：**
> - `v1.0.0` — 初始发布：RetinaFace 检测、ArcFace 特征、IoU 追踪、余弦相似度匹配、多平台摄像头
> - `v1.0.x` — Bug 修复（如有）
> - `v1.1.0` — macOS CoreML 支持 (`device="coreml"`)
> - `v1.2.0` — Async 接口 (`asyncio.to_thread` 包装)
> - `v2.0.0` — 破坏性 API 变更（如有）

---

## 四、README.md

> README 是中英双语产品首页，发布到 PyPI 和 GitHub 均使用此文件。
> **实际内容已写在仓库根目录的 `README.md` 中，此处不再重复。**
> 下面列出 README 的内容清单和编写要点。

### 4.1 README 内容清单

| 章节 | 必填 | 状态 | 说明 |
|------|:--:|:--:|------|
| 项目标题 + 一句话描述 | ✅ | ✅ | `# FaceVision — Real-time face recognition library` |
| Badges (Python / License / CI) | ✅ | ✅ | shields.io 徽章 |
| Features 列表 | ✅ | ✅ | 6 个核心特性，bullet points |
| Installation | ✅ | ✅ | CPU + GPU 三种安装方式，含 uninstall 警告 |
| Quick Start | ✅ | ✅ | ~25 行完整可运行示例 |
| Custom Detector | 推荐 | ✅ | DetectorProtocol 使用示例 |
| Documentation 链接 | 推荐 | ✅ | MkDocs 本地预览命令 |
| License 说明 | ✅ | ✅ | MIT + insightface 模型非商用警告 |
| 中文版（完整独立） | 推荐 | ✅ | 与英文版一一对应 |
| 平台支持表 | 推荐 | ⬜ | 可选：Windows/Linux/macOS GPU+Camera 矩阵 |

### 4.2 README 编写要点

1. **PyPI 长描述 = README.md 全文**（`readme = "README.md"` 在 pyproject.toml 中声明）
2. **PyPI 不支持 Markdown 中的 HTML Badges 样式**，但 shields.io badge 图片链接通常渲染良好
3. **安装命令必须是可直接复制粘贴的** — GPU extras 的 `pip uninstall` 步骤要显式写出
4. **Quick Start 代码必须是完整可运行的** — 用户复制后只需改 `camera_id` 就能跑
5. **License 免责声明必须在显眼位置** — insightface 模型非商用是用户最容易踩的坑

---

## 五、LICENSE 文件

在仓库根目录创建 `LICENSE` 文件：

```
MIT License

Copyright (c) 2025 AllenDeng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

> **注意：** MIT License 仅覆盖 FaceVision **代码**。`insightface` 自动下载的 `buffalo_l`
> 预训练模型受 insightface 的模型许可约束，README 中已做声明。

---

## 六、发版操作

### 6.1 发布前检查清单

```
[ ] 全部测试通过：pytest tests/ -v
[ ] 本地完整测试（含 GPU 检测器）：pytest tests/ -v（不 skip test_face_detector）
[ ] import 完整性：python -c "from face_vision import *"
[ ] 版本号已同步到两处：pyproject.toml + face_vision/__init__.py
[ ] README.md 内容为最新：链接、示例代码、版本号
[ ] CHANGELOG 已更新（如有）
[ ] git status 干净：无未提交的修改
[ ] 所有 PR 已合入 main 分支
[ ] CI 全绿（最后一次 push）
```

### 6.2 发版步骤

```bash
# ── Step 1: 最终验证 ──
pytest tests/ -v

# ── Step 2: 本地构建验证 ──
pip install build twine
python -m build
twine check dist/*

# ── Step 3: 试安装验证（可选，在虚拟环境中） ──
python -m venv /tmp/test-release
source /tmp/test-release/bin/activate  # Windows: \tmp\test-release\Scripts\activate
pip install dist/facevision-1.0.0-py3-none-any.whl
python -c "from face_vision import FaceDetector; d = FaceDetector(device='cpu', det_size=320); print('OK')"
deactivate

# ── Step 4: 提交并打 tag ──
git add -A
git commit -m "Release v1.0.0"
git tag -a v1.0.0 -m "FaceVision v1.0.0 — Initial release"
git push origin main --tags

# ── Step 5: 等待 CI ──
# GitHub Actions 自动触发 → Test → Build → Publish → Docs
# 监控: https://github.com/allen902/FaceVision/actions

# ── Step 6: 手动发布到 TestPyPI（可选，预验证） ──
twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ facevision
```

### 6.3 发布后验证

```bash
# 1. PyPI 页面可访问
#    https://pypi.org/project/facevision/

# 2. pip 安装验证（新环境）
pip install facevision
python -c "
from face_vision import (
    FaceVisionPipeline, FaceDetector, FaceRecognizer,
    FaceTracker, FaceDatabase, CameraThread, UNKNOWN_SENTINEL,
)
print('All imports OK')
"

# 3. GPU 安装验证（Windows）
pip uninstall -y onnxruntime
pip install facevision[gpu-win]
python -c "from face_vision import FaceDetector; d = FaceDetector(device='auto', det_size=320); print('GPU OK')"

# 4. 文档可访问
#    https://allen902.github.io/FaceVision/

# 5. GitHub Release 页面创建
#    https://github.com/allen902/FaceVision/releases/new?tag=v1.0.0
```

### 6.4 回滚方案

如果发布后发现严重问题：

```bash
# 1. 从 PyPI 移除该版本（yank，不禁用已安装的）
#    在 https://pypi.org/manage/project/facevision/releases/ 操作

# 2. 在 GitHub Release 页面标记为 pre-release 或删除

# 3. 修复后发布 patch 版本 (v1.0.1)
git checkout main
# ... fix bugs ...
git commit -m "Hotfix: ..."
git tag -a v1.0.1 -m "FaceVision v1.0.1 — Hotfix"
git push origin main --tags
```

---

## 七、CHANGELOG 管理

### 7.1 `CHANGELOG.md` 格式（Keep a Changelog 风格）

```markdown
# Changelog

All notable changes to FaceVision will be documented in this file.

## [1.0.0] — 2025-07-09

### Added
- Initial release of face_vision package
- FaceDetector: insightface RetinaFace with CUDA/DirectML/CPU auto-detection
- FaceRecognizer: 1:N cosine similarity matching with versioned cache
- FaceTracker: IoU-based multi-object tracking with majority-vote identity smoothing
- FaceDatabase: JSON + pickle persistence with CRUD operations
- CameraThread: cross-platform camera capture (DShow/AVFoundation/V4L2)
- FaceVisionPipeline: unified processing pipeline with thread-safe API
- DetectorProtocol: plug-in interface for custom detection models
- Dataclass types: BBox, DetectionResult, DetectionWithEmbedding, TrackedFace, PipelineResult
- Exception hierarchy: FaceVisionError base + 5 subclasses
- Default settings with deepcopy isolation
- Full pytest test suite (8 test modules)
- Bilingual README (English + 中文)
- MkDocs-based API documentation (English + 中文)

[1.0.0]: https://github.com/allen902/FaceVision/releases/tag/v1.0.0
```

### 7.2 CHANGELOG 更新规则

- **每次 release 前更新**，放在 release commit 中
- 格式：`[version] — YYYY-MM-DD`
- 分类：`Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`
- 底部保留版本比较链接：`[version]: https://github.com/...`

---

## 八、文档发布

### 8.1 MkDocs 配置 (`docs/mkdocs.yml`)

```yaml
site_name: FaceVision
site_description: Real-time face recognition library — API documentation
site_author: AllenDeng
repo_url: https://github.com/allen902/FaceVision
repo_name: allen902/FaceVision

theme:
  name: material
  palette:
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.sections
    - navigation.expand
    - search.suggest
    - search.highlight
    - content.code.copy
    - content.code.annotate

plugins:
  - search:
      lang:
        - en
        - zh

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences
  - admonition
  - pymdownx.details

nav:
  - English:
      - Home: en/index.md
      - Installation: en/installation.md
      - Quick Start: en/quickstart.md
      - API Reference:
          - Pipeline: en/api/pipeline.md
          - Detector: en/api/detector.md
          - Recognizer: en/api/recognizer.md
          - Tracker: en/api/tracker.md
          - Database: en/api/database.md
          - Camera: en/api/camera.md
          - Types: en/api/types.md
          - Exceptions: en/api/exceptions.md
      - Examples:
          - Basic Webcam: en/examples/basic_webcam.md
          - Custom Detector: en/examples/custom_detector.md
  - 中文:
      - 首页: zh/index.md
      - 安装: zh/installation.md
      - 快速开始: zh/quickstart.md
      - API 文档:
          - Pipeline: zh/api/pipeline.md
          - 检测器: zh/api/detector.md
          - 识别器: zh/api/recognizer.md
          - 追踪器: zh/api/tracker.md
          - 数据库: zh/api/database.md
          - 摄像头: zh/api/camera.md
          - 类型: zh/api/types.md
          - 异常: zh/api/exceptions.md
      - 示例:
          - 基础摄像头: zh/examples/basic_webcam.md
          - 自定义检测器: zh/examples/custom_detector.md
```

### 8.2 文档本地预览

```bash
pip install mkdocs mkdocs-material
cd docs
mkdocs serve
# → http://127.0.0.1:8000
```

### 8.3 文档部署

```bash
# 手动部署（需要 GitHub 写权限）
cd docs
mkdocs gh-deploy --force

# 自动部署：打 tag 后 CI publish workflow 的 docs job 自动执行
```

---

## 九、发布会签表

| 步骤 | 负责人 | 完成标准 | 签字 |
|------|--------|----------|:--:|
| 代码冻结 | — | 所有 feature PR 已合入 main | ⬜ |
| 完整测试 | — | `pytest tests/ -v` 全部通过 | ⬜ |
| 版本号更新 | — | `pyproject.toml` + `__init__.py` 版本号一致 | ⬜ |
| CHANGELOG | — | `CHANGELOG.md` 包含本次所有变更 | ⬜ |
| README 复核 | — | 链接和示例代码可运行 | ⬜ |
| 本地构建 | — | `python -m build && twine check dist/*` 通过 | ⬜ |
| 虚拟环境安装测试 | — | 新 venv 中 `pip install dist/*.whl` + import 测试 | ⬜ |
| 打 tag | — | `git tag -a v1.0.0` 推送成功 | ⬜ |
| CI 全绿 | — | GitHub Actions all green → PyPI published | ⬜ |
| PyPI 验证 | — | `pip install facevision` 从 PyPI 成功安装 | ⬜ |
| 文档上线 | — | `https://allen902.github.io/FaceVision/` 可访问 | ⬜ |
| GitHub Release | — | Release 页面包含 changelog 摘要 + artifact links | ⬜ |

---

## 十、常见问题

### Q1: `twine check` 报 `long_description` 渲染错误？

**原因：** README.md 中可能有 PyPI 不支持的 Markdown 语法（某些 HTML 标签、GitHub 专用扩展）。

**解决：**
```bash
# 使用 PyPI 的渲染检查（需先安装 readme_renderer）
pip install readme_renderer
python -m readme_renderer README.md
```

### Q2: PyPI 发布后 `pip install` 找不到新版本？

**原因：** PyPI CDN 缓存（最长 ~10 分钟）。

**解决：** 等待 10-15 分钟后重试，或指定版本号：
```bash
pip install facevision==1.0.0
```

### Q3: Trusted Publishing 失败（403）？

**排查：**
1. PyPI 项目名 `facevision` 是否已注册（需先在 PyPI 手动创建项目）
2. Trusted Publisher 配置是否正确（Owner → Repo → Workflow 三者匹配）
3. GitHub Actions 日志中 OIDC token 是否成功获取

### Q4: `onnxruntime` 与 `onnxruntime-directml` 冲突？

这是已知问题，已在 README 中说明。用户需要：
```bash
pip uninstall onnxruntime
pip install facevision[gpu-win]
```
因为两者安装到同一个 `onnxruntime/` 目录，无法共存。

### Q5: 发布后本地 `pip install -e .` 的行为变了？

`pip install -e .` (editable install) 和 `pip install facevision` (PyPI release) 安装方式不同：
- editable: `face_vision/` 目录作为 symlink → 修改代码立即生效
- release: 复制到 `site-packages/` → 需重新安装才能更新

本地开发始终用 `pip install -e .`；CI 和用户始终用 `pip install facevision`。

---

## 十一、相关文件索引

| 文件 | 用途 |
|------|------|
| [RELEASE_PLAN.md](RELEASE_PLAN.md) | 本文件 — 打包、CI/CD、发版流程 |
| [REFACTOR_PLAN.md](REFACTOR_PLAN.md) | 新仓库结构、代码改动清单、测试计划 |
| [DEVELOPER.md](DEVELOPER.md) | 开发者指南 — 本地环境搭建、架构说明、调试技巧 |
| [README.md](README.md) | 项目首页 — 中英双语、安装、快速开始、License |
| [CHANGELOG.md](CHANGELOG.md) | 版本变更记录 |
| `pyproject.toml` | 包配置、依赖、构建系统 |
| `MANIFEST.in` | sdist 包含文件声明 |
| `LICENSE` | MIT License 文本 |
| `.github/workflows/ci.yml` | PR 自动测试 |
| `.github/workflows/publish.yml` | Tag 自动发布到 PyPI + 文档部署 |
| `docs/mkdocs.yml` | 文档站点配置 |
| `docs/en/` / `docs/zh/` | 中英 API 文档 |
