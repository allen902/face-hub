# FaceVision — Developer Setup Guide

> 本文档列出 AI 无法代劳的事项：你需要自己准备和操作的内容。
> 按优先级排序，完成一个勾一个。

---

## 一、账号与基础设施

### 1.1 GitHub 仓库

- [ ] 在 GitHub 创建新仓库 `FaceVision`（建议 Public）
- [ ] 仓库设置 → Actions → General → 勾选 "Allow GitHub Actions to create and approve pull requests"
- [ ] Settings → Branches → 添加分支保护规则：`main` 分支要求 PR + 1 review
- [ ] 本地 `git init`、关联 remote、首次 push

### 1.2 PyPI 账号

- [ ] 注册 [pypi.org](https://pypi.org) 账号
- [ ] 创建 API token：Account Settings → API tokens → "facevision" → 勾选 "Upload packages"
- [ ] **或** 使用 PyPI Trusted Publishing（推荐，无需 token）：
  - PyPI 项目页面 → Settings → Publishing → 添加 Trusted Publisher
  - Owner: 你的 GitHub 用户名
  - Repository: `FaceVision`
  - Workflow: `publish.yml`
  - Environment: 留空

### 1.3 Test PyPI（可选，建议先试）

- [ ] 注册 [test.pypi.org](https://test.pypi.org)
- [ ] 同样配置 Trusted Publishing 或 API token
- [ ] 首次发布前在 Test PyPI 试跑一轮

---

## 二、仓库初始化（代码到位后）

### 2.1 配置文件检查

- [ ] `pyproject.toml` 中 `Repository` URL 改为你的实际 GitHub 地址
- [ ] `LICENSE` 文件确认 `Copyright (c) [Year] AllenDeng`
- [ ] `README.md` 中链接、安装命令确认可访问

### 2.2 Git 设置

- [ ] 创建 `.gitignore`（至少忽略以下内容）：

```
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# Virtual environments
venv/
.venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# FaceVision runtime files
face_db.json
encodings.pkl
settings.json
face_photos/

# Test artifacts
test_db.json
test_enc.pkl
```

- [ ] 首次 commit 前确认不包含任何私人数据或 API key

---

## 三、法律与许可

### 3.1 自身许可

- [ ] 确认 MIT License 全文放入 `LICENSE` 文件，署名 `AllenDeng`
- [ ] `pyproject.toml` 中 `authors` 和 license 字段确认无误

### 3.2 insightface 模型许可（重要）

- [ ] 如果计划商用，联系 `recognition-oss-pack@insightface.ai` 获取 buffalo_l 模型商用授权
- [ ] 如果非商用（学术/个人），无需操作，但应在 README 中保留警告
- [ ] 在 README License 段落中保留 insightface 模型的非商用声明

### 3.3 第三方依赖声明

- [ ] 确认 `NOTICE` 或 `README` 中列出了 Apache 2.0 / BSD 依赖的版权声明（推荐但非强制）

---

## 四、测试与验证（手动）

### 4.1 本地环境

- [ ] 创建并激活虚拟环境：`python -m venv venv`
- [ ] `pip install -e .` 从本地安装
- [ ] `pip install -e .[dev]` 安装测试工具
- [ ] `pytest tests/ -v --ignore=tests/test_face_detector.py` 确认全部通过

### 4.2 真实硬件测试

- [ ] **Windows + GPU：** 测试 `device="auto"` 是否自动选中 DirectML
- [ ] **Windows + CPU：** 测试 `device="cpu"` 模式
- [ ] **Linux：** 测试摄像头 V4L2 后端 + CUDA（如有）
- [ ] **macOS：** 测试 AVFoundation 摄像头 + CPU（确认 CoreML 不意外启用）
- [ ] **无摄像头环境：** 确认 `process_frame(frame=img)` 传入静态帧正常工作

### 4.3 端到端验证

- [ ] `FaceDatabase` 增删查改 + 持久化（关掉重开数据不丢）
- [ ] `FaceDetector` 不同 `det_size` 切换（320 → 480 → 640）
- [ ] `FaceRecognizer` 同一个人多次识别，相似度 > 0.80
- [ ] `FaceTracker` 多人脸同时追踪，track_id 正确分配
- [ ] `FaceVisionPipeline` 完整链路跑通（摄像头 → 检测 → 识别 → 追踪）

---

## 五、发布操作

### 5.1 首次发布 v1.0.0 流程

```bash
# 1. 确认全部测试通过
pytest tests/ -v --ignore=tests/test_face_detector.py

# 2. 更新版本号（两处）
#    - pyproject.toml: version = "1.0.0"
#    - face_vision/__init__.py: __version__ = "1.0.0"

# 3. 本地构建验证
pip install build
python -m build
twine check dist/*

# 4. （可选）先上传 Test PyPI
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ facevision
# 验证: python -c "from face_vision import FaceVisionPipeline; print('OK')"

# 5. 提交、打 tag、推送
git add -A
git commit -m "Release v1.0.0"
git tag v1.0.0
git push origin main --tags

# 6. 等待 GitHub Actions 完成（test → publish）
# 7. 在 PyPI 确认包已上线: https://pypi.org/project/facevision/
```

### 5.2 发布后验证

- [ ] 在新虚拟环境中 `pip install facevision` 安装最新版
- [ ] `python -c "from face_vision import *; print('OK')"` 无错误
- [ ] 跑一遍 Quick Start 示例代码

---

## 六、文档与社区

### 6.1 文档部署

- [ ] `pip install mkdocs mkdocs-material`
- [ ] `cd docs && mkdocs build` 确认无构建错误
- [ ] `mkdocs gh-deploy` 部署到 GitHub Pages
- [ ] 确认中英文版本均可正常浏览

### 6.2 可选的社区配置

- [ ] GitHub 仓库 About 区域：填写描述、标签（face-recognition, insightface, python）
- [ ] 添加 `CONTRIBUTING.md`（如接受贡献）
- [ ] 添加 `CHANGELOG.md`（记录每个版本的变更）
- [ ] 添加 Issue Templates（Bug Report / Feature Request）
- [ ] 在 README 中添加 badges（PyPI version, Python versions, License, tests passing）

---

## 七、风险清单

| 风险 | 影响 | 你的动作 |
|------|------|----------|
| insightface 模型商用限制 | 商业使用可能需授权 | 确认使用场景，必要时联系 insightface |
| ONNX Runtime GPU 变体冲突 | 用户安装 GPU 版失败 | README 已写明 `uninstall` 步骤 |
| 摄像头权限 | macOS 需用户授权终端/IDE 访问摄像头 | README 加提示 |
| 模型首次下载 | 首次 `FaceDetector()` 需下载 ~200MB 模型 | README 注明首次启动耗时 |
| PyPI 包名被占用 | `facevision` 可能已被注册 | 提前在 PyPI 检查；备选 `face-vision` |
| GitHub Actions 分钟数 | 三平台 × 3 Python 版本矩阵较大 | 关注 Actions 使用量；可减少矩阵 |

---

## 八、后续版本规划

| 版本 | 计划内容 |
|------|----------|
| v1.0.0 | 首次发布：同步接口，CPU + DirectML + CUDA |
| v1.0.x | Bug 修复 |
| v1.1.0 | macOS CoreML `device="coreml"`；async `aprocess_frame()` |
| v1.2.0 | 检测器 Registry（按名称注册/切换自定义检测器） |
| v2.0.0 | 破坏性 API 变更（如需要） |
