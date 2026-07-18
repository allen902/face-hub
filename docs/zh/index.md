<div class="fh-hero">
<canvas id="fh-particles"></canvas>
<div class="fh-hero-content">
<span class="fh-hero-badge">实时人脸识别 · v1.3.0</span>
<h1 class="fh-hero-title">Face<span class="fh-glow">Hub</span></h1>
<p class="fh-hero-typing"><span id="fh-typer" data-phrases='["检测 · 特征 · 追踪 · 匹配", "按人脸自动整理你的整个相册", "DetectorProtocol 接入任意检测器", "CUDA / DirectML / CPU 自动选择"]'></span><span class="fh-cursor"></span></p>
<div class="fh-hero-actions">
<a class="fh-btn fh-btn--primary" href="installation/">立即开始</a>
<a class="fh-btn fh-btn--ghost" href="api/pipeline/">API 文档</a>
</div>
<div class="fh-stats">
<div class="fh-stat"><span class="fh-stat-num" data-target="512">0</span><span class="fh-stat-label">维特征向量</span></div>
<div class="fh-stat"><span class="fh-stat-num" data-target="30" data-suffix=" FPS">0</span><span class="fh-stat-label">实时流水线</span></div>
<div class="fh-stat"><span class="fh-stat-num" data-target="3" data-suffix=" 平台">0</span><span class="fh-stat-label">跨平台支持</span></div>
<div class="fh-stat"><span class="fh-stat-num" data-target="100" data-suffix="%">0</span><span class="fh-stat-label">开源</span></div>
</div>
</div>
</div>

FaceHub 是一个实时人脸识别 Python 库，提供无 GUI、可独立使用的检测、
特征提取、识别、追踪、摄像头采集能力 —— 并能按人脸整理整个照片集。

## 特性 { .fh-section }

<div class="grid">
<div class="fh-card">
<span class="fh-card-icon">◎</span>
<h3>检测</h3>
<p>insightface RetinaFace，自动检测 CUDA / DirectML GPU 并回退 CPU。</p>
</div>
<div class="fh-card">
<span class="fh-card-icon">⬡</span>
<h3>特征</h3>
<p>ArcFace 512 维 L2 归一化特征向量，可直接用于余弦相似度匹配。</p>
</div>
<div class="fh-card">
<span class="fh-card-icon">⇄</span>
<h3>识别</h3>
<p>1:N 余弦相似度匹配，带版本号编码缓存，人脸库变更时才重建。</p>
</div>
<div class="fh-card">
<span class="fh-card-icon">⋈</span>
<h3>追踪</h3>
<p>IoU 多目标追踪 + 多数投票身份平滑。</p>
</div>
<div class="fh-card">
<span class="fh-card-icon">▦</span>
<h3>照片分类</h3>
<p>按人脸对照片集自动分组 —— 人脸库匹配或全自动聚类，支持导出为按人分类的文件夹。</p>
</div>
<div class="fh-card">
<span class="fh-card-icon">⌘</span>
<h3>可扩展</h3>
<p><code>DetectorProtocol</code> 协议，可接入自定义检测器（YOLO、MediaPipe……）。</p>
</div>
</div>

## 60 秒快速开始 { .fh-section }

<div class="fh-terminal" markdown>
<div class="fh-terminal-bar"><span class="fh-terminal-dot"></span><span class="fh-terminal-dot"></span><span class="fh-terminal-dot"></span><span class="fh-terminal-title">zsh — pip</span></div>

```bash
pip install face-hub
```

</div>

```python
from face_hub import classify_photos, export_to_folders

# 把一个文件夹的照片按人物自动分组
result = classify_photos(["聚会1.jpg", "聚会2.jpg", "聚会3.jpg"])

for label, group in result.groups.items():
    print(label, "→", group.photo_ids)
# person_001 → ['聚会1.jpg', '聚会3.jpg']
# person_002 → ['聚会2.jpg']

# 导出为按人分类的文件夹
export_to_folders(result, "sorted/")
```

## 平台支持 { .fh-section }

| 平台 | 检测后端 | 说明 |
|------|----------|------|
| Windows | DirectML / CUDA / CPU | 自动探测，优先 GPU |
| Linux | CUDA / CPU | 自动探测，优先 GPU |
| macOS | CPU | 仅 CPU，CoreML 计划中 |

## 下一步 { .fh-section }

[安装](installation.md){ .fh-link-card }
[快速开始](quickstart.md){ .fh-link-card }
[照片分类器](api/classifier.md){ .fh-link-card }
[API 文档](api/pipeline.md){ .fh-link-card }
