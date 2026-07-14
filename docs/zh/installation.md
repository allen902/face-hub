# 安装

## 基础安装

```bash
pip install face-hub
```

## GPU 加速（可选）

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

## 开发依赖

```bash
pip install face-hub[dev]
```

## 平台说明

- **Windows**：默认使用 DirectShow 后端打开摄像头。
- **macOS**：仅支持 CPU 推理；摄像头后端为 AVFoundation。
- **Linux**：摄像头后端为 V4L2。

## 模型许可

FaceHub 代码采用 MIT License。但 insightface 自动下载的 `buffalo_l` 预训练模型
受其非商用模型许可约束，商业使用需向 insightface 申请授权。
