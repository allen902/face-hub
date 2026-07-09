# FaceHub

FaceHub 是一个实时人脸识别 Python 库，提供无 GUI、可独立使用的检测、
特征提取、识别、追踪与摄像头采集能力。

## 特性

- 基于 insightface RetinaFace 的高精度人脸检测
- ArcFace 512 维 L2 归一化特征向量
- 1:N 余弦相似度匹配，带版本号编码缓存
- IoU 多目标追踪 + 多数投票身份平滑
- 跨平台摄像头采集线程
- `DetectorProtocol` 协议，可接入自定义检测器

## 平台支持

| 平台 | 检测后端 | 说明 |
|------|----------|------|
| Windows | DirectML / CUDA / CPU | 自动探测，优先 GPU |
| Linux | CUDA / CPU | 自动探测，优先 GPU |
| macOS | CPU | v1.0 仅 CPU，CoreML 延后至 v1.1 |

## 下一步

- [安装](installation.md)
- [快速开始](quickstart.md)
- [API 文档](api/pipeline.md)
