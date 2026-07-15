下面是我基于项目结构为 FaceHub 设计的压力与稳定性测试方案。方案以“可自动化、可复现、不依赖真实摄像头/GPU”为核心，同时保留真实模型的手动/夜间测试入口。

1. 项目结构理解
FaceHub 是一条典型的人脸识别流水线，核心数据流：


CameraThread.get_frame()
        ↓
FaceDetector.detect_with_embeddings()
        ↓
FaceTracker.update(detections, recognizer=FaceRecognizer)
        ↓
FaceHubPipeline.process_frame()
关键组件与风险点：

组件	职责	稳定性关注点
CameraThread	独立线程采集、条件变量分发最新帧	启停泄漏、多消费者、maxlen=1 丢帧
FaceDetector	InsightFace RetinaFace + ArcFace 嵌入	模型加载/热切换、GPU 异常降级、内存泄漏、并发锁
FaceRecognizer	1:N 余弦相似度匹配	大库查询性能、缓存一致性、异常维度
FaceTracker	IoU 匹配 + 多数投票平滑	大目标数 O(N²)、track 过期、身份抖动
FaceDatabase	JSON + npy 原子写入、版本号	并发 CRUD、文件损坏、路径穿越
FaceHubPipeline	总控 + FPS 统计 + 缓存同步	长时间 soak、异常隔离、锁竞争
现有 benchmark.py 已经能做性能采集，但缺少断言、内存回归检查、异常注入和并发场景，这是本次方案要补齐的。

2. 测试目标
长时间稳定运行：连续处理 10k~100k 帧不崩溃、内存增长可控。
并发与资源安全：多线程调用组件时无死锁、无句柄/线程泄漏。
降级路径可靠：GPU 异常、模型热切换、空库、坏帧都能优雅处理。
性能可度量：FPS、p50/p95/p99 延迟、内存峰值、大库匹配耗时。
CI 可复现：默认用 Mock 组件跑；真实模型/摄像头测试标记为 slow/hardware，手动或夜间执行。
3. 测试策略
分层：单组件 → FaceHubPipeline → 端到端真实模型。
Mock 优先：
FakeCamera：固定返回合成帧。
FakeDetector：返回可控 bbox/embedding。
FakeRecognizer：模拟匹配耗时、维度异常、缓存失效。 这样可以在没有 GPU/摄像头/真实人脸的情况下做可复现压力测试。
真实模型慢测：复用 benchmark.py 逻辑，加上内存和断言，作为 pytest --runslow 或 nightly job。
异常注入：在 fake 组件里故意抛 InferenceError、ValueError、OSError，验证外层捕获与降级。
回归阈值：把基线数据写入 tests/stress/baseline.json，CI 对比偏差。
4. 测试维度与用例
4.1 CameraThread
用例	场景	验证点
启停循环	start()/stop() 100/1000 次	thread.join() 退出、cap.release()、无残留线程
多消费者	10 线程并发 get_frame()	每个消费者拿到独立 copy，不“偷帧”
慢消费者	消费者处理耗时 > 帧间隔	maxlen=1 缓冲正常丢旧帧，不阻塞采集线程
异常输入	非法 camera_id、打开失败	抛出 CameraError，属性完整
4.2 FaceDetector
用例	场景	验证点
长时推理	连续推理 1k/5k 帧（CPU）	平均 FPS 不下降、内存增量 < 阈值、_inference_error_count == 0
并发推理	多线程同时 detect_with_embeddings	_lock 不导致死锁、结果正确
模型热切换	多次 reload_model(det_size=...)	旧模型释放、新模型可用、VRAM/内存不累积
异常帧	空图、1x1、非 uint8、None、超大分辨率	返回 []，不崩溃
GPU 降级	Mock app.get 连续抛异常	第 3 次后自动 fallback 到 CPU，后续调用恢复
DirectML reshape bug	注入对应异常字符串	自动切到 det_size=640 或 CPU
4.3 FaceRecognizer
用例	场景	验证点
大库匹配	1k / 5k / 10k 底库，多次查询	p95 延迟、内存占用
缓存一致性	update_cache 同 version / 不同 version	同 version 返回 False，不同 version 重建
异常输入	空 gallery、维度 256/513、None embedding、names/encodings 长度不一致	返回 (unknown, 0.0) 或抛 ValueError
并发更新	线程 A update_cache、线程 B recognize	在 pipeline 锁下无异常；单独使用时文档说明非线程安全
4.4 FaceTracker
用例	场景	验证点
大规模跟踪	单帧 50/100 个检测，连续 1000 帧	内存不爆、耗时稳定、track_count 合理
身份抖动	同一 track 名字在 A/B/unknown 间交替	majority vote 抑制抖动，is_confirmed 稳定
丢失恢复	连续 max_missed 帧无匹配	track 被移除；再次出现时生成新 track
并发更新	多线程同时 update	状态一致、无竞态
历史截断	运行多帧后	name_history 长度不超过 smooth_frames*3
4.5 FaceDatabase
用例	场景	验证点
并发 CRUD	10 线程各 100 次 add/remove	version 单调递增、JSON/npy 不损坏、最终一致性
损坏恢复	写入坏 JSON、shape 错的 npy	抛出 SerializationError/DatabaseError
大批量	5000 条记录 save/load	耗时、内存、文件权限
路径穿越	绝对路径、../etc/passwd	抛出 ValueError
4.6 FaceHubPipeline
用例	场景	验证点
Soak test	Mock 组件连续 process_frame 10k/100k 帧	FPS 稳定、内存增长 < 阈值、_total_frame_count 正确
启停幂等	start()/stop() 多次、重复 stop	不抛异常、状态正确
异常隔离	detector/recognizer/tracker/db 抛异常	包装为 FaceHubError，流水线可继续运行
动态增删库	运行中 db.add_person/remove_person	_last_db_version 同步、识别结果变化
多线程调用	多线程同时 process_frame	_lock 不饿死、无死锁
FPS 统计	固定帧率运行	fps 字段非负、与真实耗时偏差 < 10%
5. 关键指标与通过标准
指标	工具	建议阈值
内存增长	tracemalloc / psutil.Process.memory_info().rss	10k 帧增长 < 50 MB
FPS 下降	time.perf_counter()	后 50% 帧平均 FPS 不低于前 50% 的 80%
p95 延迟	同 benchmark.py 的 percentile	不高于基线 20%
异常数	pytest + 异常注入	0 未捕获异常
句柄/线程泄漏	threading.enumerate() / psutil	测试结束回到基线 ±1
数据库一致性	version 校验	并发操作后 version 单调且等于操作总数
阈值可写入 tests/stress/baseline.json，CI 自动比对。

6. 工具与基础设施

# pyproject.toml 增加 markers
[tool.pytest.ini_options]
markers = [
    "stress: 压力/稳定性测试（默认跳过）",
    "slow:   真实模型/长时间测试",
    "memory: 内存回归测试",
]
新增文件规划：


tests/
├── stress/
│   ├── conftest.py            # FakeCamera / FakeDetector / FakeRecognizer / memory helper
│   ├── test_camera_stress.py
│   ├── test_detector_stress.py
│   ├── test_recognizer_stress.py
│   ├── test_tracker_stress.py
│   ├── test_database_stress.py
│   ├── test_pipeline_stress.py
│   └── baseline.json          # 性能基线
关键 Fixture 示例

# tests/stress/conftest.py
import numpy as np
import pytest
from face_hub.types import DetectionWithEmbedding, BBox

class FakeDetector:
    def __init__(self, detections_factory):
        self._factory = detections_factory
        self.det_size = 640
        self.device = "cpu"
        self._error_count = 0

    def detect_with_embeddings(self, frame):
        if self._error_count:
            self._error_count -= 1
            raise RuntimeError("injected inference error")
        return self._factory(frame)

    def reload_model(self, det_size=None):
        if det_size is not None:
            self.det_size = det_size


class FakeCamera:
    running = False
    def start(self): self.running = True
    def stop(self):  self.running = False
    def get_frame(self, timeout=None, copy=True):
        return np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)


@pytest.fixture
def memory_helper():
    import tracemalloc, psutil, gc
    gc.collect()
    tracemalloc.start()
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    snap_before = tracemalloc.take_snapshot()
    yield type("H", (), {
        "rss": lambda: proc.memory_info().rss - rss_before,
        "snapshot": lambda: tracemalloc.take_snapshot().compare_to(snap_before, "lineno"),
    })
    tracemalloc.stop()
7. 实施路线
阶段	内容	预计产出
Phase 1	基础设施	pytest.ini markers、tests/stress/conftest.py、Fake 组件、内存 helper
Phase 2	单组件压力测试	test_camera/detector/recognizer/tracker/database_stress.py
Phase 3	Pipeline soak 测试	test_pipeline_stress.py，包含异常注入与动态库更新
Phase 4	真实模型基准	扩展 benchmark.py 或新增 test_real_model_slow.py，生成 baseline.json
Phase 5	CI 集成	GitHub Actions nightly job：pytest -m "stress or slow"，上传报告
8. 风险与注意事项
GPU/DirectML 环境差异大：真实模型测试应在独立 job 运行，失败不阻塞 PR。
模型首次加载慢：pytest-timeout 对加载阶段要单独设置宽松时间。
Windows 句柄检测：os.chmod 在 Windows 上可能失败，测试里不要当作致命错误。
不要依赖真实摄像头：压力测试默认用 FakeCamera，保证 CI 和本地结果一致。
线程安全边界：FaceRecognizer 没有内部锁，并发测试要在 FaceHubPipeline._lock 覆盖下验证；单独并发调用应视为未定义行为并文档化。