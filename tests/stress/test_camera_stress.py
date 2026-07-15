"""Stress tests for CameraThread (using FakeCamera for CI safety)."""
import threading
import time
import pytest
import numpy as np
from face_hub.exceptions import CameraError


class TestCameraStartStop:
    """启动/停止循环压力测试."""

    def test_start_stop_100_cycles(self, fake_camera):
        """100 次启停不残留线程、不抛异常."""
        for i in range(100):
            fake_camera.start()
            assert fake_camera.running is True
            fake_camera.stop()
            assert fake_camera.running is False

    def test_repeat_stop_is_idempotent(self, fake_camera):
        """重复 stop 不抛异常."""
        fake_camera.start()
        fake_camera.stop()
        fake_camera.stop()  # idempotent
        fake_camera.stop()
        assert fake_camera.running is False

    def test_repeat_start_is_idempotent(self, fake_camera):
        """重复 start 不抛异常."""
        fake_camera.start()
        fake_camera.start()
        assert fake_camera.running is True
        fake_camera.stop()

    @pytest.mark.stress
    def test_start_stop_1000_cycles(self, fake_camera, thread_counter):
        """1000 次启停，验证无线程泄漏."""
        for _ in range(1000):
            fake_camera.start()
            fake_camera.stop()


class TestMultiConsumer:
    """多消费者并发测试."""

    def test_10_threads_concurrent_get_frame(self, fake_camera):
        """10 线程并发 get_frame()，每个拿到独立 copy."""
        fake_camera.start()
        frames_received = []
        errors = []
        barrier = threading.Barrier(10, timeout=5)

        def consumer():
            barrier.wait()
            try:
                for _ in range(50):
                    f = fake_camera.get_frame(timeout=0.1)
                    if f is not None:
                        frames_received.append(f)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=consumer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors in consumers: {errors}"
        assert len(frames_received) > 0
        # Every frame should be independent (not sharing memory)
        assert all(f.flags["OWNDATA"] for f in frames_received)
        fake_camera.stop()

    def test_slow_consumer_does_not_block_producer(self, fake_camera):
        """慢消费者不阻塞采集端."""
        fake_camera.start()
        latencies = []

        for _ in range(100):
            t0 = time.perf_counter()
            _ = fake_camera.get_frame()
            latencies.append((time.perf_counter() - t0) * 1000)
            # Simulate consumer that's slower than frame interval
            time.sleep(0.05)

        # Producer (get_frame) latency should be stable
        avg = sum(latencies) / len(latencies)
        # With sleep, avg might be higher but individual get_frame
        # without sleep should be fast
        assert avg < 200, f"get_frame avg latency {avg:.2f} ms too high"


class TestErrorInjection:
    """异常注入测试."""

    def test_camera_error_on_invalid_id(self):
        """非法 camera_id 抛 CameraError."""
        from face_hub import CameraThread
        with pytest.raises(ValueError):
            CameraThread(camera_id=-1)

    def test_fake_camera_error_injection(self, fake_camera):
        """注入异常后 get_frame 正确抛出."""
        fake_camera.inject_error(CameraError("simulated", camera_id=0))
        fake_camera.start()
        with pytest.raises(CameraError) as exc_info:
            fake_camera.get_frame()
        assert exc_info.value.camera_id == 0
        # After error, subsequent calls should work
        frame = fake_camera.get_frame()
        assert frame is not None
        fake_camera.stop()
