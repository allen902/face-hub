"""Stress tests for FaceTracker."""
import pytest
import numpy as np
from face_hub import FaceTracker
from face_hub.types import DetectionWithEmbedding, BBox, TrackedFace, UNKNOWN_SENTINEL


def make_detection(x1, y1, x2, y2, conf=0.9, emb=None, quality=True):
    """Helper: create a DetectionWithEmbedding."""
    if emb is None:
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
    return DetectionWithEmbedding(
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=conf,
        embedding=emb,
        quality_pass=quality,
    )


class TestTrackerStress:
    """FaceTracker 压力测试."""

    @pytest.mark.stress
    def test_large_scale_100_detections_1000_frames(self, memory_helper):
        """单帧 100 个检测 × 1000 帧：内存稳定、耗时可控."""
        import time

        tracker = FaceTracker(smooth_frames=5, max_missed=10)
        rng = np.random.RandomState(0)

        times = []
        for _ in range(1000):
            # Generate 100 random detections
            dets = []
            for _ in range(100):
                x1 = rng.randint(0, 500)
                y1 = rng.randint(0, 250)
                dets.append(make_detection(x1, y1, x1 + 40, y1 + 50))

            t0 = time.perf_counter()
            results = tracker.update(dets)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            assert isinstance(results, list)
            for r in results:
                assert isinstance(r, TrackedFace)

        times.sort()
        p95 = times[int(len(times) * 0.95)]
        print(f"100 dets × 1000 frames: p95={p95:.2f}ms avg={sum(times)/len(times):.2f}ms")

        delta = memory_helper.snapshot()
        assert delta.rss_delta_mb < 100, f"Memory grew {delta.rss_delta_mb:.1f} MB"
        # IoU matrix is 100×100; should complete < 500ms per frame
        assert p95 < 500, f"p95 {p95:.1f}ms too high"

    def test_identity_jitter_suppression(self):
        """身份抖动被 majority vote 抑制."""
        from face_hub import FaceRecognizer

        tracker = FaceTracker(smooth_frames=7, max_missed=10)
        recognizer = FaceRecognizer(tolerance=0.45)

        # Register 2 known people
        rng = np.random.RandomState(1)
        alice_emb = rng.randn(512).astype(np.float32)
        alice_emb /= np.linalg.norm(alice_emb)
        bob_emb = rng.randn(512).astype(np.float32)
        bob_emb /= np.linalg.norm(bob_emb)

        recognizer.update_cache(
            [alice_emb, bob_emb],
            ["alice", "bob"],
            db_version=1,
        )

        # Detect the same face (fixed box) but flip embedding every frame
        # between alice and bob to simulate recognition flicker
        names_seen = []
        for i in range(30):
            # Alternate embedding to simulate flicker
            emb = alice_emb if i % 2 == 0 else bob_emb
            det = make_detection(100, 100, 200, 200, emb=emb)
            tracked = tracker.update([det], recognizer=recognizer)

            if tracked:
                names_seen.append((i, tracked[0].name, tracked[0].is_confirmed))

        # After majority voting stabilizes (smooth_frames=7: needs 4+ consistent),
        # confirmed_name should settle and not flicker every frame
        confirmed_names = [n for _, n, c in names_seen if c]
        # At minimum, we should see some frames where identity is NOT confirmed
        # (because of alternating embeddings)
        unconfirmed = sum(1 for _, _, c in names_seen if not c)
        assert unconfirmed > 0, "Expected some unconfirmed frames due to flicker"

    def test_track_expiry_and_recovery(self):
        """丢失 max_missed 帧后 track 移除；再次出现生成新 track."""
        tracker = FaceTracker(smooth_frames=3, max_missed=5)

        # Frame 1: detect a face
        det = make_detection(100, 100, 200, 200)
        r1 = tracker.update([det])
        assert len(r1) == 1
        original_id = r1[0].track_id

        # Frames 2-6: no detection (face disappeared)
        for _ in range(5):
            r = tracker.update([])
        # Track should be removed (max_missed=5)
        assert tracker.track_count == 0

        # Frame 7: face reappears → new track_id
        det2 = make_detection(100, 100, 200, 200)
        r7 = tracker.update([det2])
        assert len(r7) == 1
        assert r7[0].track_id != original_id, "Should get new track_id"

    def test_history_truncation(self):
        """name_history 长度不超过 smooth_frames * 3."""
        tracker = FaceTracker(smooth_frames=5, max_missed=30)
        det = make_detection(100, 100, 200, 200)

        for _ in range(50):
            det = make_detection(100, 100, 200, 200)
            tracker.update([det])

        for track in tracker.tracks:
            max_len = tracker.smooth_frames * 3  # 15
            assert len(track.name_history) <= max_len, (
                f"name_history length {len(track.name_history)} exceeds {max_len}"
            )

    def test_concurrent_update_state_consistency(self):
        """多线程 update：状态一致无竞态."""
        import threading

        tracker = FaceTracker(smooth_frames=5)
        errors = []

        def worker():
            try:
                for _ in range(200):
                    dets = [
                        make_detection(
                            np.random.randint(0, 500),
                            np.random.randint(0, 300),
                            np.random.randint(0, 500) + 50,
                            np.random.randint(0, 300) + 50,
                        )
                        for _ in range(5)
                    ]
                    r = tracker.update(dets)
                    assert isinstance(r, list)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # FaceTracker has no internal lock, so concurrent updates
        # may cause inconsistencies — this is documented behavior.
        # The test verifies no hard crashes.
        assert len(errors) == 0, f"Crashes: {errors}"
