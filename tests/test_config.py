"""Test config constants."""

from face_hub import DEFAULT_SETTINGS, get_default_settings


class TestDefaultSettings:
    def test_required_keys(self):
        required = [
            "device", "confidence", "tolerance", "cam_width", "cam_height",
            "cam_fps", "proc_fps", "det_size", "track_smooth",
            "min_face_size", "quality_filter",
        ]
        for key in required:
            assert key in DEFAULT_SETTINGS, f"Missing key: {key}"

    def test_deepcopy_isolation(self):
        a = get_default_settings()
        b = get_default_settings()
        a["confidence"] = 0.99
        assert b["confidence"] == 0.50
        assert a is not b
