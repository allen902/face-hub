"""
Shared pytest fixtures for FaceHub tests.
"""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_encoding():
    """Return a normalized 512-dim fake encoding."""
    vec = np.random.randn(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def sample_frame():
    """Return a dummy BGR frame (480x640)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def temp_db_paths():
    """Create temporary database paths, cleanup after test."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_db.json")
    enc_path = os.path.join(tmpdir, "test_enc.npy")
    yield db_path, enc_path
    # Cleanup (including any legacy .pkl that migration may have created)
    for ext in [".npy", ".pkl"]:
        p = os.path.join(tmpdir, f"test_enc{ext}")
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists(db_path):
        os.remove(db_path)
    os.rmdir(tmpdir)
