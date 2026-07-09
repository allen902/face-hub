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
    enc_path = os.path.join(tmpdir, "test_enc.pkl")
    yield db_path, enc_path
    # Cleanup
    for p in [db_path, enc_path]:
        if os.path.exists(p):
            os.remove(p)
    os.rmdir(tmpdir)
