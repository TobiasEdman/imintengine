"""
Shared test fixtures for IMINT Engine tests.
"""
from __future__ import annotations

import os
import sys
import shutil
import pytest
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def rgb_uniform():
    """A uniform 64x64 RGB image (all 0.5) — no variance, no edges."""
    return np.full((64, 64, 3), 0.5, dtype=np.float32)


@pytest.fixture
def rgb_random(seed=42):
    """A random 64x64 RGB image."""
    rng = np.random.RandomState(42)
    return rng.rand(64, 64, 3).astype(np.float32)


@pytest.fixture
def bands_known():
    """
    Sentinel-2 bands with known spectral properties.

    Top half: high NIR (B08=0.8), low red (B04=0.1) → high NDVI (vegetation)
    Bottom half: low NIR (B08=0.1), high red (B04=0.8) → low NDVI (bare soil)
    """
    h, w = 64, 64
    b04 = np.full((h, w), 0.1, dtype=np.float32)
    b04[h // 2:, :] = 0.8

    b08 = np.full((h, w), 0.8, dtype=np.float32)
    b08[h // 2:, :] = 0.1

    return {
        "B02": np.full((h, w), 0.2, dtype=np.float32),
        "B03": np.full((h, w), 0.3, dtype=np.float32),
        "B04": b04,
        "B08": b08,
        "B11": np.full((h, w), 0.15, dtype=np.float32),
    }


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory that is cleaned up after the test."""
    out = tmp_path / "outputs" / "2022-06-15"
    out.mkdir(parents=True)
    return str(out)


@pytest.fixture
def coords():
    """Standard test bounding box."""
    return {"west": 14.5, "south": 56.0, "east": 15.5, "north": 57.0}
