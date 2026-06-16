"""Canonical Sentinel-2 tile → RGB helpers (numpy + PIL only, no heavy deps).

The per-frame band order is Prithvi's ``[B02, B03, B04, B8A, B11, B12]`` (blue,
green, red, NIR-narrow, SWIR1, SWIR2), so true-colour RGB is bands ``[2, 1, 0]``
(B04/B03/B02). ``stretch_rgb`` applies the repo-standard per-channel 2-98%
percentile stretch used everywhere tiles are visualised for display (raw
reflectance goes to models; stretch is display-only — see
``memory/feedback_raw_reflectance_to_models``).

Extracted here so the cluster campaign dashboard (running on a minimal
python:3.11-slim pod with a sparse ``scripts/`` checkout) can reuse the exact
band order + stretch without pulling matplotlib / the ``imint`` package that
``render_tile_inspection_dashboard`` imports.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

N_BANDS = 6  # prithvi per-frame: [B02, B03, B04, B8A, B11, B12]


def png_b64(rgb_u8: np.ndarray) -> str:
    """Encode an ``(H, W, 3)`` uint8 array as a base64 PNG data-URI payload."""
    buf = io.BytesIO()
    Image.fromarray(rgb_u8, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def stretch_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-channel 2-98% percentile stretch of three bands → ``(H, W, 3)`` uint8."""
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], (2, 98))
        span = max(float(hi - lo), 1e-6)
        rgb[..., c] = np.clip((rgb[..., c] - lo) / span, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def frame_rgb(spectral: np.ndarray, fi: int) -> np.ndarray:
    """True-colour RGB for temporal frame ``fi`` of a ``(T*6, H, W)`` cube.

    ``spectral[fi*6 + {2,1,0}]`` = B04/B03/B02 → red/green/blue.
    """
    base = fi * N_BANDS
    return stretch_rgb(spectral[base + 2], spectral[base + 1], spectral[base + 0])
