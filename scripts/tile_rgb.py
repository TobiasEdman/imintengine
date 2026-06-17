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


# ── Colormaps for continuous aux channels (numpy-only, no matplotlib) ────────
# Anchor stops (position, (R, G, B)) sampled from the matplotlib colormaps the
# tile-inspection dashboard uses, so aux panels read the same without pulling
# matplotlib into the slim dashboard pod. A trailing "_r" reverses a map.
_CMAPS: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    "viridis": [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 144, 140)),
                (0.75, (93, 201, 99)), (1.0, (253, 231, 37))],
    "magma": [(0.0, (0, 0, 4)), (0.25, (80, 18, 123)), (0.5, (182, 54, 121)),
              (0.75, (252, 137, 97)), (1.0, (252, 253, 191))],
    "terrain": [(0.0, (51, 51, 153)), (0.15, (0, 153, 255)), (0.25, (0, 204, 102)),
                (0.5, (255, 255, 153)), (0.75, (128, 92, 84)), (1.0, (255, 255, 255))],
    "Blues": [(0.0, (247, 251, 255)), (0.5, (107, 174, 214)), (1.0, (8, 48, 107))],
    "RdYlGn": [(0.0, (165, 0, 38)), (0.25, (253, 174, 97)), (0.5, (255, 255, 191)),
               (0.75, (166, 217, 106)), (1.0, (0, 104, 55))],
}

_NODATA_RGB = (210, 210, 214)   # light gray for non-finite pixels (e.g. markfukt gaps)


def colormap(t01: np.ndarray, name: str) -> np.ndarray:
    """Map a ``[0, 1]`` array through a named colormap → ``(H, W, 3)`` uint8."""
    reverse = name.endswith("_r")
    stops = _CMAPS[name[:-2] if reverse else name]
    t = np.clip(t01.astype(np.float32), 0.0, 1.0)
    if reverse:
        t = 1.0 - t
    pos = np.array([s[0] for s in stops], np.float32)
    cols = np.array([s[1] for s in stops], np.float32)
    out = np.empty(t.shape + (3,), np.uint8)
    for c in range(3):
        out[..., c] = np.interp(t, pos, cols[:, c]).astype(np.uint8)
    return out


def aux_rgb(arr: np.ndarray, name: str) -> np.ndarray:
    """Colormap a continuous aux raster: 2-98% percentile-normalise the finite
    values, map through ``name``, and paint non-finite pixels as nodata gray."""
    a = np.asarray(arr, np.float32)
    finite = np.isfinite(a)
    if finite.any():
        lo, hi = np.percentile(a[finite], (2, 98))
        span = max(float(hi - lo), 1e-6)
        t = np.where(finite, np.clip((a - lo) / span, 0.0, 1.0), 0.0)
        rgb = colormap(t, name)
    else:
        rgb = np.zeros(a.shape + (3,), np.uint8)
    rgb[~finite] = _NODATA_RGB
    return rgb
