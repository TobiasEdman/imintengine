"""tile_rgb — shared Sentinel-2 RGB + aux-colormap helpers (numpy + PIL only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tile_rgb as tr  # noqa: E402


def test_frame_rgb_true_colour_band_order():
    # Distinct constant bands so we can read off which index lands in R/G/B.
    cube = np.zeros((24, 4, 4), np.float32)
    cube[0] = 0.1   # B02 (blue)
    cube[1] = 0.2   # B03 (green)
    cube[2] = 0.3   # B04 (red)
    rgb = tr.frame_rgb(cube, 0)
    assert rgb.shape == (4, 4, 3) and rgb.dtype == np.uint8
    # constant bands → 2-98% span collapses, but the channel mapping (R=B04,
    # G=B03, B=B02) is what we assert via a gradient instead:
    cube[2] = np.linspace(0, 1, 16).reshape(4, 4)        # red ramp
    rgb = tr.frame_rgb(cube, 0)
    assert rgb[0, 0, 0] < rgb[-1, -1, 0]                  # red increases along ramp


def test_png_b64_is_png():
    b64 = tr.png_b64(np.zeros((3, 3, 3), np.uint8))
    assert b64.startswith("iVBOR")                       # PNG magic, base64


def test_colormap_endpoints_and_reverse():
    t = np.array([[0.0, 1.0]], np.float32)
    fwd = tr.colormap(t, "viridis")
    assert tuple(fwd[0, 0]) == (68, 1, 84)               # viridis low
    assert tuple(fwd[0, 1]) == (253, 231, 37)            # viridis high
    rev = tr.colormap(t, "viridis_r")
    assert tuple(rev[0, 0]) == (253, 231, 37)            # reversed → high at 0
    assert tuple(rev[0, 1]) == (68, 1, 84)


def test_aux_rgb_paints_nodata_gray():
    a = np.ones((8, 8), np.float32)
    a[:, :4] = np.nan                                    # half nodata
    rgb = tr.aux_rgb(a, "Blues")
    assert rgb.shape == (8, 8, 3)
    assert (rgb[:, :4] == np.array(tr._NODATA_RGB)).all()   # nodata → gray
    assert not (rgb[:, 4:] == np.array(tr._NODATA_RGB)).all()  # data → colormapped


def test_aux_rgb_all_nodata_is_safe():
    rgb = tr.aux_rgb(np.full((4, 4), np.nan, np.float32), "viridis")
    assert (rgb == np.array(tr._NODATA_RGB)).all()
