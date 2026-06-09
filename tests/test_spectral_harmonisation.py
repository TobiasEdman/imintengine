"""Grid-snap + subpixel-coreg harmonisation — pure-logic regression tests.

Proves the Sentinel-2 spectral-fetch alignment wiring without any network
call, so it is deterministic (seeded RNG) and CI-safe:

  * ``_snap_to_target_grid`` 513->512 byte-exact when the source origin
    already matches the target grid (the common case).
  * ``_snap_to_target_grid`` with a fractional origin offset -> 512x512 with
    the Fourier sub-pixel correction applied.
  * inter-frame coreg (``fill._coreg_to_reference``) drives a known sub-pixel
    residual toward zero — the estimator is biased (~60% recovery per pass),
    so we assert residual *reduction* and direction, not exact recovery.
  * ``assemble_bands`` keeps clean frames byte-identical, replaces corrupt
    frames with fresh Prithvi, and always adds B01/B09/B08/red-edge extras.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

from imint.coregistration import estimate_subpixel_offset, subpixel_shift
from imint.fetch import _snap_to_target_grid

# scripts/ is not a package — load the fill module from its file path.
_FILL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fill_tiles_l2a.py"
_spec = importlib.util.spec_from_file_location("fill_tiles_l2a", _FILL_PATH)
fill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fill)

# A SWEREF99 TM 512px tile on a 10 m grid, origin already 10 m-aligned.
_TW, _TN, _PX, _SZ = 500000.0, 6500000.0, 10, 512
_TARGET_BOUNDS = {
    "west": _TW, "south": _TN - _SZ * _PX, "east": _TW + _SZ * _PX, "north": _TN,
}
_SRC_CRS = CRS.from_epsg(3006)


def _raw513():
    """A deterministic raw (12, 513, 513) slot — the openEO over-extent shape."""
    return np.random.default_rng(0).random((12, 513, 513)).astype(np.float32)


class TestSnapToTargetGrid:
    """imint.fetch._snap_to_target_grid — per-scene grid snap to bbox extent."""

    def test_513_to_512_byte_exact_when_origin_matches(self):
        """Origin == target grid → pure integer truncation, byte-for-byte."""
        raw = _raw513()
        src_tf = from_origin(_TW, _TN, _PX, _PX)
        out, _ = _snap_to_target_grid(
            raw, src_tf, _SRC_CRS, _TARGET_BOUNDS, pixel_size=_PX
        )
        assert out.shape == (12, 512, 512)
        assert np.array_equal(out, raw[:, :512, :512])

    def test_fractional_offset_applies_subpixel(self):
        """Sub-pixel origin offset → 512x512, finite, and NOT a plain crop."""
        raw = _raw513()
        src_tf_frac = from_origin(_TW + 3.7, _TN, _PX, _PX)  # dx = +0.37 px
        out, _ = _snap_to_target_grid(
            raw, src_tf_frac, _SRC_CRS, _TARGET_BOUNDS, pixel_size=_PX
        )
        assert out.shape == (12, 512, 512)
        assert np.all(np.isfinite(out))
        assert not np.array_equal(out, raw[:, :512, :512])


def test_coreg_reduces_subpixel_residual():
    """Inter-frame coreg drives a known sub-pixel shift toward zero.

    Uses broadband, band-limited, PERIODIC texture (``mode="wrap"``) — it
    mimics the rich multi-scale content of real S2 imagery that phase
    correlation needs to localise; a narrow-band sinusoid would not.
    """
    rng = np.random.default_rng(0)
    h = w = 256
    base = rng.standard_normal((h, w)).astype(np.float32)
    p = gaussian_filter(base, sigma=1.2, mode="wrap").astype(np.float32)
    p = (p - p.min()) / (p.max() - p.min())  # -> [0,1] periodic texture

    # Existing reference frame (H, W, 6); B04 at prithvi index 2 == p.
    ref6 = np.stack(
        [p * 0.9, p * 0.95, p, p * 1.05, p * 1.1, p * 1.15], axis=-1
    ).astype(np.float32)

    dy0, dx0 = 0.30, -0.40
    p_shift = subpixel_shift(p.astype(np.float64), dy0, dx0).astype(np.float32)
    # Fresh (12, H, W); B04 at ALL_BANDS index 2 == shifted p.
    arr12 = np.repeat(p_shift[None, :, :], 12, axis=0).copy()
    arr12[2] = p_shift

    pre_dy, pre_dx = estimate_subpixel_offset(p, arr12[2])
    aligned = fill._coreg_to_reference(arr12, ref6)
    post_dy, post_dx = estimate_subpixel_offset(p, aligned[2])

    assert aligned.shape == (12, 256, 256)
    # Pre-coreg: offset detected in the correct direction and non-trivial.
    assert pre_dy * dy0 > 0 and pre_dx * dx0 > 0
    assert abs(pre_dy) > 0.1 and abs(pre_dx) > 0.1
    # Post-coreg: residual at least halved and well under a tenth of a pixel.
    assert abs(post_dy) < abs(pre_dy) * 0.5 and abs(post_dx) < abs(pre_dx) * 0.5
    assert abs(post_dy) < 0.1 and abs(post_dx) < 0.1


class TestAssembleBands:
    """fill.assemble_bands — keep-if-clean spectral + always-add L2A extras."""

    @staticmethod
    def _run():
        """Frame 0 CLEAN (minpos < 0.095), frame 1 CORRUPT (minpos >= 0.095)."""
        rng = np.random.default_rng(0)
        h = w = 8
        f0 = rng.uniform(0.01, 0.30, (6, h, w)).astype(np.float32)
        f0[0, 0, 0] = 0.02  # force min-positive below the corrupt floor
        f1 = rng.uniform(0.12, 0.40, (6, h, w)).astype(np.float32)  # minpos >= .095
        spectral = np.concatenate([f0, f1], axis=0)  # (12, h, w)
        fresh = {
            0: rng.uniform(0.01, 0.30, (12, h, w)).astype(np.float32),
            1: rng.uniform(0.01, 0.30, (12, h, w)).astype(np.float32),
        }
        dates = ["2021-09-15", "2022-06-01"]
        spec_out, extras, stats = fill.assemble_bands(
            spectral, fresh, dates, 2, h, w, None, None, None, None
        )
        return f0, fresh, h, w, spec_out, extras, stats

    def test_clean_frame_kept_byte_identical(self):
        f0, _, _, _, spec_out, _, _ = self._run()
        assert np.array_equal(spec_out[0:6], f0)

    def test_corrupt_frame_replaced_with_fresh_prithvi(self):
        _, fresh, _, _, spec_out, _, stats = self._run()
        assert np.array_equal(spec_out[6:12], fresh[1][fill._I_PRITHVI])
        assert stats["spec_fixed"] == 1

    def test_l2a_extras_added_with_correct_shape(self):
        _, _, h, w, _, extras, _ = self._run()
        assert extras["b01"].shape == (2, h, w)
        assert extras["b09"].shape == (2, h, w)
        assert extras["b08"].shape == (2, h, w)
        assert extras["rededge"].shape == (6, h, w)

    def test_b08_taken_from_fresh_fetch(self):
        _, fresh, _, _, _, extras, _ = self._run()
        assert np.array_equal(extras["b08"][0], fresh[0][fill._I_B08])
