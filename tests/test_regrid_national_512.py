"""Pure-logic tests for scripts/regrid_national_512.py (no network / no PU).

Covers the parts that are easy to get subtly wrong:
  * inter-frame coreg lands every frame on the *composite centroid* of frame
    positions — so the stack ends mutually co-registered, and the result is
    independent of which frame is the measurement origin (``ref_idx``) rather
    than inheriting one "clearest" frame's absolute ortho error;
  * the 520 -> 512 halo crop is centred;
  * clearest-frame selection prefers valid + sharp;
  * fresh assembly produces the exact npz key/shape contract;
  * centre recovery prefers easting/northing, falls back to bbox midpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import regrid_national_512 as rg  # noqa: E402
from imint.coregistration import subpixel_shift  # noqa: E402


def _smooth_field(seed: int, size: int = 520) -> np.ndarray:
    """Band-limited, non-periodic positive field — good phase-correlation target."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((size, size)).astype(np.float32), sigma=3.0)
    return (f - f.min() + 0.05).astype(np.float32)


def _cube(field: np.ndarray) -> np.ndarray:
    """Broadcast one (H,W) field into a (12, H, W) all-band cube."""
    return np.repeat(field[None], len(rg.ALL_BANDS), axis=0).astype(np.float32)


class TestCropHalo:
    def test_shape_and_centred(self):
        arr = np.zeros((len(rg.ALL_BANDS), rg.HALO_PX, rg.HALO_PX), np.float32)
        arr[:, rg.HALO_PX // 2, rg.HALO_PX // 2] = 7.0   # mark the 520-centre
        out = rg.crop_halo(arr)
        assert out.shape == (len(rg.ALL_BANDS), rg.CANON_PX, rg.CANON_PX)
        # 520-centre (260) maps to 512-centre (256): CROP=4 px per side.
        assert out[0, rg.CANON_PX // 2, rg.CANON_PX // 2] == 7.0
        assert rg.CROP == 4


class TestClearestFrame:
    def test_prefers_more_valid_pixels(self):
        full = _cube(_smooth_field(1))
        half = _cube(_smooth_field(2))
        half[:, : rg.HALO_PX // 2, :] = 0.0   # zero out half → lower valid frac
        assert rg.clearest_frame_idx({0: half, 1: full}) == 1

    def test_tie_break_on_sharpness(self):
        sharp = _cube(_smooth_field(3))
        flat = _cube(np.full((rg.HALO_PX, rg.HALO_PX), 0.2, np.float32))
        # both fully valid; sharp has higher B04 std → wins the tie
        assert rg.clearest_frame_idx({0: flat, 1: sharp}) == 1


class TestInterframeCoreg:
    """M2 centroid contract, tested on an unambiguous bright dot. A dot makes
    position measurable to sub-pixel via centre-of-mass, so a sign/axis error is
    caught loudly — it moves the dots APART instead of onto the shared centroid
    (the smooth-field residual test this replaces passed even on the inverted
    sign that amplified drift)."""

    _POS = {0: (0, 0), 1: (3, -2), 2: (-2, 3), 3: (1, 1)}   # frame positions vs f0

    @staticmethod
    def _dot_cube(r: int, c: int, size: int = 160) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        f = np.zeros((size, size), np.float64)
        f[r, c] = 100.0
        f = gaussian_filter(f, 2.0) + 0.01
        return np.repeat(f.astype(np.float32)[None], len(rg.ALL_BANDS), axis=0)

    @staticmethod
    def _com(plane: np.ndarray) -> np.ndarray:
        from scipy.ndimage import center_of_mass
        p = np.asarray(plane, np.float64)
        p = np.clip(p - p.mean(), 0.0, None)   # isolate the dot above background
        return np.array(center_of_mass(p))

    def _frames_at(self, positions: dict[int, tuple[int, int]]) -> dict[int, np.ndarray]:
        base = self._dot_cube(80, 80)
        return {
            i: np.stack([np.roll(np.roll(b, dr, 0), dc, 1) for b in base]).astype(np.float32)
            for i, (dr, dc) in positions.items()
        }

    def test_known_drift_removed_not_amplified(self):
        frames = self._frames_at(self._POS)
        b = rg._COREG_BAND
        gap_before = np.linalg.norm(self._com(frames[2][b]) - self._com(frames[1][b]))
        out = rg.coregister_interframe(frames, ref_idx=0)
        coms = np.array([self._com(out[i][b]) for i in sorted(out)])
        spread = coms.max(0) - coms.min(0)
        # every frame collapses onto the shared centroid → sub-pixel spread.
        assert np.all(spread < 0.5), f"frames not co-registered: spread={spread}"
        # and the gap SHRANK — the inverted sign would have grown it.
        assert float(np.linalg.norm(spread)) < 0.3 * float(gap_before)

    def test_independent_of_ref_idx(self):
        """Defining property of the centroid anchor: the stack lands in the same
        place regardless of which frame is the measurement origin."""
        b = rg._COREG_BAND
        out0 = rg.coregister_interframe(self._frames_at(self._POS), ref_idx=0)
        out2 = rg.coregister_interframe(self._frames_at(self._POS), ref_idx=2)
        for i in sorted(self._POS):
            d = float(np.linalg.norm(self._com(out0[i][b]) - self._com(out2[i][b])))
            assert d < 0.3, f"frame {i} depends on ref_idx: {d:.3f}px"

    def test_no_shift_returns_equivalent(self):
        base = self._dot_cube(80, 80)
        out = rg.coregister_interframe({0: base, 1: base.copy()}, ref_idx=0)
        # identical inputs → zero centroid → every frame returned untouched.
        np.testing.assert_allclose(out[0], base, atol=1e-4)
        np.testing.assert_allclose(out[1], base, atol=1e-4)


class TestAssembleFresh:
    def test_keys_shapes_and_missing_frame(self):
        n = 4
        h = w = rg.CANON_PX
        frames = {
            0: np.ones((len(rg.ALL_BANDS), h, w), np.float32) * 1.0,
            1: np.ones((len(rg.ALL_BANDS), h, w), np.float32) * 2.0,
            # frame 2 missing → zero-filled
            3: np.ones((len(rg.ALL_BANDS), h, w), np.float32) * 3.0,
        }
        dates = ["2022-09-10", "2022-06-01", "2022-07-15", "2022-08-20"]
        spectral, extras = rg.assemble_fresh(frames, dates, n)

        assert spectral.shape == (n * 6, h, w)
        assert spectral.dtype == np.float32
        # missing frame 2's 6-band block is all zeros
        assert not np.any(spectral[2 * 6:(2 + 1) * 6])
        # present frame 0's B02 (prithvi idx 0 == ALL_BANDS idx 0) is 1.0
        assert np.allclose(spectral[0], 1.0)

        assert extras["b08"].shape == (n, h, w)
        assert extras["rededge"].shape == (n * 3, h, w)
        assert extras["b01"].shape == (n, h, w)
        assert extras["b09"].shape == (n, h, w)
        for k in ("has_b08", "has_rededge", "has_b01", "has_b09"):
            assert int(extras[k]) == 1
        # missing frame → empty date string in every extra date array
        assert list(extras["b08_dates"])[2] == ""
        assert list(extras["b08_dates"])[0] == "2022-09-10"


class TestCentreOf:
    def test_prefers_easting_northing(self):
        assert rg.centre_of({"easting": 281280, "northing": 6471280}) == (281280, 6471280)

    def test_falls_back_to_bbox_midpoint(self):
        bb = {"bbox_3006": np.array([0, 0, 5120, 5120], np.int32)}
        assert rg.centre_of(bb) == (2560, 2560)

    def test_none_when_absent(self):
        assert rg.centre_of({"spectral": np.zeros((6, 4, 4))}) is None


class TestRegridOneTileIO:
    """End-to-end local path with the DES fetch mocked (no network / no PU)."""

    def test_full_contract(self, tmp_path, monkeypatch):
        def _fake_fetch(bbox, slot_dates, source):
            # orchestrator must fetch the 520 HALO on DES
            assert source == "des"
            assert round(bbox["east"] - bbox["west"]) == rg.HALO_PX * 10
            assert round(bbox["north"] - bbox["south"]) == rg.HALO_PX * 10
            return {
                fi: (np.full((len(rg.ALL_BANDS), rg.HALO_PX, rg.HALO_PX), 0.1, np.float32), d)
                for fi, d in slot_dates.items()
            }

        monkeypatch.setattr(rg, "fetch_tile_at_specific_dates", _fake_fetch)

        # Synthetic source tile on an OFF-lattice centre (3 m E, 7 m N off-grid),
        # carrying a stale old-grid label + aux to prove they are dropped.
        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "out"
        np.savez_compressed(
            src / "tile_x.npz",
            spectral=np.zeros((24, 512, 512), np.float32),
            dates=np.array(["2022-09-10", "2022-06-01", "2022-07-15", "2022-08-20"]),
            num_frames=np.int32(4),
            easting=np.int32(281283),
            northing=np.int32(6471287),
            year=np.int32(2022),
            lpis_year=np.int32(2022),
            label=np.ones((512, 512), np.uint8),     # stale old-grid → must drop
            dem=np.ones((512, 512), np.float32),      # stale old-grid aux → must drop
        )

        r = rg.regrid_one_tile(str(src / "tile_x.npz"), str(out), skip_existing=True,
                               debug_precoreg=True)
        assert r["status"] == "ok", r
        assert r["frames"] == 4

        d = dict(np.load(out / "tile_x.npz", allow_pickle=True))

        # national sentinel + canonical size
        assert int(d["national_grid"]) == 1
        assert int(d["tile_size_px"]) == rg.CANON_PX
        assert d["source"] == "des" or str(d["source"]) == "des"

        # centre snapped to the 10 m national lattice (281283->281280, 6471287->6471290)
        assert int(d["easting"]) == 281280
        assert int(d["northing"]) == 6471290
        bb = [int(x) for x in d["bbox_3006"]]
        assert all(v % 10 == 0 for v in bb)
        assert (bb[2] - bb[0]) == rg.CANON_PX * 10 == (bb[3] - bb[1])

        # spectral cropped to canonical; extras present
        assert d["spectral"].shape == (4 * 6, rg.CANON_PX, rg.CANON_PX)
        # debug_precoreg → pre-M2 (raw M1) spectral persisted at the same shape
        assert d["spectral_precoreg"].shape == (4 * 6, rg.CANON_PX, rg.CANON_PX)
        assert d["b08"].shape == (4, rg.CANON_PX, rg.CANON_PX)
        assert d["rededge"].shape == (4 * 3, rg.CANON_PX, rg.CANON_PX)
        assert int(d["has_b01"]) == 1 and int(d["has_b09"]) == 1
        assert list(d["temporal_mask"]) == [1, 1, 1, 1]

        # grid-independent identity carried; stale old-grid label + aux dropped
        assert int(d["year"]) == 2022 and int(d["lpis_year"]) == 2022
        assert "label" not in d
        assert "dem" not in d

    def test_skips_already_national(self, tmp_path):
        src = tmp_path / "src"; src.mkdir()
        np.savez_compressed(
            src / "done.npz",
            num_frames=np.int32(4), national_grid=np.int32(1),
            dates=np.array(["2022-09-10", "", "", ""]),
            easting=np.int32(281280), northing=np.int32(6471280),
        )
        r = rg.regrid_one_tile(str(src / "done.npz"), str(tmp_path / "out"))
        assert r["status"] == "skipped" and r["reason"] == "national_grid"
