"""Pure-logic tests for scripts/regrid_national_512.py (no network / no PU).

Covers the parts that are easy to get subtly wrong:
  * inter-frame coreg registers every mover onto the chosen reference frame
    (``ref_idx``, the clearest) by mutual information — the anchor is left
    untouched and the stack ends mutually co-registered on the anchor's grid;
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
from imint.coregistration import (  # noqa: E402
    _COREG_BAND,
    clearest_frame_idx,
    coregister_interframe,
    estimate_mi_offset,
    subpixel_shift,
)
from imint.training import fetch_spectral as fs  # noqa: E402
from imint.training.openeo_tile_graph import ALL_BANDS  # noqa: E402
from imint.training.tile_assemble import assemble_fresh, crop_halo  # noqa: E402


def _smooth_field(seed: int, size: int = 520) -> np.ndarray:
    """Band-limited, non-periodic positive field — good coregistration target."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((size, size)).astype(np.float32), sigma=3.0)
    return (f - f.min() + 0.05).astype(np.float32)


def _cube(field: np.ndarray) -> np.ndarray:
    """Broadcast one (H,W) field into a (12, H, W) all-band cube."""
    return np.repeat(field[None], len(ALL_BANDS), axis=0).astype(np.float32)


class TestCropHalo:
    def test_shape_and_centred(self):
        arr = np.zeros((len(ALL_BANDS), rg.HALO_PX, rg.HALO_PX), np.float32)
        arr[:, rg.HALO_PX // 2, rg.HALO_PX // 2] = 7.0   # mark the 520-centre
        out = crop_halo(arr, crop=rg.CROP, canon=rg.CANON_PX)
        assert out.shape == (len(ALL_BANDS), rg.CANON_PX, rg.CANON_PX)
        # 520-centre (260) maps to 512-centre (256): CROP=4 px per side.
        assert out[0, rg.CANON_PX // 2, rg.CANON_PX // 2] == 7.0
        assert rg.CROP == 4


class TestClearestFrame:
    def test_prefers_more_valid_pixels(self):
        full = _cube(_smooth_field(1))
        half = _cube(_smooth_field(2))
        half[:, : rg.HALO_PX // 2, :] = 0.0   # zero out half → lower valid frac
        assert clearest_frame_idx({0: half, 1: full}) == 1

    def test_tie_break_on_sharpness(self):
        sharp = _cube(_smooth_field(3))
        flat = _cube(np.full((rg.HALO_PX, rg.HALO_PX), 0.2, np.float32))
        # both fully valid; sharp has higher B04 std → wins the tie
        assert clearest_frame_idx({0: flat, 1: sharp}) == 1


class TestInterframeCoreg:
    """M2 = register each frame to the reference by mutual information. Tested on
    a structured scene whose CONTENT differs between frames (seasonal proxy): MI
    must recover the geometry and pull the movers onto the (fixed) reference,
    where intensity correlation would chase the content change. The reference
    frame itself must come back untouched."""

    _N = 200

    @staticmethod
    def _scene(seed: int, content: float, texture: bool, size: int) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(seed)
        lab = np.zeros((size, size), int)
        lab[28:104, 18:84] = 1; lab[112:172, 36:140] = 2; lab[42:152, 112:182] = 3
        road = np.zeros((size, size), np.float32); road[:, 90:93] = 1.0; road[78:81, :] = 1.0
        img = np.zeros((size, size), np.float32)
        for k, v in {0: 0.20, 1: 0.25 + content, 2: 0.22 - content, 3: 0.28 + content}.items():
            img[lab == k] = v
        img += 0.5 * road
        if texture:
            img += 0.18 * gaussian_filter(rng.standard_normal((size, size)).astype(np.float32), 1.5) * (lab > 0)
        img += 0.02 * rng.standard_normal((size, size))
        return gaussian_filter(img, 1.0).astype(np.float32)

    def _cube(self, b04: np.ndarray) -> np.ndarray:
        # band index _COREG_BAND carries the structured B04; coreg shifts all bands by it.
        return np.repeat(b04.astype(np.float32)[None], len(ALL_BANDS), axis=0)

    def test_aligns_movers_to_reference(self):
        size = self._N
        ref_b04 = self._scene(1, 0.0, False, size)                      # reference ("autumn")
        m1 = subpixel_shift(self._scene(2, 0.40, True, size), 1.2, -0.7)  # mover, different content
        m2 = subpixel_shift(self._scene(3, 0.35, True, size), -0.9, 1.0)
        frames = {0: self._cube(ref_b04), 1: self._cube(m1), 2: self._cube(m2)}
        out, shifts = coregister_interframe(frames, ref_idx=0)
        b = _COREG_BAND
        # reference is the fixed anchor — untouched, zero reported shift
        assert np.array_equal(out[0], frames[0])
        assert shifts[0] == (0.0, 0.0)
        # movers report a real applied shift
        assert all(abs(shifts[i][0]) + abs(shifts[i][1]) > 0 for i in (1, 2))
        for i in (1, 2):
            # each mover now sits on the reference → residual offset to ref ~0
            dy, dx = estimate_mi_offset(out[i][b], ref_b04, search_px=4.0)
            assert float(np.hypot(dy, dx)) < 0.5, f"frame {i} not aligned to ref: ({dy:.2f},{dx:.2f})"
            assert not np.array_equal(out[i], frames[i])               # it was actually shifted

    def test_no_shift_returns_equivalent(self):
        ref = self._cube(self._scene(1, 0.0, False, self._N))
        out, shifts = coregister_interframe({0: ref, 1: ref.copy()}, ref_idx=0)
        # identical content → MI optimum at 0 → both frames returned untouched.
        np.testing.assert_allclose(out[0], ref, atol=1e-4)
        np.testing.assert_allclose(out[1], ref, atol=1e-4)
        assert shifts[0] == (0.0, 0.0) and shifts[1] == (0.0, 0.0)


class TestAssembleFresh:
    def test_keys_shapes_and_missing_frame(self):
        n = 4
        h = w = rg.CANON_PX
        frames = {
            0: np.ones((len(ALL_BANDS), h, w), np.float32) * 1.0,
            1: np.ones((len(ALL_BANDS), h, w), np.float32) * 2.0,
            # frame 2 missing → zero-filled
            3: np.ones((len(ALL_BANDS), h, w), np.float32) * 3.0,
        }
        dates = ["2022-09-10", "2022-06-01", "2022-07-15", "2022-08-20"]
        spectral, extras = assemble_fresh(frames, dates, n, canon=rg.CANON_PX)

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


class TestRegridOneTileShim:
    """regrid_one_tile delegates to fetch_tile_spectral, then persists the result
    with the regrid-specific national_grid flag + grid-independent carry-keys.
    The entry is mocked (its correctness lives in test_fetch_tile_spectral) so this
    isolates the shim's read → call → persist responsibility."""

    def test_persists_entry_result_with_regrid_fields(self, tmp_path, monkeypatch):
        src = tmp_path / "tile_x.npz"
        np.savez(
            src,
            num_frames=np.int32(4),
            dates=np.array(["2021-09-10", "2022-06-01", "", "2022-08-20"]),
            easting=np.int32(500000), northing=np.int32(6500000),
            year=np.int32(2022), lpis_year=np.int32(2022), tessera_year=np.int32(2022),
            national_grid=np.int32(0),
        )

        captured = {}

        def _fake_entry(centre, *, tile, dates, n_frames, backend, halo_px,
                        coregister, return_precoreg):
            captured.update(centre=centre, dates=dates, n_frames=n_frames, backend=backend,
                            halo_px=halo_px, coregister=coregister,
                            return_precoreg=return_precoreg, tile_size=tile.size_px)
            return {
                "spectral": np.zeros((24, 4, 4), np.float32),
                "temporal_mask": np.array([1, 1, 0, 1], np.uint8),
                "dates": np.array(["2021-09-10", "2022-06-01", "", "2022-08-20"]),
                "easting": np.int32(500000), "northing": np.int32(6500000),
                "coreg_ref_frame": np.int32(0), "coreg_m2": np.int32(1),
                "coreg_anchor_valid_frac": np.float32(0.99),
            }

        monkeypatch.setattr(rg, "fetch_tile_spectral", _fake_entry)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        r = rg.regrid_one_tile(str(src), str(out_dir), skip_existing=True, debug_precoreg=False)

        assert r["status"] == "ok"
        # Entry called with the source centre + stored dates + DES + canonical geometry.
        assert captured["centre"] == (500000, 6500000)
        assert captured["dates"] == {0: "2021-09-10", 1: "2022-06-01", 3: "2022-08-20"}
        assert captured["n_frames"] == 4 and captured["backend"] == "des"
        assert captured["tile_size"] == rg.CANON_PX
        assert captured["halo_px"] == rg.HALO_PX - rg.CANON_PX
        assert captured["return_precoreg"] is False
        # Persisted: entry result (incl. quality flag) + national_grid + carry-keys.
        saved = dict(np.load(out_dir / "tile_x.npz", allow_pickle=True))
        assert int(saved["national_grid"]) == 1
        assert int(saved["coreg_m2"]) == 1
        assert int(saved["year"]) == 2022
        assert int(saved["lpis_year"]) == 2022 and int(saved["tessera_year"]) == 2022
        assert list(saved["temporal_mask"]) == [1, 1, 0, 1]

    def test_skips_existing_national_grid(self, tmp_path):
        src = tmp_path / "t.npz"
        np.savez(src, national_grid=np.int32(1), num_frames=np.int32(4))
        r = rg.regrid_one_tile(str(src), str(tmp_path), skip_existing=True)
        assert r["status"] == "skipped"


class TestRegridOneTileIO:
    """End-to-end local path with the DES fetch mocked (no network / no PU)."""

    def test_full_contract(self, tmp_path, monkeypatch):
        def _fake_fetch(bbox, slot_dates, source, with_scl=False):
            # orchestrator must fetch the 520 HALO on DES
            assert source == "des"
            assert round(bbox["east"] - bbox["west"]) == rg.HALO_PX * 10
            assert round(bbox["north"] - bbox["south"]) == rg.HALO_PX * 10
            frame = np.full((len(ALL_BANDS), rg.HALO_PX, rg.HALO_PX), 0.1, np.float32)
            if with_scl:
                scl = np.full((rg.HALO_PX, rg.HALO_PX), 4, np.uint8)
                return {fi: (frame, scl, d) for fi, d in slot_dates.items()}
            return {fi: (frame, d) for fi, d in slot_dates.items()}

        # The shim delegates to fetch_tile_spectral, which calls the tile-graph
        # inside imint.training.fetch_spectral — mock it there.
        monkeypatch.setattr(fs, "fetch_tile_at_specific_dates", _fake_fetch)

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
