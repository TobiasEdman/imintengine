"""Tests for the through-entry rewire of ``scripts/fetch_unified_tiles.py``.

Covers the two pieces 4b-3 step 1 introduces:
  * ``_split_entry_result`` — the pure 5-slot → (4 temporal frames + extras +
    2016 background) split, including the ``has_*`` recompute over the 4-frame
    slice and background-present/absent branching;
  * ``fetch_tile`` — the orchestrator now drives ``select_slot_dates`` →
    ``fetch_tile_spectral(n_frames=5)`` → split → .npz. The network/PU calls
    (entry, NMD label, aux, date selection) are monkeypatched, so the test
    asserts the *layout contract*, not real fetching.

No network, no PU. ``scripts`` is not a package → loaded via ``sys.path`` like
``tests/test_regrid_national_512.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fetch_unified_tiles as fut  # noqa: E402
from imint.training.tile_config import TileConfig  # noqa: E402

_VPP = [(150, 170), (180, 200), (210, 230)]   # any non-None → skips on-demand VPP


def _fake_res(h: int = 8, w: int = 8, *, bg: bool = True,
              b08_only_bg: bool = False) -> dict:
    """A fake 5-slot ``fetch_tile_spectral`` result with slot-distinct values.

    Slot ``fi``'s 6-band spectral block is filled with constant ``fi + 1`` so the
    split's slot→channel mapping is verifiable. ``b08_only_bg`` puts B08 signal in
    slot 4 ONLY, so ``has_b08`` recomputed over slots 0-3 must come out 0 (while
    the entry's own all-5-slot ``has_b08`` stays 1).
    """
    nb, nf = 6, 5
    spectral = np.zeros((nb * nf, h, w), np.float32)
    for fi in range(nf):
        spectral[fi * nb:(fi + 1) * nb] = fi + 1
    mask = np.ones(nf, np.uint8)
    dates = ["2021-09-15", "2022-05-01", "2022-06-15", "2022-07-20", "2016-07-15"]
    doy = [258, 121, 166, 201, 197]
    if not bg:
        mask[4] = 0
        spectral[4 * nb:5 * nb] = 0.0
        dates[4] = ""
        doy[4] = 0

    b08 = np.ones((nf, h, w), np.float32)
    if b08_only_bg:
        b08[:4] = 0.0

    return {
        "spectral": spectral,
        "temporal_mask": mask,
        "doy": np.array(doy, np.int32),
        "dates": np.array(dates),
        "multitemporal": np.int32(1),
        "num_frames": np.int32(nf),
        "num_bands": np.int32(nb),
        "bbox_3006": np.array([400000, 6400000, 405120, 6405120], np.int32),
        "easting": np.int32(402560),
        "northing": np.int32(6402560),
        "tile_size_px": np.int32(512),
        "source": "des",
        "coreg_ref_frame": np.int32(2),
        "coreg_m2": np.int32(1),
        "coreg_n_aligned": np.int32(3),
        "coreg_max_shift": np.float32(1.5),
        "coreg_anchor_valid_frac": np.float32(0.97),
        "coreg_shifts": np.arange(nf * 2, dtype=np.float32).reshape(nf, 2),
        "b08": b08, "b08_dates": np.array(dates), "has_b08": np.int32(1),
        "rededge": np.ones((nf * 3, h, w), np.float32),
        "rededge_dates": np.array(dates), "has_rededge": np.int32(1),
        "b01": np.full((nf, h, w), 2.0, np.float32),
        "b01_dates": np.array(dates), "has_b01": np.int32(1),
        "b09": np.full((nf, h, w), 3.0, np.float32),
        "b09_dates": np.array(dates), "has_b09": np.int32(1),
    }


class TestSplitEntryResult:
    def test_core_shapes_and_slot_mapping(self):
        core, _, _ = fut._split_entry_result(_fake_res())
        assert core["spectral"].shape == (24, 8, 8)
        for fi in range(4):                       # slots 0-3 → values 1,2,3,4
            assert np.all(core["spectral"][fi * 6:(fi + 1) * 6] == fi + 1)
        assert core["temporal_mask"].shape == (4,)
        assert core["doy"].shape == (4,)
        assert core["dates"].shape == (4,)
        assert core["coreg_shifts"].shape == (4, 2)
        assert int(core["num_frames"]) == 4
        assert int(core["num_bands"]) == 6

    def test_coreg_provenance_passthrough(self):
        res = _fake_res()
        core, _, _ = fut._split_entry_result(res)
        assert int(core["coreg_ref_frame"]) == 2
        assert int(core["coreg_m2"]) == 1
        assert int(core["coreg_n_aligned"]) == 3
        assert float(core["coreg_max_shift"]) == pytest.approx(1.5)
        assert float(core["coreg_anchor_valid_frac"]) == pytest.approx(0.97)
        np.testing.assert_array_equal(core["bbox_3006"], res["bbox_3006"])
        np.testing.assert_array_equal(core["coreg_shifts"], res["coreg_shifts"][:4])

    def test_extras_sliced_to_four_frames(self):
        _, extras, _ = fut._split_entry_result(_fake_res())
        assert extras["b08"].shape == (4, 8, 8)
        assert extras["rededge"].shape == (12, 8, 8)   # 4 frames * 3 bands
        assert extras["b01"].shape == (4, 8, 8)
        assert extras["b09"].shape == (4, 8, 8)
        assert extras["b08_dates"].shape == (4,)
        assert extras["rededge_dates"].shape == (4,)

    def test_has_flags_recomputed_over_slice(self):
        # B08 signal only in slot 4 → has_b08 over slots 0-3 must be 0, even
        # though the entry's own (all-5-slot) has_b08 is 1. Proves the split
        # RECOMPUTES has_* on the 4-frame slice rather than passing it through.
        res = _fake_res(b08_only_bg=True)
        assert int(res["has_b08"]) == 1
        _, extras, _ = fut._split_entry_result(res)
        assert int(extras["has_b08"]) == 0
        assert int(extras["has_b01"]) == 1            # b01 nonzero everywhere

    def test_background_present(self):
        _, _, bg = fut._split_entry_result(_fake_res(bg=True))
        assert bg is not None
        assert bg["frame_2016"].shape == (6, 8, 8)
        assert np.all(bg["frame_2016"] == 5)          # slot-4 block constant
        assert bg["frame_2016_date"].decode() == "2016-07-15"
        assert int(bg["frame_2016_year"]) == 2016
        assert int(bg["frame_2016_doy"]) == 197
        assert int(bg["has_frame_2016"]) == 1

    def test_background_absent(self):
        _, _, bg = fut._split_entry_result(_fake_res(bg=False))
        assert bg is None


class TestFetchTileThroughEntry:
    @staticmethod
    def _loc(tile: TileConfig, year: int | None = None) -> dict:
        bbox = tile.bbox_from_center(402560, 6402560)
        loc = {"name": "tile_test", "source": "lulc", "bbox_3006": bbox}
        if year is not None:
            loc["year"] = year
        return loc

    @staticmethod
    def _patch_common(monkeypatch, res: dict) -> None:
        monkeypatch.setattr(fut, "fetch_tile_spectral", lambda *a, **k: res)
        monkeypatch.setattr(fut, "fetch_nmd_label_local",
                            lambda bbox, tile: np.zeros((8, 8), np.uint8))
        monkeypatch.setattr(fut, "fetch_aux_channels",
                            lambda bbox, tile: {"dem": np.zeros((8, 8), np.float32)})

    def test_saves_split_layout(self, tmp_path, monkeypatch):
        tile = TileConfig(size_px=512)
        self._patch_common(monkeypatch, _fake_res())
        monkeypatch.setattr(
            fut, "select_slot_dates",
            lambda coords, **k: {0: "2021-09-15", 1: "2022-05-01",
                                 2: "2022-06-15", 3: "2022-07-20", 4: "2016-07-15"})

        # cloud_max + sources still accepted (main() passes them).
        r = fut.fetch_tile(self._loc(tile, year=2022), ["2022"], str(tmp_path),
                           tile, cloud_max=30.0, vpp_cache={"tile_test": _VPP},
                           sources=("des",))
        assert r == {"name": "tile_test", "status": "ok",
                     "valid_frames": 4, "has_bg": 1}

        d = dict(np.load(tmp_path / "tile_test.npz", allow_pickle=True))
        assert d["spectral"].shape == (24, 8, 8)
        assert int(d["num_frames"]) == 4 and int(d["num_bands"]) == 6
        assert d["temporal_mask"].shape == (4,)
        assert d["frame_2016"].shape == (6, 8, 8)
        assert int(d["frame_2016_year"]) == 2016
        assert int(d["has_frame_2016"]) == 1
        assert d["b08"].shape == (4, 8, 8)
        assert d["rededge"].shape == (12, 8, 8)
        assert int(d["coreg_m2"]) == 1
        assert "label" in d and "dem" in d
        assert str(d["source"]) == "lulc"
        # frame_2016_cloud_pct was a legacy-only key; the entry path drops it.
        assert "frame_2016_cloud_pct" not in d

    def test_des_only_no_background(self, tmp_path, monkeypatch):
        tile = TileConfig(size_px=512)
        self._patch_common(monkeypatch, _fake_res(bg=False))
        monkeypatch.setattr(
            fut, "select_slot_dates",
            lambda coords, **k: {0: "2021-09-15", 1: "2022-05-01",
                                 2: "2022-06-15", 3: "2022-07-20"})
        r = fut.fetch_tile(self._loc(tile, year=2022), ["2022"], str(tmp_path),
                           tile, vpp_cache={"tile_test": _VPP})
        assert r["status"] == "ok" and r["has_bg"] == 0
        d = dict(np.load(tmp_path / "tile_test.npz", allow_pickle=True))
        assert "frame_2016" not in d
        assert "has_frame_2016" not in d

    def test_skips_existing(self, tmp_path, monkeypatch):
        tile = TileConfig(size_px=512)
        np.savez_compressed(tmp_path / "tile_test.npz", dummy=np.zeros(1))

        def _boom(*a, **k):
            raise AssertionError("must not fetch when the tile already exists")
        monkeypatch.setattr(fut, "fetch_tile_spectral", _boom)

        r = fut.fetch_tile(self._loc(tile, year=2022), ["2022"], str(tmp_path),
                           tile, vpp_cache={"tile_test": _VPP})
        assert r["status"] == "skipped"

    def test_no_year_multi_year_fallback(self, tmp_path, monkeypatch):
        # No loc["year"] → select_slot_dates is looped over --years, each slot
        # filled by the FIRST year that yields a date (setdefault). 2018 fills
        # slots 0+4; 2019 fills the missing VPP slots 1-3; 2022 is never probed
        # because all 5 slots are full by then.
        tile = TileConfig(size_px=512)
        res = _fake_res()
        self._patch_common(monkeypatch, res)

        years_seen: list[int] = []

        def _sel(coords, *, tile_year, vpp_windows, **k):
            years_seen.append(tile_year)
            if tile_year == 2018:
                return {0: "2017-09-15", 4: "2016-07-15"}
            if tile_year == 2019:
                return {1: "2019-05-01", 2: "2019-06-15", 3: "2019-07-20",
                        4: "2016-07-15"}
            return {}
        monkeypatch.setattr(fut, "select_slot_dates", _sel)

        captured: dict = {}

        def _spy_entry(center, *, tile, dates, n_frames, backend):
            captured.update(center=center, dates=dict(dates),
                            n_frames=n_frames, backend=backend)
            return res
        monkeypatch.setattr(fut, "fetch_tile_spectral", _spy_entry)

        r = fut.fetch_tile(self._loc(tile, year=None),
                           ["2018", "2019", "2022"], str(tmp_path), tile,
                           vpp_cache={"tile_test": _VPP})
        assert r["status"] == "ok"
        assert captured["dates"] == {0: "2017-09-15", 1: "2019-05-01",
                                     2: "2019-06-15", 3: "2019-07-20",
                                     4: "2016-07-15"}
        assert captured["n_frames"] == 5
        assert captured["backend"] == "des"
        assert captured["center"] == (402560, 6402560)
        assert years_seen == [2018, 2019]            # 2022 not probed

    def test_no_clean_dates_fails(self, tmp_path, monkeypatch):
        tile = TileConfig(size_px=512)
        self._patch_common(monkeypatch, _fake_res())
        monkeypatch.setattr(fut, "select_slot_dates", lambda coords, **k: {})
        r = fut.fetch_tile(self._loc(tile, year=2022), ["2022"], str(tmp_path),
                           tile, vpp_cache={"tile_test": _VPP})
        assert r["status"] == "failed"
        assert r["reason"] == "no_scenes"
        assert not (tmp_path / "tile_test.npz").exists()

    def test_all_temporal_frames_empty_fails(self, tmp_path, monkeypatch):
        # Entry returned a result, but every temporal slot (0-3) is empty —
        # only the 2016 background fetched. The temporal_mask.sum()==0 guard
        # must fail the tile and write no npz (a bg-only tile is not trainable).
        tile = TileConfig(size_px=512)
        res = _fake_res()
        res["temporal_mask"] = np.array([0, 0, 0, 0, 1], np.uint8)
        self._patch_common(monkeypatch, res)
        monkeypatch.setattr(fut, "select_slot_dates",
                            lambda coords, **k: {4: "2016-07-15"})
        r = fut.fetch_tile(self._loc(tile, year=2022), ["2022"], str(tmp_path),
                           tile, vpp_cache={"tile_test": _VPP})
        assert r["status"] == "failed"
        assert r["reason"] == "no_scenes"
        assert not (tmp_path / "tile_test.npz").exists()


class TestRefetchTileThroughEntry:
    @staticmethod
    def _write_source(path, *, with_bg=True, extra=None):
        """A pre-existing tile .npz: stored dates + aux + stale derived fields."""
        src = {
            "dates": np.array(["2021-09-15", "2022-05-01",
                               "2022-06-15", "2022-07-20"]),
            "spectral": np.zeros((24, 8, 8), np.float32),     # stale
            "temporal_mask": np.ones(4, np.uint8),
            "doy": np.array([258, 121, 166, 201], np.int32),
            "multitemporal": np.int32(1),
            "num_frames": np.int32(4),
            "num_bands": np.int32(6),
            # non-spectral — MUST be preserved:
            "dem": np.full((8, 8), 42.0, np.float32),
            "nmd_label_raw": np.full((8, 8), 7, np.uint8),
            # spectral-derived stale — MUST be dropped/rebuilt:
            "frame_2016_cloud_pct": np.float32(0.5),
            "coreg_m2": np.int32(0),
            "b08": np.zeros((4, 8, 8), np.float32), "has_b08": np.int32(0),
            # rebuilt separately by build_labels → dropped:
            "label": np.zeros((8, 8), np.uint8),
            "source": "crop",
            "year": 2022,
        }
        if with_bg:
            src["frame_2016_date"] = np.bytes_("2016-07-15")
            src["frame_2016"] = np.zeros((6, 8, 8), np.float32)   # stale
        if extra:
            src.update(extra)
        np.savez_compressed(path, **src)

    @staticmethod
    def _loc(tile, existing_path):
        bbox = tile.bbox_from_center(402560, 6402560)
        return {"name": "tile_test", "source": "crop", "bbox_3006": bbox,
                "year": 2022, "_existing_path": str(existing_path),
                "_has_lpis": True}

    def _no_reselect(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("refetch must NOT re-select dates")
        monkeypatch.setattr(fut, "select_slot_dates", _boom)

    def test_reuses_stored_dates_no_reselect(self, tmp_path, monkeypatch):
        tile = TileConfig(size_px=512)
        src = tmp_path / "src.npz"
        self._write_source(src, with_bg=True)
        out = tmp_path / "out"
        out.mkdir()
        self._no_reselect(monkeypatch)

        captured = {}

        def _spy(center, *, tile, dates, n_frames, backend):
            captured.update(dates=dict(dates), n_frames=n_frames, backend=backend)
            return _fake_res()
        monkeypatch.setattr(fut, "fetch_tile_spectral", _spy)

        r = fut.refetch_tile(self._loc(tile, src), ["2022"], str(out), tile)
        assert r["status"] == "ok"
        # Stored dates reused verbatim — slots 0-3 from `dates`, slot 4 from
        # frame_2016_date. No re-selection (select_slot_dates would have raised).
        assert captured["dates"] == {0: "2021-09-15", 1: "2022-05-01",
                                     2: "2022-06-15", 3: "2022-07-20",
                                     4: "2016-07-15"}
        assert captured["n_frames"] == 5 and captured["backend"] == "des"

    def test_preserves_aux_rebuilds_spectral(self, tmp_path, monkeypatch):
        tile = TileConfig(size_px=512)
        src = tmp_path / "src.npz"
        self._write_source(src, with_bg=True)
        out = tmp_path / "out"
        out.mkdir()
        self._no_reselect(monkeypatch)
        monkeypatch.setattr(fut, "fetch_tile_spectral", lambda *a, **k: _fake_res())

        r = fut.refetch_tile(self._loc(tile, src), ["2022"], str(out), tile)
        assert r["status"] == "ok"
        d = dict(np.load(out / "tile_test.npz", allow_pickle=True))

        # Non-spectral preserved:
        assert np.all(d["dem"] == 42.0)
        assert np.all(d["nmd_label_raw"] == 7)
        # Spectral-derived rebuilt from the entry (not the stale source):
        assert d["spectral"].shape == (24, 8, 8)
        assert np.all(d["spectral"][:6] == 1)          # entry slot-0 value
        assert int(d["coreg_m2"]) == 1                 # was 0 in source
        assert int(d["has_b08"]) == 1                  # was 0 in source
        assert d["frame_2016"].shape == (6, 8, 8)
        assert int(d["frame_2016_year"]) == 2016
        # Stale spectral-derived dropped; label left to build_labels:
        assert "frame_2016_cloud_pct" not in d
        assert "label" not in d
        assert str(d["source"]) == "crop"

    def test_no_stored_dates_fails(self, tmp_path, monkeypatch):
        tile = TileConfig(size_px=512)
        src = tmp_path / "src.npz"
        # A source with neither `dates` nor `frame_2016_date`.
        np.savez_compressed(src, dem=np.zeros((8, 8), np.float32),
                            source="lulc")
        out = tmp_path / "out"
        out.mkdir()
        self._no_reselect(monkeypatch)

        def _boom_entry(*a, **k):
            raise AssertionError("must not fetch without stored dates")
        monkeypatch.setattr(fut, "fetch_tile_spectral", _boom_entry)

        loc = {"name": "tile_test", "source": "lulc",
               "bbox_3006": tile.bbox_from_center(402560, 6402560),
               "_existing_path": str(src)}
        r = fut.refetch_tile(loc, ["2022"], str(out), tile)
        assert r["status"] == "failed"
        assert r["reason"] == "no_stored_dates"
        assert not (out / "tile_test.npz").exists()

    def test_skips_existing_multitemporal_without_force(self, tmp_path, monkeypatch):
        tile = TileConfig(size_px=512)
        src = tmp_path / "src.npz"
        self._write_source(src)
        out = tmp_path / "out"
        out.mkdir()
        # An already-multitemporal output → skip unless force.
        np.savez_compressed(out / "tile_test.npz", multitemporal=np.int32(1),
                            num_frames=np.int32(4))

        def _boom_entry(*a, **k):
            raise AssertionError("must skip without force")
        monkeypatch.setattr(fut, "fetch_tile_spectral", _boom_entry)

        r = fut.refetch_tile(self._loc(tile, src), ["2022"], str(out), tile,
                             force=False)
        assert r["status"] == "skipped"

    def test_force_refetches_existing(self, tmp_path, monkeypatch):
        tile = TileConfig(size_px=512)
        src = tmp_path / "src.npz"
        self._write_source(src)
        out = tmp_path / "out"
        out.mkdir()
        np.savez_compressed(out / "tile_test.npz", multitemporal=np.int32(1),
                            num_frames=np.int32(4))
        self._no_reselect(monkeypatch)
        monkeypatch.setattr(fut, "fetch_tile_spectral", lambda *a, **k: _fake_res())

        r = fut.refetch_tile(self._loc(tile, src), ["2022"], str(out), tile,
                             force=True)
        assert r["status"] == "ok"
        assert r["valid_frames"] == 4

    def test_no_stored_background_omits_slot4(self, tmp_path, monkeypatch):
        tile = TileConfig(size_px=512)
        src = tmp_path / "src.npz"
        self._write_source(src, with_bg=False)        # no frame_2016_date
        out = tmp_path / "out"
        out.mkdir()
        self._no_reselect(monkeypatch)

        captured = {}

        def _spy(center, *, tile, dates, n_frames, backend):
            captured.update(dates=dict(dates))
            return _fake_res(bg=False)
        monkeypatch.setattr(fut, "fetch_tile_spectral", _spy)

        r = fut.refetch_tile(self._loc(tile, src), ["2022"], str(out), tile)
        assert r["status"] == "ok"
        assert captured["dates"] == {0: "2021-09-15", 1: "2022-05-01",
                                     2: "2022-06-15", 3: "2022-07-20"}
        assert 4 not in captured["dates"]
        d = dict(np.load(out / "tile_test.npz", allow_pickle=True))
        assert "frame_2016" not in d
