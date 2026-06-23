"""Contract tests for bbox/size_px consistency in the fetch chain.

Parametrized over multiple TileConfig sizes to catch any regression
where a hardcoded TILE_SIZE_M / TILE_SIZE_PX leaks back into the
pipeline.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from imint.training.tile_bbox import resolve_tile_bbox
from imint.training.tile_config import TileConfig


@pytest.mark.parametrize("size_px", [256, 512, 1024])
class TestResolveBboxSizes:
    def test_filename_gives_correct_extent(self, size_px):
        t = TileConfig(size_px=size_px)
        b = resolve_tile_bbox(name="tile_281280_6471280", tile=t, npz_data=None)
        assert b is not None
        assert (b["east"] - b["west"]) == size_px * 10
        assert (b["north"] - b["south"]) == size_px * 10

    def test_easting_northing_npz_gives_correct_extent(self, size_px):
        t = TileConfig(size_px=size_px)
        npz = {"easting": np.int32(500000), "northing": np.int32(6500000)}
        b = resolve_tile_bbox(name="unused_name", tile=t, npz_data=npz)
        assert b is not None
        assert (b["east"] - b["west"]) == size_px * 10
        # Assertion must pass — resolve always normalizes to current size
        t.assert_bbox_matches(b)

    def test_stale_bbox_array_is_renormalized(self, size_px):
        """Even if the .npz carries a stale bbox of wrong extent, the
        helper re-centers at the current TileConfig size."""
        t = TileConfig(size_px=size_px)
        # Fake a stale 256-era bbox (2560m wide) — shouldn't corrupt output
        npz = {
            "bbox_3006": np.array([100000, 6500000, 102560, 6502560], dtype=np.int32),
        }
        b = resolve_tile_bbox(name="unused_name", tile=t, npz_data=npz)
        assert b is not None
        assert (b["east"] - b["west"]) == size_px * 10, \
            f"size_px={size_px} → stale bbox was NOT renormalized"


@pytest.mark.parametrize("size_px", [256, 512])
class TestFetchS2SceneAssertion:
    """The assertion added to imint.training.cdse_s2.fetch_s2_scene must
    reject mismatched (bbox, size_px) before any HTTP call."""

    def test_correct_extent_passes_assertion(self, size_px):
        """We only verify the assertion layer — don't actually call CDSE.
        A mismatch should raise ValueError BEFORE _get_token() is reached."""
        from imint.training import cdse_s2

        t = TileConfig(size_px=size_px)
        # Build a correctly-sized bbox
        bbox = t.bbox_from_center(500000, 6500000)

        with patch.object(cdse_s2, "_get_token", side_effect=AssertionError("mocked")):
            # Assertion passes → _get_token called → our AssertionError raised
            with pytest.raises(AssertionError, match="mocked"):
                cdse_s2.fetch_s2_scene(
                    bbox["west"], bbox["south"], bbox["east"], bbox["north"],
                    date="2024-06-01", size_px=size_px,
                )

    def test_wrong_extent_raises_before_network(self, size_px):
        """Passing a mismatched bbox must raise ValueError before the
        fetcher does any network I/O."""
        from imint.training import cdse_s2

        # Construct a stale bbox that mimics the actual bug: 2560m extent
        # with size_px=512 would have silently returned 5m GSD data.
        if size_px == 256:
            # For 256, simulate a 128-px-era stale bbox (1280m)
            stale = {"west": 100000, "east": 101280,
                     "south": 6500000, "north": 6501280}
        else:
            stale = {"west": 100000, "east": 102560,
                     "south": 6500000, "north": 6502560}

        # Mock _get_token so a passing assertion reaches it.
        # A failing assertion raises ValueError before network I/O.
        with patch.object(cdse_s2, "_get_token",
                          side_effect=AssertionError("should not reach network")):
            with pytest.raises(ValueError, match="bbox/size_px mismatch"):
                cdse_s2.fetch_s2_scene(
                    stale["west"], stale["south"], stale["east"], stale["north"],
                    date="2024-06-01", size_px=size_px,
                )


class TestTileFetchModuleNoGlobals:
    """tile_fetch should no longer expose TILE_SIZE_M / TILE_SIZE_PX."""

    def test_no_module_level_tile_size_m(self):
        import imint.training.tile_fetch as tf
        assert not hasattr(tf, "TILE_SIZE_M"), \
            "TILE_SIZE_M was removed intentionally — any remaining " \
            "reference is stale code that will silently reintroduce the bug."

    def test_no_module_level_tile_size_px(self):
        import imint.training.tile_fetch as tf
        assert not hasattr(tf, "TILE_SIZE_PX"), \
            "TILE_SIZE_PX was removed intentionally — tile size is a " \
            "TileConfig parameter, not a module global."


class TestGrunddataExtentGuard:
    """skg_grunddata.fetch_grunddata_tile must reject a (bbox, size_px) whose
    extent implies a GSD != 10 m BEFORE the HTTP call — the fetcher-level guard
    the SKG/DEM/markfukt path lacked when prefetch_aux silently produced aux on
    the central-2560m-stretched grid (volume/basal/diameter/dem/markfukt/vpp on
    unified_v2_512 were the central quarter of each tile, scaled 2x)."""

    def test_mismatched_extent_raises_before_network(self):
        from imint.training import skg_grunddata
        # 2560 m extent rendered at 512 px → 5 m/px: the actual bug.
        with patch.object(skg_grunddata, "_download_tile",
                          side_effect=AssertionError("should not reach network")):
            with pytest.raises(ValueError, match="bbox/size_px mismatch"):
                skg_grunddata.fetch_volume_tile(
                    100000, 6500000, 102560, 6502560, size_px=512)

    def test_correct_extent_passes_guard_to_download(self):
        from imint.training import skg_grunddata
        # 5120 m extent at 512 px → 10 m/px: guard passes, reaches the HTTP layer.
        with patch.object(skg_grunddata, "_download_tile",
                          side_effect=AssertionError("reached download")):
            with pytest.raises(AssertionError, match="reached download"):
                skg_grunddata.fetch_volume_tile(
                    100000, 6500000, 105120, 6505120, size_px=512)

    def test_256_dataset_extent_is_valid(self):
        # 2560 m at 256 px IS correct (10 m/px) — must NOT raise on the guard.
        from imint.training import skg_grunddata
        with patch.object(skg_grunddata, "_download_tile",
                          side_effect=AssertionError("reached download")):
            with pytest.raises(AssertionError, match="reached download"):
                skg_grunddata.fetch_volume_tile(
                    100000, 6500000, 102560, 6502560, size_px=256)


class TestPrefetchAuxBboxCoupling:
    """Regression for the prefetch_aux 256/512 misalignment: the fetch bbox must
    be derived from the tile's grid via the SSOT resolve_tile_bbox (extent coupled
    to the spectral pixel count), NEVER from --patch-size-m/half_m or a stale
    bbox_3006. Would FAIL on the old `_bbox_from_tile(tile_path, half_m)` path."""

    def test_fetch_bbox_coupled_to_spectral_px(self, tmp_path):
        pa = pytest.importorskip("scripts.prefetch_aux")

        px = 64                          # 64 px tile → expect 640 m extent
        center = (391280, 6201280)
        # A stale 256-era bbox (2560 m) stored on the tile — must be ignored.
        stale = np.array([center[0] - 1280, center[1] - 1280,
                          center[0] + 1280, center[1] + 1280], dtype=np.int32)
        npz = tmp_path / "tile_391280_6201280.npz"
        np.savez(npz,
                 spectral=np.zeros((1, px, px), np.float32),
                 bbox_3006=stale,
                 easting=np.int32(center[0]), northing=np.int32(center[1]))

        captured = {}

        def fake_dem(west, south, east, north, *, size_px, cache_dir=None):
            captured["bbox"] = (west, south, east, north)
            hw = size_px if isinstance(size_px, tuple) else (size_px, size_px)
            return np.zeros(hw, np.float32)

        # half_m=99999 simulates a wildly wrong --patch-size-m; must be ignored.
        with patch.dict(pa._CHANNEL_FETCHERS, {"dem": fake_dem}):
            pa._add_channels_to_tile(npz, ["dem"], 99999, {"dem": None})

        w, s, e, n = captured["bbox"]
        assert (e - w) == px * 10 == 640, \
            f"extent {(e - w)} m decoupled from {px}px — expected 640 m " \
            f"(old half_m path would give {2 * 99999}, stale bbox would give 2560)"
        assert (n - s) == px * 10 == 640
        assert (w + e) // 2 == center[0]          # center preserved
        assert (s + n) // 2 == center[1]

    def test_force_overwrites_present_channels(self, tmp_path):
        """--force must re-fetch channels that already exist (to overwrite the
        wrong-grid aux); without it, present channels are skipped."""
        pa = pytest.importorskip("scripts.prefetch_aux")
        npz = tmp_path / "tile_1_2.npz"
        np.savez(npz, dem=np.zeros((4, 4), np.float32),
                 volume=np.zeros((4, 4), np.float32))
        assert pa._tile_missing_channels(npz, ["dem", "volume"]) == []
        assert pa._tile_missing_channels(
            npz, ["dem", "volume"], force=True) == ["dem", "volume"]
