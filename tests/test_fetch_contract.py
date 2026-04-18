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
