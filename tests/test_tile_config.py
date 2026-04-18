"""Tests for imint.training.tile_config.TileConfig."""
from __future__ import annotations

import pytest

from imint.training.tile_config import TileConfig


class TestConstruction:
    def test_defaults_to_10m_gsd(self):
        t = TileConfig(size_px=512)
        assert t.gsd_m == 10.0

    def test_rejects_zero_or_negative(self):
        with pytest.raises(ValueError):
            TileConfig(size_px=0)
        with pytest.raises(ValueError):
            TileConfig(size_px=-1)
        with pytest.raises(ValueError):
            TileConfig(size_px=256, gsd_m=0)

    def test_frozen(self):
        t = TileConfig(size_px=256)
        with pytest.raises(Exception):
            t.size_px = 512  # frozen dataclass

    @pytest.mark.parametrize("size_px", [128, 256, 384, 512, 1024, 2048])
    def test_various_sizes(self, size_px):
        t = TileConfig(size_px=size_px)
        assert t.size_m == size_px * 10
        assert t.half_m == t.size_m // 2


class TestBboxFromCenter:
    def test_centered_square(self):
        t = TileConfig(size_px=512)
        b = t.bbox_from_center(east=281280, north=6471280)
        # 512 × 10 = 5120 m, half = 2560
        assert b == {
            "west":  278720, "east":  283840,
            "south": 6468720, "north": 6473840,
        }
        assert (b["east"] - b["west"]) == 5120
        assert (b["north"] - b["south"]) == 5120

    def test_256_size(self):
        t = TileConfig(size_px=256)
        b = t.bbox_from_center(east=500000, north=6500000)
        # 256 × 10 = 2560 m, half = 1280
        assert b == {
            "west":  498720, "east":  501280,
            "south": 6498720, "north": 6501280,
        }

    def test_float_center_coerced_to_int(self):
        t = TileConfig(size_px=512)
        b = t.bbox_from_center(east=281280.5, north=6471280.9)
        # int() truncates toward zero
        assert b["west"] == 281280 - 2560
        assert b["east"] == 281280 + 2560

    def test_custom_gsd(self):
        t = TileConfig(size_px=100, gsd_m=20.0)
        # 100 × 20 = 2000 m, half = 1000
        b = t.bbox_from_center(east=500000, north=6500000)
        assert (b["east"] - b["west"]) == 2000


class TestAssertBboxMatches:
    def test_correct_bbox_passes(self):
        t = TileConfig(size_px=512)
        t.assert_bbox_matches(t.bbox_from_center(281280, 6471280))  # no raise

    def test_wrong_extent_raises(self):
        """256-era manifest bbox (2560m) with a 512-px config → reject."""
        t = TileConfig(size_px=512)
        stale = {
            "west": 280000, "east": 282560,   # 2560m wide
            "south": 6470000, "north": 6472560,
        }
        with pytest.raises(ValueError, match="expects 5120m bbox"):
            t.assert_bbox_matches(stale)

    def test_one_meter_tolerance(self):
        t = TileConfig(size_px=512)
        # +1 m difference — allowed
        t.assert_bbox_matches({
            "west": 0, "east": 5121,
            "south": 0, "north": 5121,
        })
        # +2 m difference — rejected
        with pytest.raises(ValueError):
            t.assert_bbox_matches({
                "west": 0, "east": 5122,
                "south": 0, "north": 5122,
            })

    def test_non_square_raises(self):
        t = TileConfig(size_px=512)
        with pytest.raises(ValueError):
            t.assert_bbox_matches({
                "west": 0, "east": 5120,
                "south": 0, "north": 2560,  # wrong N-S extent
            })


class TestRoundTrip:
    @pytest.mark.parametrize("size_px", [256, 512, 1024])
    @pytest.mark.parametrize("center", [(281280, 6471280), (500000, 6500000), (0, 0)])
    def test_center_bbox_center_roundtrip(self, size_px, center):
        t = TileConfig(size_px=size_px)
        e, n = center
        b = t.bbox_from_center(e, n)
        # Center should be recoverable exactly
        cx = (b["west"] + b["east"]) // 2
        cy = (b["south"] + b["north"]) // 2
        assert (cx, cy) == (e, n)
        # Assert should pass on the bbox we just built
        t.assert_bbox_matches(b)
