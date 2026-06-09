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

    def test_off_lattice_center_snaps_to_grid(self):
        """Off-lattice centre rounds to the nearest gsd multiple (not int-trunc)."""
        t = TileConfig(size_px=512)
        b = t.bbox_from_center(east=281283, north=6471286)  # 3 m E, 4 m N off-grid
        assert (b["west"] + b["east"]) // 2 == 281280    # snapped down
        assert (b["south"] + b["north"]) // 2 == 6471290  # snapped up
        # Fractional inputs snap the same way.
        bf = t.bbox_from_center(east=281280.5, north=6471280.9)
        assert (bf["west"] + bf["east"]) // 2 == 281280

    def test_all_edges_on_lattice(self):
        """Arbitrary centres → all four edges are exact 10 m multiples."""
        t = TileConfig(size_px=512)
        for e, n in [(281283, 6471287), (500001, 6499999), (123456, 6543217)]:
            b = t.bbox_from_center(e, n)
            for k in ("west", "east", "south", "north"):
                assert b[k] % 10 == 0

    def test_odd_size_px_rejected(self):
        """Odd size_px → half_m off the gsd lattice → ValueError."""
        t = TileConfig(size_px=513)
        with pytest.raises(ValueError, match="must be even"):
            t.bbox_from_center(500000, 6500000)

    def test_520_512_256_cocentred(self):
        """Halo (520), canonical (512) and legacy (256) tiles from one centre
        share a single lattice centre → centred crops are clean (store-fork A)."""
        e, n = 281283, 6471287
        centres = {
            (
                (b["west"] + b["east"]) // 2,
                (b["south"] + b["north"]) // 2,
            )
            for b in (
                TileConfig(size_px=px).bbox_from_center(e, n) for px in (520, 512, 256)
            )
        }
        assert len(centres) == 1

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


class TestNativeWindow:
    """native_window() must return integer windows for lattice-aligned bboxes
    and raise loudly otherwise — it replaces the old nearest-neighbour resample."""

    # NMD raster geometry: EPSG:3006, origin on exact 10 m multiples, 10 m pixel.
    NMD_ORIGIN_X = 208450.0
    NMD_ORIGIN_Y = 7671060.0

    def _nmd_transform(self, origin_x=None, origin_y=None):
        from rasterio.transform import from_origin
        return from_origin(
            self.NMD_ORIGIN_X if origin_x is None else origin_x,
            self.NMD_ORIGIN_Y if origin_y is None else origin_y,
            10.0, 10.0,
        )

    @pytest.mark.parametrize("size_px", [256, 512, 520])
    def test_lattice_aligned_bbox_gives_integer_window(self, size_px):
        t = TileConfig(size_px=size_px)
        b = t.bbox_from_center(281283, 6471287)  # off-grid centre → snapped bbox
        win = t.native_window(self._nmd_transform(), b["west"], b["south"], b["east"], b["north"])
        assert win.col_off == int(win.col_off)
        assert win.row_off == int(win.row_off)
        assert win.width == size_px
        assert win.height == size_px

    def test_window_offsets_match_hand_computation(self):
        t = TileConfig(size_px=512)
        b = t.bbox_from_center(281280, 6471280)
        win = t.native_window(self._nmd_transform(), b["west"], b["south"], b["east"], b["north"])
        # col = (west - origin_x)/10 ; row = (origin_y - north)/10
        assert win.col_off == (b["west"] - self.NMD_ORIGIN_X) / 10
        assert win.row_off == (self.NMD_ORIGIN_Y - b["north"]) / 10
        assert (win.col_off, win.row_off, win.width, win.height) == (7027, 119722, 512, 512)

    def test_off_lattice_raster_raises(self):
        """Raster origin off the 10 m lattice → fractional offset → loud raise,
        NOT a silent nearest-neighbour resample."""
        t = TileConfig(size_px=512)
        b = t.bbox_from_center(281280, 6471280)
        off = self._nmd_transform(origin_x=self.NMD_ORIGIN_X + 3.0)  # 3 m off-grid
        with pytest.raises(ValueError, match="offset not integer"):
            t.native_window(off, b["west"], b["south"], b["east"], b["north"])

    def test_wrong_extent_raises(self):
        """Lattice-aligned but wrong-sized bbox (256 m extent under a 512 config)
        → size mismatch raises rather than resampling 256→512."""
        t = TileConfig(size_px=512)
        # 2560 m square (a 256-tile extent), edges still on the lattice.
        west, south = 278720, 6468720
        with pytest.raises(ValueError, match="extent wrong"):
            t.native_window(self._nmd_transform(), west, south, west + 2560, south + 2560)


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
