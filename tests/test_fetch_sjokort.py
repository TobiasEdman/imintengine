"""
Tests for sjökort (nautical chart) fetching, tiling, rendering, and SLU GET integration.

All tests are offline — no real SLU GET connection or Shibboleth session needed.
S-57 rendering tests mock geopandas/fiona to avoid needing actual ENC data.
"""
from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest

from imint.fetch import (
    FetchError,
    NMD_GRID_SIZE,
    TARGET_CRS,
    SLU_GET_BASE_URL,
    SLU_GET_DOWNLOAD_URL,
    SLU_GET_JOB_ID,
    SLU_GET_MAX_AREA_M2,
    SLU_GET_DEFAULT_MARGIN_M,
    SjokortTile,
    SjokortFetchResult,
    _sjokort_cache_key,
    _tile_sjokort_bbox,
    _order_sjokort_tile,
    _download_sjokort_zip,
    _extract_s57_from_zip,
    _pad_sweref_bbox,
    _deduplicate_s57,
    _depth_colour,
    _S57_DEPTH_COLOURS,
    _S57_LAND_COLOUR,
    _S57_BG_COLOUR,
    fetch_sjokort_data,
)


# ── Constants ────────────────────────────────────────────────────────────────

class TestSjokortConstants:
    """Verify sjökort-related constants are set correctly."""

    def test_slu_get_base_url(self):
        """SLU GET base URL should point to maps.slu.se."""
        assert "maps.slu.se" in SLU_GET_BASE_URL
        assert SLU_GET_BASE_URL.startswith("https://")

    def test_slu_get_download_url(self):
        """Download URL should be under the same host."""
        assert "maps.slu.se" in SLU_GET_DOWNLOAD_URL
        assert "done" in SLU_GET_DOWNLOAD_URL

    def test_slu_get_job_id(self):
        """Job ID for sjökort vector data."""
        assert SLU_GET_JOB_ID == "sjokort_vektor"

    def test_max_area(self):
        """Maximum tile area should be ~2.5 km²."""
        assert SLU_GET_MAX_AREA_M2 == 2_500_000

    def test_default_margin(self):
        """Default margin should be 1 km."""
        assert SLU_GET_DEFAULT_MARGIN_M == 1000

    def test_depth_colours_ordered(self):
        """Depth colour ranges should be contiguous and ascending."""
        for i in range(len(_S57_DEPTH_COLOURS) - 1):
            _, hi, _ = _S57_DEPTH_COLOURS[i]
            lo_next, _, _ = _S57_DEPTH_COLOURS[i + 1]
            assert hi == lo_next, f"Gap between depth range {i} and {i+1}"

    def test_depth_colours_start_at_zero(self):
        """First depth range should start at 0."""
        assert _S57_DEPTH_COLOURS[0][0] == 0

    def test_land_colour_is_hex(self):
        """Land colour should be a valid hex colour string."""
        assert _S57_LAND_COLOUR.startswith("#")
        assert len(_S57_LAND_COLOUR) == 7


# ── SjokortTile ──────────────────────────────────────────────────────────────

class TestSjokortTile:
    """Verify SjokortTile dataclass and area calculation."""

    def test_basic_creation(self):
        """Create a tile with SWEREF99 TM bounds."""
        tile = SjokortTile(north=6488000, south=6487000, east=283000, west=282000)
        assert tile.north == 6488000
        assert tile.south == 6487000
        assert tile.east == 283000
        assert tile.west == 282000

    def test_area_computed(self):
        """area_m2 should be auto-computed from bounds."""
        tile = SjokortTile(north=6488000, south=6487000, east=283000, west=282000)
        assert tile.area_m2 == 1000 * 1000  # 1 km²

    def test_area_larger(self):
        """Verify area for a 2x3 km tile."""
        tile = SjokortTile(north=6490000, south=6487000, east=284000, west=282000)
        assert tile.area_m2 == 2000 * 3000  # 6 km²

    def test_uuid_default_none(self):
        """UUID should be None before ordering."""
        tile = SjokortTile(north=6488000, south=6487000, east=283000, west=282000)
        assert tile.uuid is None

    def test_uuid_settable(self):
        """UUID should be settable after ordering."""
        tile = SjokortTile(north=6488000, south=6487000, east=283000, west=282000)
        tile.uuid = "abc-123"
        assert tile.uuid == "abc-123"


# ── SjokortFetchResult ───────────────────────────────────────────────────────

class TestSjokortFetchResult:
    """Verify SjokortFetchResult dataclass."""

    def test_basic_creation(self):
        """Create a result with required fields."""
        result = SjokortFetchResult(
            s57_paths=[Path("/tmp/a.000")],
            tiles=[],
            bbox_wgs84={"west": 16.0, "south": 58.4, "east": 16.1, "north": 58.5},
            bbox_sweref={"west": 282000, "south": 6487000, "east": 283000, "north": 6488000},
        )
        assert len(result.s57_paths) == 1
        assert result.from_cache is False

    def test_optional_fields(self):
        """Optional fields should default to None/False."""
        result = SjokortFetchResult(
            s57_paths=[], tiles=[],
            bbox_wgs84={}, bbox_sweref={},
        )
        assert result.bbox_sweref_padded is None
        assert result.rendered_png is None
        assert result.from_cache is False

    def test_padded_bbox(self):
        """bbox_sweref_padded should be settable."""
        result = SjokortFetchResult(
            s57_paths=[], tiles=[],
            bbox_wgs84={}, bbox_sweref={},
            bbox_sweref_padded={"west": 281000, "south": 6486000,
                                "east": 284000, "north": 6489000},
        )
        assert result.bbox_sweref_padded["west"] == 281000


# ── _sjokort_cache_key ────────────────────────────────────────────────────────

class TestSjokortCacheKey:
    """Verify deterministic cache key generation."""

    def test_deterministic(self):
        """Same coords should produce same key."""
        coords = {"west": 282000, "south": 6487000, "east": 283000, "north": 6488000}
        k1 = _sjokort_cache_key(coords)
        k2 = _sjokort_cache_key(coords)
        assert k1 == k2

    def test_different_coords_different_key(self):
        """Different coords should produce different keys."""
        c1 = {"west": 282000, "south": 6487000, "east": 283000, "north": 6488000}
        c2 = {"west": 282010, "south": 6487000, "east": 283000, "north": 6488000}
        assert _sjokort_cache_key(c1) != _sjokort_cache_key(c2)

    def test_key_is_string(self):
        """Key should be a hex string."""
        coords = {"west": 282000, "south": 6487000, "east": 283000, "north": 6488000}
        key = _sjokort_cache_key(coords)
        assert isinstance(key, str)
        assert len(key) == 16  # sha256[:16]

    def test_key_is_hex(self):
        """Key should contain only hex characters."""
        coords = {"west": 282000, "south": 6487000, "east": 283000, "north": 6488000}
        key = _sjokort_cache_key(coords)
        int(key, 16)  # Should not raise


# ── _pad_sweref_bbox ──────────────────────────────────────────────────────────

class TestPadSwerefBbox:
    """Verify bbox margin expansion and grid snapping."""

    def test_expands_by_margin(self):
        """Bbox should be expanded by margin_m on each side."""
        coords = {"west": 282000, "south": 6487000, "east": 283000, "north": 6488000,
                  "crs": TARGET_CRS}
        padded = _pad_sweref_bbox(coords, margin_m=1000)
        assert padded["west"] <= 282000 - 1000
        assert padded["south"] <= 6487000 - 1000
        assert padded["east"] >= 283000 + 1000
        assert padded["north"] >= 6488000 + 1000

    def test_grid_aligned(self):
        """Padded bbox should remain aligned to NMD 10m grid."""
        coords = {"west": 282000, "south": 6487000, "east": 283000, "north": 6488000}
        padded = _pad_sweref_bbox(coords, margin_m=1000)
        grid = NMD_GRID_SIZE
        assert padded["west"] % grid == 0
        assert padded["south"] % grid == 0
        assert padded["east"] % grid == 0
        assert padded["north"] % grid == 0

    def test_zero_margin(self):
        """Zero margin should return identical bounds (already aligned)."""
        coords = {"west": 282000, "south": 6487000, "east": 283000, "north": 6488000,
                  "crs": TARGET_CRS}
        padded = _pad_sweref_bbox(coords, margin_m=0)
        assert padded["west"] == 282000
        assert padded["south"] == 6487000
        assert padded["east"] == 283000
        assert padded["north"] == 6488000

    def test_preserves_crs(self):
        """CRS key should be preserved in output."""
        coords = {"west": 282000, "south": 6487000, "east": 283000, "north": 6488000,
                  "crs": TARGET_CRS}
        padded = _pad_sweref_bbox(coords, margin_m=500)
        assert padded["crs"] == TARGET_CRS

    def test_default_crs_when_missing(self):
        """Should default to TARGET_CRS if crs key is missing."""
        coords = {"west": 282000, "south": 6487000, "east": 283000, "north": 6488000}
        padded = _pad_sweref_bbox(coords, margin_m=500)
        assert padded["crs"] == TARGET_CRS

    def test_non_aligned_input(self):
        """Non-aligned input should be properly snapped after padding."""
        coords = {"west": 282005, "south": 6487003, "east": 283007, "north": 6488009}
        padded = _pad_sweref_bbox(coords, margin_m=100)
        grid = NMD_GRID_SIZE
        assert padded["west"] % grid == 0
        assert padded["south"] % grid == 0
        assert padded["east"] % grid == 0
        assert padded["north"] % grid == 0
        # West/south floored, east/north ceiled
        assert padded["west"] <= 282005 - 100
        assert padded["south"] <= 6487003 - 100
        assert padded["east"] >= 283007 + 100
        assert padded["north"] >= 6488009 + 100

    def test_small_margin(self):
        """A margin smaller than grid size should still expand by at least one grid cell."""
        coords = {"west": 282000, "south": 6487000, "east": 283000, "north": 6488000}
        padded = _pad_sweref_bbox(coords, margin_m=5)
        # floor((282000 - 5) / 10) * 10 = floor(281999.5 / 10) * 10 = 281990
        assert padded["west"] == 281990
        assert padded["south"] == 6486990


# ── _tile_sjokort_bbox ────────────────────────────────────────────────────────

class TestTileSjokortBbox:
    """Verify bbox tiling for SLU GET area limits."""

    def test_small_area_no_split(self):
        """Area under max should return a single tile."""
        coords = {"west": 282000, "south": 6487000, "east": 283000, "north": 6488000}
        # 1 km² = 1,000,000 m² < 2,500,000 m²
        tiles = _tile_sjokort_bbox(coords)
        assert len(tiles) == 1
        assert tiles[0].west == 282000
        assert tiles[0].east == 283000

    def test_large_area_splits(self):
        """Area over max should be split into multiple tiles."""
        # 5 km × 5 km = 25 km² >> 2.5 km²
        coords = {"west": 280000, "south": 6480000, "east": 285000, "north": 6485000}
        tiles = _tile_sjokort_bbox(coords)
        assert len(tiles) > 1

    def test_tiles_cover_full_bbox(self):
        """All tiles together should cover the full bbox."""
        coords = {"west": 280000, "south": 6480000, "east": 285000, "north": 6485000}
        tiles = _tile_sjokort_bbox(coords)
        # Check west/east coverage
        min_west = min(t.west for t in tiles)
        max_east = max(t.east for t in tiles)
        min_south = min(t.south for t in tiles)
        max_north = max(t.north for t in tiles)
        assert min_west == coords["west"]
        assert max_east == coords["east"]
        assert min_south == coords["south"]
        assert max_north == coords["north"]

    def test_tiles_under_area_limit(self):
        """Each tile should be under the max area limit."""
        coords = {"west": 280000, "south": 6480000, "east": 285000, "north": 6485000}
        tiles = _tile_sjokort_bbox(coords)
        for tile in tiles:
            assert tile.area_m2 <= SLU_GET_MAX_AREA_M2, (
                f"Tile area {tile.area_m2} exceeds max {SLU_GET_MAX_AREA_M2}"
            )

    def test_tiles_grid_aligned(self):
        """Tile boundaries should be grid-aligned."""
        coords = {"west": 280000, "south": 6480000, "east": 285240, "north": 6489730}
        tiles = _tile_sjokort_bbox(coords)
        grid = NMD_GRID_SIZE
        for tile in tiles:
            assert tile.west % grid == 0
            assert tile.south % grid == 0
            # Only the last column/row edge aligns to the original bbox edge
            # (which may or may not be grid-aligned), so we check internal edges

    def test_exact_boundary_no_split(self):
        """Area exactly at the limit should not be split."""
        # Make a bbox with area exactly 2,500,000 m² = 2500m × 1000m
        coords = {"west": 280000, "south": 6480000, "east": 282500, "north": 6481000}
        tiles = _tile_sjokort_bbox(coords)
        assert len(tiles) == 1

    def test_tile_count_for_known_area(self):
        """Verify tile count for the actual demo area (5240m × 9730m ≈ 51 km²)."""
        coords = {"west": 280080, "south": 6479260, "east": 285320, "north": 6488990}
        tiles = _tile_sjokort_bbox(coords)
        # 51 km² / 2.5 km² ≈ 20.4, so at least 21 tiles
        assert len(tiles) >= 20
        for tile in tiles:
            assert tile.area_m2 <= SLU_GET_MAX_AREA_M2


# ── _order_sjokort_tile ───────────────────────────────────────────────────────

class TestOrderSjokortTile:
    """Verify SLU GET order API integration (mocked)."""

    def test_successful_order(self):
        """Successful order should populate tile UUID."""
        tile = SjokortTile(north=6488000, south=6487000, east=283000, west=282000)
        session = MagicMock()
        session.post.return_value.json.return_value = {"Uuid": "abc-def-123"}
        session.post.return_value.raise_for_status = MagicMock()

        result = _order_sjokort_tile(tile, "test@test.se", session)
        assert result.uuid == "abc-def-123"

    def test_correct_url(self):
        """Order should POST to the correct SLU GET endpoint."""
        tile = SjokortTile(north=6488000, south=6487000, east=283000, west=282000)
        session = MagicMock()
        session.post.return_value.json.return_value = {"Uuid": "xyz"}
        session.post.return_value.raise_for_status = MagicMock()

        _order_sjokort_tile(tile, "test@test.se", session)

        expected_url = (
            f"{SLU_GET_BASE_URL}/api/job/{SLU_GET_JOB_ID}"
            f"/6488000/6487000/283000/282000/test@test.se"
        )
        session.post.assert_called_once_with(expected_url, timeout=30)

    def test_missing_uuid_raises(self):
        """Response without Uuid should raise FetchError."""
        tile = SjokortTile(north=6488000, south=6487000, east=283000, west=282000)
        session = MagicMock()
        session.post.return_value.json.return_value = {"Error": "something"}
        session.post.return_value.raise_for_status = MagicMock()

        with pytest.raises(FetchError, match="no Uuid"):
            _order_sjokort_tile(tile, "test@test.se", session)

    def test_network_error_raises(self):
        """Network failure should raise FetchError."""
        tile = SjokortTile(north=6488000, south=6487000, east=283000, west=282000)
        session = MagicMock()
        session.post.side_effect = ConnectionError("timeout")

        with pytest.raises(FetchError, match="order request failed"):
            _order_sjokort_tile(tile, "test@test.se", session)


# ── _download_sjokort_zip ─────────────────────────────────────────────────────

class TestDownloadSjokortZip:
    """Verify sjökort ZIP download with polling (mocked)."""

    def test_immediate_zip(self, tmp_path):
        """ZIP available immediately → single request, returns path."""
        session = MagicMock()
        zip_content = b"PK\x03\x04" + b"\x00" * 100  # ZIP magic bytes
        resp = MagicMock()
        resp.content = zip_content
        resp.headers = {"Content-Type": "application/zip"}
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        result = _download_sjokort_zip("test-uuid", tmp_path, session=session)
        assert result.exists()
        assert result.name == "test-uuid.zip"
        assert result.read_bytes() == zip_content

    def test_polls_until_ready(self, tmp_path):
        """Should retry when server returns HTML (processing)."""
        session = MagicMock()

        html_resp = MagicMock()
        html_resp.content = b"<html>Processing...</html>"
        html_resp.headers = {"Content-Type": "text/html"}
        html_resp.raise_for_status = MagicMock()

        zip_resp = MagicMock()
        zip_resp.content = b"PK\x03\x04" + b"\x00" * 50
        zip_resp.headers = {"Content-Type": "application/zip"}
        zip_resp.raise_for_status = MagicMock()

        session.get.side_effect = [html_resp, html_resp, zip_resp]

        with patch("time.sleep"):
            result = _download_sjokort_zip(
                "poll-uuid", tmp_path, session=session,
                max_retries=5, retry_delay=0.01,
            )
        assert result.exists()
        assert session.get.call_count == 3

    def test_exhausted_retries_raises(self, tmp_path):
        """Should raise FetchError when retries exhausted."""
        session = MagicMock()
        html_resp = MagicMock()
        html_resp.content = b"<html>Still processing</html>"
        html_resp.headers = {"Content-Type": "text/html"}
        html_resp.raise_for_status = MagicMock()
        session.get.return_value = html_resp

        with patch("time.sleep"):
            with pytest.raises(FetchError, match="not ready"):
                _download_sjokort_zip(
                    "slow-uuid", tmp_path, session=session,
                    max_retries=2, retry_delay=0.01,
                )

    def test_correct_download_url(self, tmp_path):
        """Should request the correct URL."""
        session = MagicMock()
        resp = MagicMock()
        resp.content = b"PK\x03\x04" + b"\x00" * 10
        resp.headers = {"Content-Type": "application/zip"}
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        _download_sjokort_zip("my-uuid-123", tmp_path, session=session)
        session.get.assert_called_once_with(
            f"{SLU_GET_DOWNLOAD_URL}/my-uuid-123.zip", timeout=60
        )


# ── _extract_s57_from_zip ─────────────────────────────────────────────────────

class TestExtractS57FromZip:
    """Verify S-57 extraction from ZIP archives."""

    def _create_test_zip(self, zip_path: Path, filenames: list[str]):
        """Create a ZIP archive with dummy files."""
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name in filenames:
                zf.writestr(name, f"dummy content for {name}")

    def test_extracts_000_files(self, tmp_path):
        """Should extract only .000 files from the ZIP."""
        zip_path = tmp_path / "test.zip"
        self._create_test_zip(zip_path, [
            "SE4HIAX9.000", "SE3EIAX7.000", "readme.txt", "metadata.xml",
        ])
        dest = tmp_path / "out"
        dest.mkdir()
        result = _extract_s57_from_zip(zip_path, dest)
        assert len(result) == 2
        assert all(p.suffix == ".000" for p in result)

    def test_flattens_subdirectories(self, tmp_path):
        """Should extract files from subdirectories to a flat dir."""
        zip_path = tmp_path / "test.zip"
        self._create_test_zip(zip_path, [
            "ENC_ROOT/SE4HIAX9.000", "ENC_ROOT/SE3EIAX7.000",
        ])
        dest = tmp_path / "out"
        dest.mkdir()
        result = _extract_s57_from_zip(zip_path, dest)
        assert len(result) == 2
        # All files should be directly in dest, not in subdirectories
        for p in result:
            assert p.parent == dest

    def test_empty_zip(self, tmp_path):
        """ZIP with no .000 files should return empty list."""
        zip_path = tmp_path / "empty.zip"
        self._create_test_zip(zip_path, ["readme.txt", "data.csv"])
        dest = tmp_path / "out"
        dest.mkdir()
        result = _extract_s57_from_zip(zip_path, dest)
        assert result == []


# ── _deduplicate_s57 ──────────────────────────────────────────────────────────

class TestDeduplicateS57:
    """Verify ENC cell deduplication across tiles."""

    def test_removes_duplicates(self, tmp_path):
        """Same filename stem in different dirs → keep first."""
        (tmp_path / "tile1").mkdir()
        (tmp_path / "tile2").mkdir()
        p1 = tmp_path / "tile1" / "SE4HIAX9.000"
        p2 = tmp_path / "tile2" / "SE4HIAX9.000"
        p1.write_text("first")
        p2.write_text("second")
        result = _deduplicate_s57([p1, p2])
        assert len(result) == 1
        assert result[0] == p1  # first occurrence kept

    def test_keeps_unique_cells(self, tmp_path):
        """Different ENC cells should all be kept."""
        paths = []
        for name in ["SE2EIAX6.000", "SE3EIAX7.000", "SE4HIAX9.000"]:
            p = tmp_path / name
            p.write_text("data")
            paths.append(p)
        result = _deduplicate_s57(paths)
        assert len(result) == 3

    def test_empty_input(self):
        """Empty input should return empty list."""
        assert _deduplicate_s57([]) == []

    def test_single_file(self, tmp_path):
        """Single file should be returned as-is."""
        p = tmp_path / "SE4HIAX9.000"
        p.write_text("data")
        result = _deduplicate_s57([p])
        assert len(result) == 1
        assert result[0] == p

    def test_result_is_sorted(self, tmp_path):
        """Result should be sorted by path."""
        paths = []
        for name in ["ZZ9.000", "AA1.000", "MM5.000"]:
            p = tmp_path / name
            p.write_text("data")
            paths.append(p)
        result = _deduplicate_s57(paths)
        assert result == sorted(result)

    def test_realistic_duplication(self, tmp_path):
        """Simulate real SLU GET output: 6 unique cells × 24 tiles = 110 files."""
        unique_names = [
            "SE2EIAX6.000", "SE3EIAX7.000", "SE3EIAX8.000",
            "SE4HIAX8.000", "SE4HIAX9.000", "SE4HIAX0.000",
        ]
        all_paths = []
        for tile_idx in range(24):
            tile_dir = tmp_path / f"tile_{tile_idx:02d}"
            tile_dir.mkdir()
            for name in unique_names:
                p = tile_dir / name
                p.write_text(f"tile {tile_idx}")
                all_paths.append(p)

        # Should produce 144 paths but only 6 unique
        assert len(all_paths) == 144
        result = _deduplicate_s57(all_paths)
        assert len(result) == 6


# ── _depth_colour ─────────────────────────────────────────────────────────────

class TestDepthColour:
    """Verify S-57 depth-to-colour mapping."""

    def test_shallow_water(self):
        """0–3m range should return very shallow colour."""
        colour = _depth_colour(0.0, 2.0)  # midpoint = 1.0
        assert colour == _S57_DEPTH_COLOURS[0][2]

    def test_moderate_depth(self):
        """10–20m range should return moderate colour."""
        colour = _depth_colour(10.0, 20.0)  # midpoint = 15.0
        assert colour == _S57_DEPTH_COLOURS[3][2]

    def test_deep_water(self):
        """50+ m should return deep colour."""
        colour = _depth_colour(50.0, 100.0)  # midpoint = 75.0
        assert colour == _S57_DEPTH_COLOURS[6][2]

    def test_none_drval1(self):
        """None DRVAL1 should be treated as 0."""
        colour = _depth_colour(None, 4.0)  # midpoint = 2.0
        assert colour == _S57_DEPTH_COLOURS[0][2]

    def test_none_drval2(self):
        """None DRVAL2 should be treated as 0."""
        colour = _depth_colour(4.0, None)  # midpoint = 2.0
        assert colour == _S57_DEPTH_COLOURS[0][2]

    def test_both_none(self):
        """Both None → midpoint=0 → first range."""
        colour = _depth_colour(None, None)
        assert colour == _S57_DEPTH_COLOURS[0][2]

    def test_very_deep_fallback(self):
        """Depth beyond all ranges should return background colour."""
        colour = _depth_colour(1000.0, 2000.0)  # midpoint = 1500
        assert colour == _S57_BG_COLOUR

    def test_all_colours_are_hex(self):
        """All returned colours should be valid hex strings."""
        test_cases = [
            (0, 1), (3, 6), (10, 15), (20, 30), (30, 40), (50, 80), (500, 600),
        ]
        for d1, d2 in test_cases:
            colour = _depth_colour(d1, d2)
            assert colour.startswith("#"), f"Colour {colour} for ({d1},{d2}) not hex"
            assert len(colour) == 7


# ── render_sjokort_png (mocked) ───────────────────────────────────────────────

class TestRenderSjokortPng:
    """Verify S-57 rendering function."""

    def _has_geopandas(self):
        """Check if geopandas/fiona/matplotlib are available."""
        try:
            import geopandas  # noqa: F401
            import fiona  # noqa: F401
            import matplotlib  # noqa: F401
            from PIL import Image  # noqa: F401
            return True
        except ImportError:
            return False

    def test_missing_deps_raises_fetch_error(self):
        """Should raise FetchError if geopandas is not installed."""
        if self._has_geopandas():
            pytest.skip("geopandas is installed — cannot test ImportError path")

        from imint.fetch import render_sjokort_png

        with pytest.raises(FetchError, match="requires geopandas"):
            render_sjokort_png(
                [Path("/fake.000")],
                {"west": 16.0, "south": 58.4, "east": 16.1, "north": 58.5},
                Path("/tmp/out.png"), 324, 573,
            )

    def test_renders_with_real_deps(self, tmp_path):
        """Should render empty chart (no matching features) to PNG."""
        if not self._has_geopandas():
            pytest.skip("geopandas not available")

        from imint.fetch import render_sjokort_png

        # Create a dummy .000 file (fiona will fail to read it, _read_layer
        # catches exceptions and returns empty GeoDataFrame)
        dummy_s57 = tmp_path / "DUMMY.000"
        dummy_s57.write_bytes(b"\x00" * 100)

        output = tmp_path / "sjokort.png"
        bbox = {"west": 16.0, "south": 58.4, "east": 16.1, "north": 58.5}

        result = render_sjokort_png([dummy_s57], bbox, output, 100, 150)
        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0


# ── fetch_sjokort_data ────────────────────────────────────────────────────────

class TestFetchSjokortData:
    """Verify main sjökort fetch orchestration."""

    def test_s57_dir_mode(self, tmp_path):
        """Mode 1: Pre-extracted S-57 directory should skip order/download."""
        s57_dir = tmp_path / "s57"
        s57_dir.mkdir()
        for name in ["SE4HIAX9.000", "SE3EIAX7.000"]:
            (s57_dir / name).write_text("dummy")

        coords = {"west": 16.0, "south": 58.4, "east": 16.1, "north": 58.5}

        with patch("imint.fetch._to_nmd_grid") as mock_grid:
            mock_grid.return_value = {
                "west": 282000, "south": 6487000,
                "east": 283000, "north": 6488000, "crs": TARGET_CRS,
            }
            result = fetch_sjokort_data(coords=coords, s57_dir=str(s57_dir), margin_m=0)

        assert len(result.s57_paths) == 2
        assert result.from_cache is False
        assert result.tiles == []

    def test_s57_dir_deduplicates(self, tmp_path):
        """Mode 1: Duplicate ENC cells across subdirs should be deduplicated."""
        s57_dir = tmp_path / "s57"
        for sub in ["tile1", "tile2"]:
            d = s57_dir / sub
            d.mkdir(parents=True)
            (d / "SE4HIAX9.000").write_text("data")
            (d / "SE3EIAX7.000").write_text("data")

        coords = {"west": 16.0, "south": 58.4, "east": 16.1, "north": 58.5}

        with patch("imint.fetch._to_nmd_grid") as mock_grid:
            mock_grid.return_value = {
                "west": 282000, "south": 6487000,
                "east": 283000, "north": 6488000, "crs": TARGET_CRS,
            }
            result = fetch_sjokort_data(coords=coords, s57_dir=str(s57_dir), margin_m=0)

        assert len(result.s57_paths) == 2  # 4 files → 2 unique

    def test_s57_dir_empty_raises(self, tmp_path):
        """Mode 1: Empty s57_dir should raise FetchError."""
        s57_dir = tmp_path / "empty"
        s57_dir.mkdir()

        coords = {"west": 16.0, "south": 58.4, "east": 16.1, "north": 58.5}

        with patch("imint.fetch._to_nmd_grid") as mock_grid:
            mock_grid.return_value = {
                "west": 282000, "south": 6487000,
                "east": 283000, "north": 6488000, "crs": TARGET_CRS,
            }
            with pytest.raises(FetchError, match="No S-57"):
                fetch_sjokort_data(coords=coords, s57_dir=str(s57_dir))

    def test_cache_hit(self, tmp_path):
        """Mode 2: Cached S-57 files should be returned without ordering."""
        coords = {"west": 16.0, "south": 58.4, "east": 16.1, "north": 58.5}
        projected = {"west": 282000, "south": 6487000,
                     "east": 283000, "north": 6488000, "crs": TARGET_CRS}

        with patch("imint.fetch._to_nmd_grid") as mock_grid, \
             patch("imint.fetch._sjokort_cache_key") as mock_key:
            mock_grid.return_value = projected
            mock_key.return_value = "test_cache_key"

            # Create cache dir with S-57 files
            cache_dir = tmp_path / "cache"
            cache_s57_dir = cache_dir / "test_cache_key"
            cache_s57_dir.mkdir(parents=True)
            (cache_s57_dir / "SE4HIAX9.000").write_text("cached")

            result = fetch_sjokort_data(
                coords=coords, cache_dir=str(cache_dir), margin_m=0,
            )

        assert result.from_cache is True
        assert len(result.s57_paths) == 1

    def test_no_email_no_s57_dir_no_cache_raises(self, tmp_path):
        """Mode 3: No email + no s57_dir + no cache should raise ValueError."""
        coords = {"west": 16.0, "south": 58.4, "east": 16.1, "north": 58.5}

        with patch("imint.fetch._to_nmd_grid") as mock_grid:
            mock_grid.return_value = {
                "west": 282000, "south": 6487000,
                "east": 283000, "north": 6488000, "crs": TARGET_CRS,
            }
            with pytest.raises(ValueError, match="email is required"):
                fetch_sjokort_data(
                    coords=coords, cache_dir=str(tmp_path / "empty_cache"),
                    margin_m=0,
                )

    def test_render_requires_output_path(self, tmp_path):
        """render=True without output_path should raise ValueError."""
        coords = {"west": 16.0, "south": 58.4, "east": 16.1, "north": 58.5}

        with pytest.raises(ValueError, match="output_path is required"):
            fetch_sjokort_data(coords=coords, render=True, output_path=None)

    def test_margin_expands_ordering_bbox(self, tmp_path):
        """Margin should expand the bbox used for ordering."""
        s57_dir = tmp_path / "s57"
        s57_dir.mkdir()
        (s57_dir / "SE4HIAX9.000").write_text("data")

        coords = {"west": 16.0, "south": 58.4, "east": 16.1, "north": 58.5}

        with patch("imint.fetch._to_nmd_grid") as mock_grid, \
             patch("imint.fetch._pad_sweref_bbox") as mock_pad:
            mock_grid.return_value = {
                "west": 282000, "south": 6487000,
                "east": 283000, "north": 6488000, "crs": TARGET_CRS,
            }
            mock_pad.return_value = {
                "west": 281000, "south": 6486000,
                "east": 284000, "north": 6489000, "crs": TARGET_CRS,
            }

            result = fetch_sjokort_data(
                coords=coords, s57_dir=str(s57_dir), margin_m=1000,
            )

        mock_pad.assert_called_once()
        assert result.bbox_sweref_padded is not None
        assert result.bbox_sweref_padded["west"] == 281000

    def test_render_with_s57_dir(self, tmp_path):
        """render=True should call render_sjokort_png."""
        s57_dir = tmp_path / "s57"
        s57_dir.mkdir()
        (s57_dir / "SE4HIAX9.000").write_text("data")

        coords = {"west": 16.0, "south": 58.4, "east": 16.1, "north": 58.5}
        output = tmp_path / "out.png"

        with patch("imint.fetch._to_nmd_grid") as mock_grid, \
             patch("imint.fetch.render_sjokort_png") as mock_render:
            mock_grid.return_value = {
                "west": 282000, "south": 6487000,
                "east": 283000, "north": 6488000, "crs": TARGET_CRS,
            }
            mock_render.return_value = output

            result = fetch_sjokort_data(
                coords=coords, s57_dir=str(s57_dir),
                render=True, output_path=str(output),
                img_w=200, img_h=300, margin_m=0,
            )

        mock_render.assert_called_once()
        assert result.rendered_png == output

    def test_full_order_mode(self, tmp_path):
        """Mode 3: Full order+download flow (mocked)."""
        coords = {"west": 16.0, "south": 58.4, "east": 16.1, "north": 58.5}
        projected = {"west": 282000, "south": 6487000,
                     "east": 283000, "north": 6488000, "crs": TARGET_CRS}

        # Create a real ZIP with a .000 file
        zip_path = tmp_path / "zips" / "test-uuid.zip"
        zip_path.parent.mkdir(parents=True)
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("SE4HIAX9.000", "test data")

        with patch("imint.fetch._to_nmd_grid") as mock_grid, \
             patch("imint.fetch._sjokort_cache_key") as mock_key, \
             patch("imint.fetch._order_sjokort_tile") as mock_order, \
             patch("imint.fetch._download_sjokort_zip") as mock_download:
            mock_grid.return_value = projected
            mock_key.return_value = "order_cache_key"

            # Mock order to set UUID on tile
            def set_uuid(tile, email, session):
                tile.uuid = "test-uuid"
                return tile
            mock_order.side_effect = set_uuid
            mock_download.return_value = zip_path

            cache_dir = tmp_path / "cache"
            result = fetch_sjokort_data(
                coords=coords, email="test@test.se",
                cache_dir=str(cache_dir), margin_m=0,
            )

        assert len(result.s57_paths) >= 1
        assert result.from_cache is False
        mock_order.assert_called_once()
        mock_download.assert_called_once()
