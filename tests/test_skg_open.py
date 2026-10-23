"""Tests for the Skogsstyrelsen open-data mirror reader (IM-022).

Synthetic county rasters stand in for the PVC mirror: a 10 m omdrev 2
raster, a 10 m omdrev 3 raster covering part of the area, and a 12.5 m
omdrev 1 raster, so the newest-omdrev-per-pixel policy, the lattice
alignment, nodata mapping and the cache contract are checked offline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imint.training import skg_grunddata, skg_open
from imint.training.nmd_grid import is_on_grid, snap_bounds, to_nmd_grid_bounds
from tests.skg_mirror_fixture import write_raster

# A 64 x 64 cell tile on the lattice.
W, S, E, N = 500_000, 6_250_000, 500_640, 6_250_640
PX = 64


@pytest.fixture
def mirror(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(skg_open.ENV_MIRROR_DIR, str(tmp_path))
    g = tmp_path / "grunddata"
    # omdrev 2: 10 m, covers the tile plus a 100 m margin, value = 1000 + row*10 + col%10
    rows = cols = PX + 20
    om2 = (1000 + np.arange(rows)[:, None] * 10 + np.arange(cols)[None, :] % 10).astype(np.int16)
    om2[:, -30:] = -1                 # nodata on the east side (county edge)
    write_raster(g / "volym_omdrev2.vrt", om2, W - 100, N + 100, 10.0, nodata=-1)
    # omdrev 3: 10 m, only the NW quarter of the tile, value 3
    om3 = np.full((PX // 2, PX // 2), 3, dtype=np.int16)
    om3[0, 0] = -1                    # one nodata pixel falls through to omdrev 2
    write_raster(g / "volym_omdrev3.vrt", om3, W, N, 10.0, nodata=-1)
    # omdrev 1: 12.5 m, whole area, constant 7 -> bilinear of a constant is 7
    om1 = np.full((80, 80), 7, dtype=np.int16)
    write_raster(g / "volym_omdrev1.vrt", om1, W - 200, N + 200, 12.5, nodata=-1)
    # tradhojd 10 m derivative in dm, covers the tile; nodata strip at the top
    th = np.full((PX, PX), 125, dtype=np.int16)
    th[:4, :] = -1
    th[10, 10] = -3                   # negative garbage is clipped like before
    write_raster(tmp_path / "tradhojd" / "tradhojd_10m.vrt", th, W, N, 10.0, nodata=-1)
    # the real mirror's per-county rasters and manifest: county 01 holds the tile
    write_raster(g / "omdrev2" / "volym_01.tif", om2, W - 100, N + 100, 10.0, nodata=-1)
    write_raster(tmp_path / "tradhojd" / "tradhojd_10m_01.tif", th, W, N, 10.0, nodata=-1)
    _write_manifest(tmp_path, ["grunddata/omdrev2/volym_01.tif", "tradhojd/tradhojd_10m_01.tif"])
    return om2, om3


def _write_manifest(root: Path, derived: list[str]) -> None:
    import json
    th = [d for d in derived if d.startswith("tradhojd/tradhojd_10m_")]
    (root / "MANIFEST.json").write_text(json.dumps({
        "files": {}, "derived": {d: {"sha256": "x"} for d in derived},
        "vrt": {"tradhojd/tradhojd_10m.vrt": {"sources": th}} if th else {}}))


def test_newest_omdrev_wins_per_pixel(mirror):
    om2, om3 = mirror
    arr = skg_open.read_grunddata("volym", W, S, E, N, size_px=PX)
    assert arr.shape == (PX, PX) and arr.dtype == np.float32
    q = PX // 2
    assert (arr[:q, :q][1:, 1:] == 3).all()                       # omdrev 3 where present
    assert arr[0, 0] == om2[10, 10]                               # its nodata pixel -> omdrev 2
    np.testing.assert_array_equal(arr[q:, :q], om2[10 + q:10 + PX, 10:10 + q])   # exact lattice read
    assert (arr[:, -20:] == 7).all()                              # omdrev 2 nodata -> omdrev 1
    assert (arr >= 0).all()


def test_nodata_everywhere_is_zero_not_an_error(mirror, tmp_path):
    (tmp_path / "grunddata" / "volym_omdrev1.vrt").unlink()
    arr = skg_open.read_grunddata("volym", W, S, E, N, size_px=PX)
    assert (arr[:, -20:] == 0).all()


def test_tile_outside_every_county_is_all_zero(mirror):
    arr = skg_open.read_grunddata("volym", W + 50_000, S, E + 50_000, N, size_px=PX)
    assert arr.shape == (PX, PX) and not arr.any()


def test_off_lattice_bbox_is_rejected(mirror):
    with pytest.raises(ValueError, match="not on the 10 m NMD lattice"):
        skg_open.read_grunddata("volym", W + 5, S, E + 5, N, size_px=PX)


def test_unknown_variable_is_rejected(mirror):
    with pytest.raises(ValueError, match="unknown grunddata variable"):
        skg_open.read_grunddata("biomassa", W, S, E, N, size_px=PX)


def test_missing_mirror_fails_loud(monkeypatch, tmp_path):
    monkeypatch.setenv(skg_open.ENV_MIRROR_DIR, str(tmp_path / "nowhere"))
    with pytest.raises(skg_open.SkgMirrorMissingError, match="SKG_OPEN_DIR"):
        skg_open.read_grunddata("volym", W, S, E, N, size_px=PX)
    with pytest.raises(skg_open.SkgMirrorMissingError, match="trädhöjd"):
        skg_open.read_tradhojd(W, S, E, N, size_px=PX)


def test_tree_height_is_metres_and_cache_is_decimetres(mirror, tmp_path):
    cache = tmp_path / "cache"
    m = skg_grunddata.fetch_tree_height_tile(W, S, E, N, size_px=PX, cache_dir=cache)
    assert m.dtype == np.float32 and m[20, 20] == pytest.approx(12.5)
    assert np.isnan(m[:4]).all() and m[10, 10] == 0
    cached = np.load(cache / f"tree_height_nan_{W}_{S}_{E}_{N}.npy")
    assert cached[20, 20] == 125 and np.isnan(cached[0, 0])


def test_tree_height_nodata_is_nan_inside_and_outside_a_county(mirror):
    """The fixture's height raster is -1 in its top 4 rows. County 01's grunddata covers the
    tile's 44 western columns (its east side is nodata): nodata is NaN on both sides, and the
    product's own values, 0 included, pass through."""
    dm = skg_open.read_tradhojd(W, S, E, N, size_px=PX)
    assert np.isnan(dm[:4]).all()
    assert not np.isnan(dm[4:]).any() and dm[20, 20] == 125 and dm[10, 10] == 0


def test_county_cells_are_the_valid_mask_cells(mirror, tmp_path):
    import json
    m = json.loads((tmp_path / "MANIFEST.json").read_text())
    inside = skg_open.county_cells(W, S, E, N, (PX, PX), {"01"}, m)
    assert inside[:, :44].all() and not inside[:, 44:].any()
    assert not skg_open.county_cells(W, S, E, N, (PX, PX), set(), m).any()


def test_public_fetchers_share_the_cache_contract(mirror, tmp_path):
    cache = tmp_path / "cache"
    a = skg_grunddata.fetch_volume_tile(W, S, E, N, size_px=PX, cache_dir=cache)
    assert sorted(p.name for p in cache.iterdir()) == [f"volume_{W}_{S}_{E}_{N}.npy"]
    for p in (tmp_path / "grunddata").glob("*.vrt"):     # the mosaics the reader uses
        p.unlink()
    b = skg_grunddata.fetch_volume_tile(W, S, E, N, size_px=PX, cache_dir=cache)
    np.testing.assert_array_equal(a, b)
    # a cached array of another shape is not returned for a different grid
    with pytest.raises(skg_open.SkgMirrorMissingError):
        skg_grunddata.fetch_volume_tile(W, S, W + 320, S + 320, size_px=32, cache_dir=cache)


def test_basal_area_and_diameter_read_their_own_products(mirror, tmp_path):
    g = tmp_path / "grunddata"
    for var, val in (("grundyta", 21), ("medeldiameter", 17)):
        write_raster(g / f"{var}_omdrev2.vrt", np.full((PX, PX), val, np.int16), W, N, 10.0, nodata=-1)
    assert skg_grunddata.fetch_basal_area_tile(W, S, E, N, size_px=PX)[5, 5] == 21
    assert skg_grunddata.fetch_diameter_tile(W, S, E, N, size_px=PX)[5, 5] == 17


# ── grid snapping ───────────────────────────────────────────────────────────

def test_snap_bounds_expands_outwards_to_10m():
    assert snap_bounds(100_004.9, 6_500_001.0, 100_995.1, 6_500_999.0) == (100_000, 6_500_000, 101_000, 6_501_000)
    assert snap_bounds(100_000, 6_500_000, 101_000, 6_501_000) == (100_000, 6_500_000, 101_000, 6_501_000)


def test_is_on_grid():
    assert is_on_grid(100_000, 6_500_000.0)
    assert not is_on_grid(100_005)


def test_wgs84_bbox_lands_on_the_lattice_in_3006():
    b = to_nmd_grid_bounds({"west": 15.85, "south": 64.25, "east": 16.02, "north": 64.34})
    assert is_on_grid(*b.values()) and b["west"] < b["east"] and b["south"] < b["north"]
    assert 500_000 < b["west"] < 600_000 and 7_100_000 < b["south"] < 7_200_000


def test_cache_hit_cannot_bypass_the_grid_guard(mirror, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    # a stale 2x2 array under a 40 m bbox: the extent implies 20 m/px
    np.save(cache / f"volume_{W}_{S}_{W + 40}_{S + 40}.npy", np.ones((2, 2), np.float32))
    with pytest.raises(ValueError, match="bbox/size_px mismatch"):
        skg_grunddata.fetch_volume_tile(W, S, W + 40, S + 40, size_px=2, cache_dir=cache)
    with pytest.raises(ValueError, match="not on the 10 m NMD lattice"):
        skg_grunddata.fetch_volume_tile(W + 5, S, W + 25, S + 20, size_px=2, cache_dir=cache)


def test_wetland_demo_box_matches_its_pixel_count():
    import importlib.util
    src = Path(__file__).resolve().parents[1] / "demos" / "wetland_pirinen" / "fetch_and_render.py"
    text = src.read_text(encoding="utf-8")
    # evaluate only the geometry block: constants + _projected_box + the assert
    ns: dict = {"np": np}
    start = text.index("AOI_LAT = ")
    end = text.index("assert E - W == N - S == SIZE_PX * 10") + len("assert E - W == N - S == SIZE_PX * 10")
    block = text[start:end]
    block = block[:block.index("WGS84_BBOX")] + block[block.index("def _projected_box"):]
    exec(block, ns)
    assert ns["E"] - ns["W"] == ns["N"] - ns["S"] == ns["SIZE_PX"] * 10 == 10_000
    assert is_on_grid(ns["W"], ns["S"], ns["E"], ns["N"])
    assert 500_000 < ns["W"] < 600_000 and 7_100_000 < ns["S"] < 7_200_000


# ---- Trädhöjd is staged county by county: the reader must never return an unstaged county as zeros ----

def test_tree_height_in_an_unstaged_county_fails_loud(mirror, tmp_path):
    _write_manifest(tmp_path, ["grunddata/omdrev2/volym_01.tif"])          # 01 not staged for height
    with pytest.raises(skg_open.SkgMirrorMissingError, match=r"county \['01'\] is not staged"):
        skg_open.read_tradhojd(W, S, E, N, size_px=PX)


def test_tree_height_outside_every_county_fails_loud(mirror):
    with pytest.raises(skg_open.SkgMirrorMissingError, match="no county's grunddata covers"):
        skg_open.read_tradhojd(W + 50_000, S, E + 50_000, N, size_px=PX)


def test_a_tile_on_a_county_border_needs_both_counties_staged(mirror, tmp_path):
    other = np.full((PX, PX), 900, dtype=np.int16)
    write_raster(tmp_path / "grunddata" / "omdrev2" / "volym_03.tif", other, W + 320, N, 10.0, nodata=-1)
    _write_manifest(tmp_path, ["grunddata/omdrev2/volym_01.tif", "grunddata/omdrev2/volym_03.tif",
                               "tradhojd/tradhojd_10m_01.tif"])
    assert skg_open.counties_in_window(W, S, E, N) == {"01", "03"}
    with pytest.raises(skg_open.SkgMirrorMissingError, match=r"\['03'\]"):
        skg_open.read_tradhojd(W, S, E, N, size_px=PX)


def test_county_padding_outside_its_polygon_does_not_count(mirror, tmp_path):
    pad = np.full((PX, PX), -1, dtype=np.int16)                            # all nodata over the tile
    write_raster(tmp_path / "grunddata" / "omdrev2" / "volym_03.tif", pad, W, N, 10.0, nodata=-1)
    _write_manifest(tmp_path, ["grunddata/omdrev2/volym_01.tif", "grunddata/omdrev2/volym_03.tif",
                               "tradhojd/tradhojd_10m_01.tif"])
    assert skg_open.counties_in_window(W, S, E, N) == {"01"}
    assert skg_open.read_tradhojd(W, S, E, N, size_px=PX)[20, 20] == 125


def test_a_recorded_but_absent_raster_is_not_staged(mirror, tmp_path):
    (tmp_path / "tradhojd" / "tradhojd_10m_01.tif").unlink()
    with pytest.raises(skg_open.SkgMirrorMissingError, match="not staged"):
        skg_open.read_tradhojd(W, S, E, N, size_px=PX)


def test_a_county_recorded_as_derived_but_not_yet_in_the_mosaic_is_not_staged(mirror, tmp_path):
    """Between saving the derivation and rebuilding the VRT the county must still refuse (review B2)."""
    import json
    m = json.loads((tmp_path / "MANIFEST.json").read_text())
    m["vrt"] = {}                                                      # derived recorded, mosaic not yet
    (tmp_path / "MANIFEST.json").write_text(json.dumps(m))
    with pytest.raises(skg_open.SkgMirrorMissingError, match="not staged"):
        skg_open.read_tradhojd(W, S, E, N, size_px=PX)


def test_markfukt_completes_the_county_mask_where_grunddata_is_empty(mirror, tmp_path):
    """Measured 2026-10-19: one top-up tile has no grunddata county but markfukt places it (review Q1)."""
    mf = np.full((PX * 5, PX * 5), 40, dtype=np.uint16)              # 2 m, all land in county 25
    write_raster(tmp_path / "markfukt" / "markfukt_25.tif", mf, W + 50_000, N, 2.0, nodata=65535)
    _write_manifest(tmp_path, ["grunddata/omdrev2/volym_01.tif", "markfukt/markfukt_25.tif",
                               "tradhojd/tradhojd_10m_01.tif"])
    assert skg_open.counties_in_window(W + 50_000, S, E + 50_000, N) == {"25"}
    with pytest.raises(skg_open.SkgMirrorMissingError, match=r"\['25'\] is not staged"):
        skg_open.read_tradhojd(W + 50_000, S, E + 50_000, N, size_px=PX)


def test_markfukt_fill_255_outside_its_county_does_not_count(mirror, tmp_path):
    mf = np.full((PX * 5, PX * 5), 255, dtype=np.uint16)
    write_raster(tmp_path / "markfukt" / "markfukt_03.tif", mf, W, N, 2.0, nodata=65535)
    _write_manifest(tmp_path, ["grunddata/omdrev2/volym_01.tif", "markfukt/markfukt_03.tif",
                               "tradhojd/tradhojd_10m_01.tif"])
    assert skg_open.counties_in_window(W, S, E, N) == {"01"}


def test_tree_height_touching_a_recorded_upstream_gap_fails_loud(mirror, tmp_path):
    import json

    def with_gaps(gaps):
        m = json.loads((tmp_path / "MANIFEST.json").read_text())
        m["derived"]["tradhojd/tradhojd_10m_01.tif"]["gaps"] = gaps
        (tmp_path / "MANIFEST.json").write_text(json.dumps(m))

    with_gaps([[W + 300, S + 300, W + 310, S + 310]])               # one 10 m cell inside the window
    with pytest.raises(skg_open.SkgMirrorMissingError, match=r"upstream gap of county \['01'\]"):
        skg_open.read_tradhojd(W, S, E, N, size_px=PX)
    with_gaps([[E, S, E + 10, N]])                                  # touching the window's edge only
    assert skg_open.read_tradhojd(W, S, E, N, size_px=PX)[20, 20] == 125


def test_an_unreadable_county_raster_in_the_mosaic_raises_not_zeros(tmp_path):
    """A 1280 x 2560 px window over a 2-source mosaic is read by GDAL 3.10 in threads, which would
    return the cut source as nodata (then 0) without the serial read; a 512 px window is serial anyway."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.errors import RasterioIOError

    from rasterio.transform import from_origin

    px = 1280
    for i, name in enumerate(("a.tif", "b.tif")):             # tiled, as the county COGs are
        with rasterio.open(tmp_path / name, "w", driver="GTiff", width=px, height=px, count=1, dtype="int16",
                           crs="EPSG:3006", tiled=True, blockxsize=128, blockysize=128, nodata=-1,
                           transform=from_origin(W + i * px * 10, N, 10, 10)) as d:
            d.write(np.full((px, px), 125, dtype=np.int16), 1)
    raw = (tmp_path / "b.tif").read_bytes()
    (tmp_path / "b.tif").write_bytes(raw[: len(raw) * 6 // 10])
    src = "".join(f'<SimpleSource><SourceFilename relativeToVRT="1">{n}</SourceFilename><SourceBand>1</SourceBand>'
                  f'<SrcRect xOff="0" yOff="0" xSize="{px}" ySize="{px}" />'
                  f'<DstRect xOff="{i * px}" yOff="0" xSize="{px}" ySize="{px}" /></SimpleSource>'
                  for i, n in enumerate(("a.tif", "b.tif")))
    (tmp_path / "m.vrt").write_text(
        f'<VRTDataset rasterXSize="{2 * px}" rasterYSize="{px}"><SRS>{CRS.from_epsg(3006).to_wkt()}</SRS>'
        f'<GeoTransform>{W}, 10, 0, {N}, 0, -10</GeoTransform><VRTRasterBand dataType="Int16" band="1">'
        f'<NoDataValue>-1</NoDataValue>{src}</VRTRasterBand></VRTDataset>')
    with pytest.raises(RasterioIOError):
        skg_open._read_window(tmp_path / "m.vrt", W, N - px * 10, W + 2 * px * 10, N, (px, 2 * px),
                              resampling=Resampling.nearest, fill=-1)
