"""TESSERA Zarr source: parity with the tile path, zone seams, gaps, cache, layout."""
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

zarr = pytest.importorskip("zarr")

from imint.inference import lclu_zarr
from imint.inference.lclu import Grid, InputUnavailable
from scripts import prepare_lclu as prep

YEARS = [2024, 2025]


def write_store(path, zones):
    """zones[n] = (x0, y0, embeddings (T,128,H,W) int8, scales (T,H,W) float32)."""
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    for number, (x0, y0, emb, scales) in zones.items():
        group = root.create_group(f"utm{number:02d}")
        group.attrs["proj:code"] = f"EPSG:{32600 + number}"
        group.create_array("embeddings", data=emb, chunks=(1, 128, 64, 64))
        group.create_array("scales", data=scales, chunks=(1, 64, 64))
        group.create_array("time", data=np.array(YEARS, np.int32))
        rows, cols = scales.shape[1:]
        group.create_array("x", data=x0 + (np.arange(cols) + .5) * 10)
        group.create_array("y", data=y0 - (np.arange(rows) + .5) * 10)
    return str(path)


def lattice_block(grid, zone, margin=1500):
    """Origin on the zone's 10 m lattice and a shape that covers the grid with margin."""
    b = grid.bounds()
    west, south, east, north = transform_bounds("EPSG:3006", f"EPSG:{32600 + zone}",
                                                b["west"], b["south"], b["east"], b["north"])
    x0, y0 = np.floor((west - margin) / 10) * 10, np.ceil((north + margin) / 10) * 10
    cols, rows = int(np.ceil((east + margin - x0) / 10)), int(np.ceil((y0 - south + margin) / 10))
    return x0, y0, rows, cols


def synthetic(rows, cols, seed=0, lake=True):
    rng = np.random.default_rng(seed)
    q = rng.integers(-120, 120, (rows, cols, 128), dtype=np.int8)
    q[q == 0] = 1
    scales = np.linspace(.001, .02, rows * cols, dtype=np.float32).reshape(rows, cols)
    water = np.zeros((rows, cols), bool)
    water[200:260, 300:380] = lake
    q[water] = 0
    return q, scales, water


def zarr_zone(q, scales, water, years=YEARS):
    emb = np.zeros((len(years), 128, *scales.shape), np.int8)
    sc = np.full((len(years), *scales.shape), np.inf, np.float32)
    emb[-1] = q.transpose(2, 0, 1)
    sc[-1] = np.where(water, np.nan, scales)
    return emb, sc


def test_matches_the_tile_path_when_a_tile_sits_on_the_lattice(tmp_path, monkeypatch):
    grid = Grid.from_lonlat(16.5, 59.3)
    x0, y0, rows, cols = lattice_block(grid, 33)
    q, scales, water = synthetic(rows, cols)
    tx = from_origin(x0, y0, 10, 10)
    ref = tmp_path / "ref.tif"
    with rasterio.open(ref, "w", driver="GTiff", width=cols, height=rows, count=1,
                       dtype="uint8", crs="EPSG:32633", transform=tx) as dst:
        dst.write(np.ones((rows, cols), np.uint8), 1)
    arrays = {16.45: (q.astype(np.float32) * scales[:, :, None], "EPSG:32633", tx)}
    gt = SimpleNamespace(
        registry=SimpleNamespace(load_blocks_for_region=lambda **kw: [(2025, 16.45, 59.3)],
                                 fetch_landmask=lambda **kw: ref),
        fetch_embedding=lambda **kw: arrays[kw["lon"]])
    monkeypatch.setattr(prep, "embedding_files", lambda *_: {ref: ref.stat().st_size})
    expected, expected_valid, _ = prep.assemble_embedding(gt, grid, 2025, max_blocks=4, max_bytes=1)

    store = write_store(tmp_path / "v1.zarr", {33: (x0, y0, *zarr_zone(q, scales, water))})
    b = grid.bounds()
    bounds = (b["west"], b["south"], b["east"], b["north"])
    result, valid, provenance = lclu_zarr.read_embedding(store, bounds, 512, 2025,
                                                         cache_dir=tmp_path / "cache")
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(valid, expected_valid)
    assert not valid.all() and valid.sum() > 200_000          # the water patch is invalid
    assert provenance["embedding"] == lclu_zarr.ZARR_EMBEDDING
    assert provenance["zones"][0]["water_px"] > 0 and provenance["zones"][0]["missing_px"] == 0


def two_zone_store(tmp_path, grid, overlap_m=60.0, gap_m=0.0):
    """Zone 33 holds the tiles west of 18° E, zone 34 those east, as the builder assigns them."""
    zones = {}
    for zone in (33, 34):
        x0, y0, rows, cols = lattice_block(grid, zone)
        q, scales, _ = synthetic(rows, cols, seed=zone, lake=False)
        xs, ys = np.meshgrid(x0 + (np.arange(cols) + .5) * 10, y0 - (np.arange(rows) + .5) * 10)
        lon, _ = Transformer.from_crs(f"EPSG:{32600 + zone}", "EPSG:4326",
                                      always_xy=True).transform(xs, ys)
        metres_east_of_18 = (lon - 18.0) * 111_320 * np.cos(np.radians(59.3))
        own = (metres_east_of_18 < overlap_m if zone == 33 else metres_east_of_18 > -overlap_m)
        own &= np.abs(metres_east_of_18) >= gap_m
        emb, sc = zarr_zone(q, scales, np.zeros_like(own))
        sc[-1][~own] = np.inf
        emb[-1][:, ~own] = 0
        zones[zone] = (x0, y0, emb, sc)
    return write_store(tmp_path / "v1.zarr", zones)


def bounds_of(grid):
    b = grid.bounds()
    return (b["west"], b["south"], b["east"], b["north"])


def test_a_window_across_18_east_reads_both_zones_without_a_gap(tmp_path):
    grid = Grid.from_lonlat(18.0, 59.3)
    store = two_zone_store(tmp_path, grid)
    result, valid, provenance = lclu_zarr.read_embedding(store, bounds_of(grid), 512, 2025,
                                                         cache_dir=tmp_path / "cache")
    assert [z["zone"] for z in provenance["zones"]] == [33, 34]
    assert all(z["missing_px"] > 0 for z in provenance["zones"])
    assert valid.mean() > .99 and np.isfinite(result).all()


def test_a_strip_missing_in_both_zones_is_a_gap_not_water(tmp_path):
    grid = Grid.from_lonlat(18.0, 59.3)
    store = two_zone_store(tmp_path, grid, overlap_m=0.0, gap_m=200.0)
    with pytest.raises(InputUnavailable, match="no data for"):
        lclu_zarr.read_embedding(store, bounds_of(grid), 512, 2025, cache_dir=tmp_path / "cache")


def test_a_second_read_comes_from_the_disk_cache(tmp_path, monkeypatch):
    grid = Grid.from_lonlat(18.0, 59.3)
    store = two_zone_store(tmp_path, grid)
    first = lclu_zarr.read_embedding(store, bounds_of(grid), 512, 2025, cache_dir=tmp_path / "c")
    cached = sorted((tmp_path / "c").rglob("*.npz"))
    assert len(cached) == 2 and not list((tmp_path / "c").rglob("*.tmp*"))

    class Forbidden:
        def __getitem__(self, _):
            raise AssertionError("store data read despite a cached window")

    opened = lclu_zarr.open_zone
    monkeypatch.setattr(lclu_zarr, "open_zone", lambda root, n: dataclasses.replace(
        opened(root, n), embeddings=Forbidden(), scales=Forbidden()))
    second = lclu_zarr.read_embedding(store, bounds_of(grid), 512, 2025, cache_dir=tmp_path / "c")
    for a, b in zip(first[:2], second[:2]):
        np.testing.assert_array_equal(a, b)


def test_training_rule_keeps_every_serving_pixel_and_unknown_rules_fail(tmp_path):
    grid = Grid.from_lonlat(16.5, 59.3)
    x0, y0, rows, cols = lattice_block(grid, 33)
    store = write_store(tmp_path / "v1.zarr", {33: (x0, y0, *zarr_zone(*synthetic(rows, cols)))})
    _, serving, _ = lclu_zarr.read_embedding(store, bounds_of(grid), 512, 2025,
                                             cache_dir=tmp_path / "c")
    _, training, provenance = lclu_zarr.read_embedding(store, bounds_of(grid), 512, 2025,
                                                       cache_dir=tmp_path / "c", merge="training")
    assert (training >= serving).all() and training.sum() > serving.sum()
    assert provenance["merge"] == "training"
    with pytest.raises(ValueError, match="merge"):
        lclu_zarr.read_embedding(store, bounds_of(grid), 512, 2025, cache_dir=tmp_path / "c",
                                 merge="nearest")


def test_missing_year_zone_and_bad_layout_fail_loudly(tmp_path):
    grid = Grid.from_lonlat(16.5, 59.3)
    x0, y0, rows, cols = lattice_block(grid, 33)
    emb, sc = zarr_zone(*synthetic(rows, cols))
    store = write_store(tmp_path / "ok.zarr", {33: (x0, y0, emb, sc)})
    with pytest.raises(InputUnavailable, match="no 2023"):
        lclu_zarr.read_embedding(store, bounds_of(grid), 512, 2023, cache_dir=tmp_path / "c")
    with pytest.raises(InputUnavailable, match="no zone utm34"):
        lclu_zarr.read_embedding(store, bounds_of(Grid.from_lonlat(19.0, 59.3)), 512, 2025,
                                 cache_dir=tmp_path / "c")
    wrong = write_store(tmp_path / "wrong.zarr", {33: (x0, y0, emb.astype(np.int16), sc)})
    with pytest.raises(ValueError, match="layout"):
        lclu_zarr.read_embedding(wrong, bounds_of(grid), 512, 2025, cache_dir=tmp_path / "c")
    zarr.open_group(store, mode="r+")["utm33"].attrs["proj:code"] = "EPSG:32634"
    with pytest.raises(ValueError, match="declares EPSG:32634"):
        lclu_zarr.read_embedding(store, bounds_of(grid), 512, 2025, cache_dir=tmp_path / "c")


def test_tiles_go_to_the_zone_of_their_centre():
    assert lclu_zarr.tile_zones(17.93, 59.2, 17.99, 59.3) == [33]
    assert lclu_zarr.tile_zones(17.97, 59.2, 18.04, 59.3) == [33, 34]
    assert lclu_zarr.tile_zones(20.1, 67.8, 20.3, 67.9) == [34]
    with pytest.raises(ValueError, match="northern"):
        lclu_zarr.tile_zones(18.0, -1.0, 18.1, 1.0)


# ── scripts/compare_tessera_zarr.py against cohort-style tiles ──────────────

def cohort_tile(tmp_path, grid, name, block, source="geotessera-0.10.2"):
    """A tile whose tessera comes from the enrichment code on one NPY-style block."""
    from scripts.enrich_tiles_tessera import _fetch_tessera_for_tile

    arr, crs, tx = block
    gt = SimpleNamespace(
        registry=SimpleNamespace(load_blocks_for_region=lambda **kw: [(2025, 16.45, 59.3)]),
        fetch_embeddings=lambda blocks: iter([(2025, 16.45, 59.3, arr, crs, tx)]))
    tessera, n = _fetch_tessera_for_tile(gt, grid.bounds(), 2025, 512)
    assert n == 1
    np.savez(tmp_path / "tiles" / f"{name}.npz", tessera=tessera, tessera_year=np.int32(2025),
             has_tessera=np.int32(1), tessera_source=np.array(source),
             easting=np.int32(grid.east), northing=np.int32(grid.north))


@pytest.mark.parametrize("offset_m, identical", [(0.0, True), (3.0, False)])
def test_comparison_finds_identity_on_the_lattice_and_the_rounding_shift_off_it(
        tmp_path, offset_m, identical):
    from scripts import compare_tessera_zarr as compare

    grid = Grid.from_lonlat(16.5, 59.3)
    x0, y0, rows, cols = lattice_block(grid, 33)
    q, scales, water = synthetic(rows, cols)
    (tmp_path / "tiles").mkdir()
    # The NPY tile keeps its own origin; the store builder rounds it to the lattice.
    block = (q.astype(np.float32) * scales[:, :, None], "EPSG:32633",
             from_origin(x0 + offset_m, y0, 10, 10))
    cohort_tile(tmp_path, grid, "a_tile", block)
    cohort_tile(tmp_path, grid, "other_source", block, source="geotessera-0.7.5")
    store = write_store(tmp_path / "v1.zarr", {33: (x0, y0, *zarr_zone(q, scales, water))})
    report = tmp_path / "out" / "report.json"
    assert compare.main(["--data-dir", str(tmp_path / "tiles"), "--n", "5", "--store", store,
                         "--cache-dir", str(tmp_path / "cache"), "--report", str(report)]) == 0
    result = json.loads(report.read_text())
    assert [t["tile"] for t in result["tiles"]] == ["a_tile"]   # other sources are excluded
    summary = result["summary"]
    assert summary["ok"] == 1 and summary["valid_agreement"] > .99
    if identical:
        assert summary["identical_fraction"] == 1.0 and summary["max_abs_diff"] == 0.0
    else:
        assert summary["identical_fraction"] < .01 and summary["max_abs_diff"] > 0


def test_comparison_refuses_a_directory_without_cohort_tiles(tmp_path):
    from scripts import compare_tessera_zarr as compare

    (tmp_path / "tiles").mkdir()
    assert compare.main(["--data-dir", str(tmp_path / "tiles"), "--cache-dir", str(tmp_path),
                         "--report", str(tmp_path / "r.json")]) == 1


def test_compare_job_pins_the_code_it_runs():
    import re
    import subprocess
    from pathlib import Path

    yaml = pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[2]
    job = yaml.safe_load((root / "k8s/tessera-zarr-compare-job.yaml").read_text())
    container = job["spec"]["template"]["spec"]["containers"][0]
    script = container["command"][-1]
    assert "@sha256:" in container["image"]
    assert 'test "$(git rev-parse HEAD)" = "$SHA"' in script
    assert "python3 scripts/compare_tessera_zarr.py" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    sha = re.search(r"SHA=([0-9a-f]{40})", script)[1]
    git = ["git", "-C", str(root)]
    if subprocess.run([*git, "cat-file", "-e", sha], capture_output=True).returncode:
        pytest.skip(f"{sha} not in this clone")
    assert subprocess.run([*git, "merge-base", "--is-ancestor", sha, "HEAD"]).returncode == 0
    changed = subprocess.run([*git, "diff", "--name-only", sha, "HEAD", "--",
                              "scripts/compare_tessera_zarr.py", "imint/inference/lclu_zarr.py"],
                             capture_output=True, text=True, check=True).stdout.split()
    assert changed == [], f"bump SHA in k8s/tessera-zarr-compare-job.yaml; changed: {changed}"
