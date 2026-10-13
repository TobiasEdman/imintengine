"""Offline builder integration: real rasters, parquet, rasterizers and dataset."""
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_labels as builder


@pytest.fixture(params=[False, True], ids=["inplace", "sidecar"])
def case(tmp_path, monkeypatch, request):
    # Builder caches are process-wide; each case has its own reference directory.
    monkeypatch.setattr(builder, "_lpis_sp", {})
    monkeypatch.setattr(builder, "_sks_utforda_sp", None)
    monkeypatch.setattr(builder, "_sks_anmalda_sp", None)
    tile = tmp_path / "tile.npz"
    shape = (8, 8)
    payload = dict(
        spectral=np.ones((24, *shape), dtype=np.float32),
        easting=np.float64(500000), northing=np.float64(6500000),
        dates=np.array(["2021-09-10", "2022-05-10", "2022-06-10", "2022-08-10"]),
        label=np.full(shape, 11, dtype=np.uint8),
        nmd_label_raw=np.full(shape, 2, dtype=np.uint8),
        nmd_area_ha=np.full(shape, .01, dtype=np.float32),
        label_mask=np.full(shape, 4, dtype=np.uint16),
        parcel_area_ha=np.full(shape, .01, dtype=np.float32),
        n_parcels=np.int32(99), harvest_mask=np.ones(shape, dtype=np.uint8),
        n_harvest_polygons=np.int32(99), n_mature_polygons=np.int32(99),
        unrelated=np.array([42]),
    )
    np.savez(tile, **payload)
    out = tmp_path / "labels"
    out.mkdir()
    target = out / tile.name if request.param else tile
    if request.param:
        np.savez(target, label=payload["label"], parcel_area_ha=payload["parcel_area_ha"])
    nmd = tmp_path / "nmd.tif"
    # Deliberately larger than the tile; either center/bbox resolution stays local.
    with rasterio.open(nmd, "w", driver="GTiff", width=32, height=32,
                       count=1, dtype="uint16", crs="EPSG:3006",
                       transform=from_origin(499840, 6500160, 10, 10)) as dst:
        dst.write(np.full((32, 32), 111, dtype=np.uint16), 1)
    refs = tmp_path / "refs"
    refs.mkdir()
    files = {
        "lpis": refs / "jordbruksskiften_2022_spatial.parquet",
        "utforda": refs / "utforda_avverkningar_spatial.parquet",
        "anmalda": refs / "avverkningsanmalningar_spatial.parquet",
    }

    def references(mode="empty"):
        geom = box(499840, 6499840, 500160, 6500160)
        if mode == "empty":
            geom = box(600000, 6600000, 600080, 6600080)
        for kind, path in files.items():
            columns = {"grdkod_mar": [4], "area_ha": [.04]} if kind == "lpis" else {}
            if kind == "utforda":
                columns["Avvdatum"] = ["2010-01-01" if mode == "ineligible" else "2022-05-01"]
            # Ineligible tests isolate the harvest date filter, with empty LPIS/mature.
            geometry = geom
            if mode == "ineligible" and kind != "utforda":
                geometry = box(600000, 6600000, 600080, 6600080)
            gdf = gpd.GeoDataFrame(columns, geometry=[geometry], crs="EPSG:3006")
            bounds = gdf.geometry.bounds
            for key in ("minx", "miny", "maxx", "maxy"):
                gdf[f"_bbox_{key}"] = bounds[key]
            gdf.to_parquet(path)

    references()

    def run():
        return builder.build_tile_label(str(tile), str(nmd), str(refs), str(refs),
                                        label_out_dir=str(out) if request.param else None)

    return dict(tile=tile, target=target, payload=payload, run=run, files=files,
                references=references, sidecar=request.param, root=tmp_path, out=out)


def assert_preserved(case, source_bytes, target_bytes):
    assert case["tile"].read_bytes() == source_bytes
    assert case["target"].read_bytes() == target_bytes
    assert not list(case["root"].rglob("*.tmp.npz"))


@pytest.mark.parametrize("metadata, reason", [
    ({"year": 2021}, "disagrees"),
    ({"year": 2022, "lpis_year": 2021}, "Conflicting"),
    ({"lpis_year": 2021}, "disagrees"),
    ({"dates": None}, "no_tile_year"),
    ({"dates": np.array(["", "invalid"])}, "no_tile_year"),
])
def test_reject_year_before_reference_access(case, monkeypatch, metadata, reason):
    payload = case["payload"] | metadata
    payload = {k: v for k, v in payload.items() if v is not None}
    np.savez(case["tile"], **payload)
    source, target = case["tile"].read_bytes(), case["target"].read_bytes()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("reference accessed before validating year")

    for name in ("fetch_nmd_label_local", "_lpis_handle", "_sks_utforda_handle", "_sks_anmalda_handle"):
        monkeypatch.setattr(builder, name, forbidden)
    result = case["run"]()
    assert result["status"] == "failed", result
    assert reason in result["reason"], result
    assert calls == []
    assert_preserved(case, source, target)


@pytest.mark.parametrize("mode", ["empty", "ineligible"])
def test_empty_queries_clear_all_owned_fields_and_dataset_weights(case, mode):
    case["references"](mode)
    before = case["tile"].read_bytes()
    result = case["run"]()
    assert result["status"] == "ok", result
    with np.load(case["target"], allow_pickle=False) as z:
        for key in ("label_mask", "parcel_area_ha", "n_parcels", "harvest_mask",
                    "n_harvest_polygons", "n_mature_polygons"):
            assert np.all(z[key] == 0), key
        assert z["label_mask"].dtype == np.uint16
        assert z["harvest_mask"].dtype == np.uint8
        assert z["parcel_area_ha"].dtype == np.float32
        np.testing.assert_array_equal(z["label"], np.ones((8, 8), dtype=np.uint8))
        assert np.all(z["nmd_label_raw"] == 1)
        np.testing.assert_allclose(z["nmd_area_ha"], .64)
    assert_source_fields(case, before)
    sample = dataset_sample(case)
    assert bool((sample["label"] == 1).all())
    np.testing.assert_allclose(sample["pixel_weight"].numpy(), 1.25)


def assert_source_fields(case, before):
    if case["sidecar"]:
        assert case["tile"].read_bytes() == before
        with np.load(case["target"]) as z:
            assert "spectral" not in z and "unrelated" not in z
    else:
        with np.load(case["tile"]) as z:
            for key in ("spectral", "easting", "northing", "dates", "unrelated"):
                np.testing.assert_array_equal(z[key], case["payload"][key])


def dataset_sample(case):
    from imint.training.unified_dataset import UnifiedDataset
    (case["root"] / "split_val.txt").write_text(case["tile"].name + "\n")
    ds = UnifiedDataset(lulc_dir=case["root"], split="val", patch_size=8,
                        enable_aux=False, label_dir=case["out"] if case["sidecar"] else None)
    return ds[0]


@pytest.mark.parametrize("overlay", ["crop", "harvest"])
def test_positive_overlays_replace_stale_values(case, overlay):
    case["references"]("positive")
    if overlay == "crop":
        with rasterio.open(case["root"] / "nmd.tif", "r+") as dst:
            dst.write(np.full((32, 32), 3, dtype=np.uint16), 1)
    # Remove the other overlay spatially, retaining an available reference.
    kind = "utforda" if overlay == "crop" else "lpis"
    path = case["files"][kind]
    gdf = gpd.read_parquet(path)
    gdf.geometry = [box(600000, 6600000, 600080, 6600080)]
    for key in ("minx", "miny", "maxx", "maxy"):
        gdf[f"_bbox_{key}"] = gdf.geometry.bounds[key]
    gdf.to_parquet(path)
    before = case["tile"].read_bytes()
    result = case["run"]()
    assert result["status"] == "ok", result
    with np.load(case["target"]) as z:
        assert np.all(z["label"] == (11 if overlay == "crop" else 22))
        assert z["n_parcels"] == (1 if overlay == "crop" else 0)
        assert z["n_harvest_polygons"] == (0 if overlay == "crop" else 1)
        assert z["n_mature_polygons"] == 1
        assert np.all(z["label_mask"] == (4 if overlay == "crop" else 0))
        assert np.all(z["harvest_mask"] == (0 if overlay == "crop" else 1))
        np.testing.assert_allclose(z["parcel_area_ha"], .04 if overlay == "crop" else 0)
        np.testing.assert_allclose(z["nmd_area_ha"], .64)
    assert_source_fields(case, before)
    sample = dataset_sample(case)
    assert bool((sample["label"] == (11 if overlay == "crop" else 22)).all())
    np.testing.assert_allclose(sample["pixel_weight"].numpy(), 4 if overlay == "crop" else 1.25)


@pytest.mark.parametrize("kind", ["lpis", "utforda", "anmalda"])
@pytest.mark.parametrize("unavailable", ["missing", "corrupt"])
def test_unavailable_reference_preserves_existing_outputs(case, kind, unavailable):
    path = case["files"][kind]
    if unavailable == "missing":
        path.unlink()
    else:
        path.write_bytes(b"not a parquet file")
    source, target = case["tile"].read_bytes(), case["target"].read_bytes()
    result = case["run"]()
    assert result["status"] == "failed", result
    if unavailable == "missing":
        assert kind in result["reason"], result
    assert_preserved(case, source, target)


@pytest.mark.parametrize("metadata", [
    {"year": 2022},
    {"lpis_year": 2022},
    {"year": 2022, "lpis_year": 2022},
    {"year": 2022, "dates": None},
    {"dates": np.array(["2021-09-10", "2021-10-10", "2022-05-10", "2022-08-10"])},
])
def test_valid_year_sources_use_current_reference(case, metadata):
    payload = {k: v for k, v in (case["payload"] | metadata).items() if v is not None}
    np.savez(case["tile"], **payload)
    # Only 2022 LPIS exists: choosing the previous autumn would fail closed.
    result = case["run"]()
    assert result["status"] == "ok", result


def test_unavailable_nmd_preserves_existing_outputs(case, monkeypatch):
    monkeypatch.setattr(builder, "fetch_nmd_label_local", lambda *a, **kw: None)
    source, target = case["tile"].read_bytes(), case["target"].read_bytes()
    result = case["run"]()
    assert result["status"] == "failed" and result["reason"] == "no_nmd"
    assert_preserved(case, source, target)


def test_empty_crop_named_tile_still_fails(case):
    renamed = case["tile"].with_name("crop_tile.npz")
    case["tile"].rename(renamed)
    source = renamed.read_bytes()
    result = builder.build_tile_label(
        str(renamed), str(case["root"] / "nmd.tif"),
        str(case["root"] / "refs"), str(case["root"] / "refs"),
        label_out_dir=str(case["out"]) if case["sidecar"] else None,
    )
    assert result["status"] == "failed" and "no LPIS overlay" in result["reason"]
    assert renamed.read_bytes() == source
    assert not (case["out"] / renamed.name).exists()
