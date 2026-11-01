"""National coverage audit on a small synthetic 'Sweden' with real file formats."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import audit_lclu_national_coverage as audit

pytest.importorskip("mgrs")
COUNTIES = audit.SWEDISH_COUNTIES


@pytest.fixture
def inputs(tmp_path):
    boundary = tmp_path / "sweden.geojson"
    boundary.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[[17.52, 59.81], [17.78, 59.81],
                                                          [17.78, 59.89], [17.52, 59.89],
                                                          [17.52, 59.81]]]}}]}))
    # Registry: 2024 complete; 2023 has one incomplete tile; noise rows are ignored.
    whole, scales = 128 + 128 * 1000, 128 + 4 * 1000
    rows = [("1.0", "vultr", 2024, lon, 59.85, whole, scales) for lon in (17.55, 17.65, 17.75)]
    rows += [("1.0", "vultr", 2023, 17.55, 59.85, whole, scales),
             ("1.0", "vultr", 2023, 17.65, 59.85, whole, scales),
             ("1.0", "vultr", 2023, 17.75, 59.85, whole - 5, scales),
             ("1.1", "cambridge", 2023, 17.75, 59.85, whole, scales),
             ("1.0", "vultr", 2024, 19.05, 63.85, whole, scales)]
    names = ["version", "variant", "year", "lon", "lat", "grid_size", "scales_size"]
    registry = tmp_path / "manifest.parquet"
    pq.write_table(pa.table({n: [r[i] for r in rows] for i, n in enumerate(names)}), registry)
    # HR-VPP: every MGRS tile complete for 2024; 2023 lacks MINV.
    _, sweden_3006 = audit.load_sweden(boundary)
    tiles = sorted(audit.mgrs_tiles(sweden_3006))
    vpp = tmp_path / "vpp_wekeo"
    vpp.mkdir()
    index = {}
    for year in (2023, 2024):
        for tile in tiles:
            for metric in ("SOSD", "EOSD", "LENGTH", "MAXV", "MINV"):
                if year == 2023 and metric == "MINV":
                    continue
                name = f"VPP_{year}_T{tile}_{metric}_s1.tif"
                (vpp / name).write_bytes(b"x" * 1000)
                index[name] = {"metric": metric, "tileId": f"T{tile}", "year": year,
                               "season": 1, "bounds_4326": [17, 59, 18, 60]}
    (vpp / "index.json").write_text(json.dumps(index))
    # SKG mirror: every county present in every layer.
    skg = tmp_path / "skg_open"
    manifest = {"derived": {}, "vrt": {}}
    for layer in ("markfukt/markfukt_", "tradhojd/tradhojd_10m_",
                  "grunddata/omdrev2/volym_", "grunddata/omdrev2/grundyta_"):
        for county in COUNTIES:
            rel = f"{layer}{county}.tif"
            (skg / rel).parent.mkdir(parents=True, exist_ok=True)
            (skg / rel).write_bytes(b"x")
            manifest["derived"][rel] = {"sha256": "0" * 64}
    manifest["vrt"]["tradhojd/tradhojd_10m.vrt"] = {
        "sources": [f"tradhojd/tradhojd_10m_{c}.tif" for c in COUNTIES]}
    for name in ("volym", "grundyta"):
        manifest["vrt"][f"grunddata/{name}_omdrev2.vrt"] = {
            "sources": [f"grunddata/omdrev2/{name}_{c}.tif" for c in COUNTIES]}
    (skg / "MANIFEST.json").write_text(json.dumps(manifest))
    return dict(boundary=boundary, registry=registry, vpp_index=vpp / "index.json",
                skg_manifest=skg / "MANIFEST.json", tiles=tiles, root=tmp_path)


def test_audit_counts_each_source_and_names_the_ready_years(inputs):
    report = audit.audit(inputs["boundary"], inputs["registry"], inputs["vpp_index"],
                         inputs["skg_manifest"], [2023, 2024])
    assert report["tessera"]["2024"] == {"expected_tiles": 3, "present": 3, "incomplete": 0,
                                         "missing": 0, "missing_listed": []}
    assert report["tessera"]["2023"]["present"] == 2 and report["tessera"]["2023"]["incomplete"] == 1
    assert report["tessera"]["2023"]["missing_listed"] == ["17.75_59.85"]
    assert report["hr_vpp"]["years"]["2024"]["missing"] == 0
    assert report["hr_vpp"]["years"]["2023"]["missing"] == len(inputs["tiles"])
    assert report["hr_vpp"]["gapfill_bytes_at_cached_mean"] == len(inputs["tiles"]) * 5 * 1000
    assert all(not layer["missing"] for layer in report["static_aux"].values())
    assert set(report["static_aux"]) == {"markfukt", "tradhojd", "volym_omdrev2", "grundyta_omdrev2"}
    assert report["ready_years"] == ["2024"]
    assert set(report["inputs"]) == {"boundary", "registry", "vpp_index", "skg_manifest"}


def test_a_county_missing_from_one_layer_blocks_every_year(inputs):
    manifest = json.loads(inputs["skg_manifest"].read_text())
    (inputs["skg_manifest"].parent / "markfukt/markfukt_09.tif").unlink()
    inputs["skg_manifest"].write_text(json.dumps(manifest))
    report = audit.audit(inputs["boundary"], inputs["registry"], inputs["vpp_index"],
                         inputs["skg_manifest"], [2024])
    assert report["static_aux"]["markfukt"]["missing"] == ["09"]
    assert report["ready_years"] == []


def test_cli_writes_the_report_and_a_summary(inputs, capsys):
    out = inputs["root"] / "out" / "report.json"
    assert audit.main(["--boundary", str(inputs["boundary"]), "--registry", str(inputs["registry"]),
                       "--vpp-index", str(inputs["vpp_index"]), "--skg-manifest",
                       str(inputs["skg_manifest"]), "--years", "2023", "2024",
                       "--report", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["ready_years"] == ["2024"] and summary["tessera_missing"] == {"2023": 1, "2024": 0}
    assert json.loads(out.read_text())["years"] == [2023, 2024]
