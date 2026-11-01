#!/usr/bin/env python3
"""National coverage of every annual-map input, per year (IM-026, package 3).

Read-only audit for docs/plans/lclu_national_service.md. For Sweden — a
boundary GeoJSON in EPSG:4326, e.g. SCB's RegSO 2025 areas for the whole
country — and each requested year it reports what the service would need and
what is there:

* TESSERA: 0.1° v1/vultr registry tiles that intersect Sweden, present and
  complete, using the registry rows and completeness rule of the campaign
  inventory (scripts/inventory_lclu_demo.py).
* HR-VPP: MGRS tiles that intersect Sweden with all five season-1 metrics in
  the WEkEO index and on disk — the serving rule — the missing pairs, and
  their size at the mean cached COG size (the gap-fill input).
* Forestry (volume, basal area, mean diameter; any scanning round), tree
  height and markfukt: counties present in the SKG mirror, against the 21
  Swedish counties.
* DEM: the GLO-30 1° tiles the country needs, read on demand from public COGs.

No network access; every input is hashed into the report.

    python scripts/audit_lclu_national_coverage.py --boundary sweden.geojson \\
        --registry manifest.parquet --vpp-index /data/vpp_wekeo/index.json \\
        --skg-manifest /data/skg_open/MANIFEST.json --report report.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from imint.inference.lclu import YEARS, public_identity  # noqa: E402
from scripts.cache_predict import file_identity  # noqa: E402

SWEDISH_COUNTIES = ("01", "03", "04", "05", "06", "07", "08", "09", "10", "12", "13", "14",
                    "17", "18", "19", "20", "21", "22", "23", "24", "25")
SUB_BOX_M = 5000      # as plan_lclu_vpp_gapfill: MGRS sampling is exact only for small boxes
LISTED = 50           # missing items listed per year; counts are always complete


def load_sweden(boundary: Path):
    """(EPSG:4326 polygon, EPSG:3006 polygon) of the union of the boundary features."""
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform, unary_union

    features = json.loads(boundary.read_text())["features"]
    if not features:
        raise ValueError("Boundary has no features")
    wgs = unary_union([shape(f["geometry"]) for f in features])
    if wgs.is_empty or not wgs.is_valid:
        raise ValueError("Boundary union is empty or invalid")
    return wgs, transform(Transformer.from_crs(4326, 3006, always_xy=True).transform, wgs)


def tessera(registry: Path, sweden, years) -> dict:
    from shapely.geometry import box
    from shapely.prepared import prep
    from inventory_lclu_demo import block_complete, registry_rows

    inside = prep(sweden)
    west, south, east, north = sweden.bounds
    cells = {(round((i + .5) / 10, 2), round((j + .5) / 10, 2))
             for i in range(math.floor(west * 10), math.floor(east * 10) + 1)
             for j in range(math.floor(south * 10), math.floor(north * 10) + 1)
             if inside.intersects(box(i / 10, j / 10, (i + 1) / 10, (j + 1) / 10))}
    present = defaultdict(set)
    incomplete = defaultdict(list)
    for row in registry_rows(registry, sweden.bounds, years):
        cell = (round(row["lon"], 2), round(row["lat"], 2))
        if cell not in cells:
            continue
        if block_complete(row):
            present[row["year"]].add(cell)
        else:
            incomplete[row["year"]].append(cell)
    out = {}
    for year in years:
        missing = sorted(cells - present[year])
        out[str(year)] = {"expected_tiles": len(cells), "present": len(present[year]),
                          "incomplete": len(incomplete[year]), "missing": len(missing),
                          "missing_listed": [f"{lon:.2f}_{lat:.2f}" for lon, lat in missing[:LISTED]]}
    return out


def mgrs_tiles(sweden_3006) -> set[str]:
    """MGRS tiles touching Sweden, from small sub-boxes that intersect the country."""
    from shapely.geometry import box
    from shapely.prepared import prep
    import derive_missing_vpp_mgrs

    inside = prep(sweden_3006)
    west, south, east, north = sweden_3006.bounds
    tiles: set[str] = set()
    for i in range(math.ceil((east - west) / SUB_BOX_M)):
        for j in range(math.ceil((north - south) / SUB_BOX_M)):
            x0, y0 = west + i * SUB_BOX_M, south + j * SUB_BOX_M
            if inside.intersects(box(x0, y0, x0 + SUB_BOX_M, y0 + SUB_BOX_M)):
                tiles |= derive_missing_vpp_mgrs.mgrs_tiles_for_bbox(
                    {"west": x0, "south": y0, "east": x0 + SUB_BOX_M, "north": y0 + SUB_BOX_M})
    return tiles


def hr_vpp(index_path: Path, tiles: set[str], years) -> dict:
    """Per year: MGRS tiles with all five season-1 metrics indexed and on disk."""
    from imint.data.wekeo_vpp import _VPP_PRODUCT_TYPES
    metrics = set(_VPP_PRODUCT_TYPES)
    index = json.loads(index_path.read_text())
    have = defaultdict(set)
    sizes = []
    for name, meta in index.items():
        path = index_path.parent / name
        if meta.get("season") != 1 or not path.is_file():
            continue
        tile = str(meta["tileId"]).removeprefix("T")
        have[(tile, int(meta["year"]))].add(str(meta["metric"]).upper())
        sizes.append(path.stat().st_size)
    mean_cog = sum(sizes) / len(sizes) if sizes else None
    out, total_missing = {}, 0
    for year in years:
        complete = {t for t in tiles if metrics <= have[(t, year)]}
        missing = sorted(tiles - complete)
        total_missing += len(missing)
        out[str(year)] = {"expected_tiles": len(tiles), "complete": len(complete),
                          "missing": len(missing), "missing_listed": missing[:LISTED]}
    estimate = None if mean_cog is None else round(total_missing * len(metrics) * mean_cog)
    return {"years": out, "missing_pairs": total_missing,
            "gapfill_bytes_at_cached_mean": estimate, "cached_cogs": len(sizes)}


def static_aux(skg_manifest: Path) -> dict:
    """Counties present per SKG layer, by the mirror's own county lists.

    A grunddata variable is read newest scanning round first, per pixel
    (``skg_open.OMDREV_PRIORITY``), and omdrev 3 exists for nine counties
    only. A county therefore counts as covered for a variable when any round
    has it; the per-round counties are reported alongside.
    """
    os.environ["SKG_OPEN_DIR"] = str(skg_manifest.parent)
    from imint.training import skg_open

    manifest = json.loads(skg_manifest.read_text())
    county = re.compile(r"_(\d{2})\.tif$")

    def counties(sources: list[str]) -> set[str]:
        return {m.group(1) for s in sources
                if (m := county.search(s)) and (skg_manifest.parent / s).exists()}

    def coverage(found: set[str]) -> dict:
        return {"present": sorted(found), "missing": sorted(set(SWEDISH_COUNTIES) - found)}

    report = {"markfukt": coverage(set(skg_open._county_rasters(manifest, "markfukt/markfukt_"))),
              "tradhojd": coverage(skg_open.staged_tradhojd_counties(manifest))}
    vrts = manifest.get("vrt", {})
    for variable in skg_open.GRUNDDATA_VARIABLES:
        by_omdrev = {str(r): counties(vrts[key]["sources"]) for r in skg_open.OMDREV_PRIORITY
                     if (key := f"grunddata/{variable}_omdrev{r}.vrt") in vrts}
        report[variable] = {**coverage(set().union(*by_omdrev.values())),
                            "by_omdrev": {r: sorted(c) for r, c in sorted(by_omdrev.items())}}
    return report


def dem(sweden) -> dict:
    from shapely.geometry import box
    from shapely.prepared import prep

    inside = prep(sweden)
    west, south, east, north = sweden.bounds
    tiles = [(lon, lat) for lon in range(math.floor(west), math.ceil(east))
             for lat in range(math.floor(south), math.ceil(north))
             if inside.intersects(box(lon, lat, lon + 1, lat + 1))]
    return {"glo30_tiles": len(tiles),
            "policy": "public Copernicus GLO-30 COGs, read per window on demand"}


def audit(boundary: Path, registry: Path, vpp_index: Path, skg_manifest: Path, years) -> dict:
    sweden, sweden_3006 = load_sweden(boundary)
    tiles = mgrs_tiles(sweden_3006)
    report = {"version": 1, "years": list(years),
              "inputs": {name: public_identity(file_identity(path)) for name, path in
                         (("boundary", boundary), ("registry", registry),
                          ("vpp_index", vpp_index), ("skg_manifest", skg_manifest))},
              "sweden_area_km2": round(sweden_3006.area / 1e6),
              "tessera": tessera(registry, sweden, years),
              "hr_vpp": {"mgrs_tiles": sorted(tiles), **hr_vpp(vpp_index, tiles, years)},
              "static_aux": static_aux(skg_manifest), "dem": dem(sweden),
              "note": ("Metadata and file presence only. A year is offered nationally only "
                       "after qualification receipts; no other year's data stands in.")}
    report["ready_years"] = [str(y) for y in years
                             if report["tessera"][str(y)]["missing"] == 0
                             and report["hr_vpp"]["years"][str(y)]["missing"] == 0
                             and not any(v["missing"] for v in report["static_aux"].values())]
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("boundary", "registry", "vpp-index", "skg-manifest", "report"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    args = ap.parse_args(argv)
    report = audit(args.boundary, args.registry, args.vpp_index, args.skg_manifest, args.years)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"ready_years": report["ready_years"],
                      "tessera_missing": {y: v["missing"] for y, v in report["tessera"].items()},
                      "hr_vpp_missing": {y: v["missing"] for y, v in report["hr_vpp"]["years"].items()},
                      "gapfill_bytes": report["hr_vpp"]["gapfill_bytes_at_cached_mean"],
                      "static_missing": {k: v["missing"] for k, v in report["static_aux"].items()
                                         if v["missing"]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
