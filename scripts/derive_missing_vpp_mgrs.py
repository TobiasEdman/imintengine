#!/usr/bin/env python3
"""Derive the (MGRS, year) set needed to WEkEO-gap-fill VPP for a tile dir.

Post-run step 1 of the orphan-512 VPP remediation (checkpoint 2026-07-03):
~44% of the tiles written by campaign-orphan-512 lack the five
``vpp_{sosd,eosd,length,maxv,minv}`` channels because their MGRS×year COGs
were absent from the WEkEO cache (built for the 44-MGRS recoreg footprint)
and ``VPP_SOURCE=wekeo`` correctly refused to fall back to CDSE PU.

This script scans a tile directory, finds every tile whose VPP block is
missing/empty (same predicate ``backfill_vpp._vpp_is_empty`` uses, so the
derived set covers exactly what ``backfill_vpp.py --force`` will request),
derives the covering Sentinel-2 MGRS tile IDs from each tile's EPSG:3006
bbox, subtracts the (MGRS, year) pairs already complete in the WEkEO COG
cache, and emits the remainder as ready-to-run
``scripts/prefetch_vpp_wekeo.py`` invocations + a JSON report.

The bbox → MGRS primitive lives here because the repo had none
(``prefetch_vpp_wekeo.py`` docstring): bbox corners + center are converted
to WGS84 (pyproj) and mapped to their containing 100 km MGRS cell (mgrs
lib, precision 0) — the cell ID is the S2 tileId whose HR-VPP COG covers
the point.

Usage (one-shot pod, after campaign-orphan-512 is terminal):
    python scripts/derive_missing_vpp_mgrs.py \\
        --data-dir /data/unified_v2_512_orphans_staging \\
        --vpp-cog-dir /data/vpp_wekeo \\
        --output /data/audits/orphan_512_missing_vpp_mgrs.json

Requires: numpy, pyproj, mgrs (pip install mgrs).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from backfill_vpp import _tile_year, _vpp_is_empty  # noqa: E402
from imint.training.tile_bbox import resolve_fetch_bbox  # noqa: E402
from imint.training.wekeo_vpp import _parse_vpp_filename  # noqa: E402

# All five HR-VPP metrics must be cached for a (MGRS, year) to count as
# covered — a partial set would still make fetch_vpp_tiles_local come back
# under the coverage floor and the backfill would re-miss.
_REQUIRED_METRICS = frozenset({"sosd", "eosd", "length", "maxv", "minv"})


# ── bbox → MGRS ──────────────────────────────────────────────────────────

def mgrs_tiles_for_bbox(bbox: dict[str, float]) -> set[str]:
    """Containing S2/MGRS tile IDs for an EPSG:3006 bbox (corners + center).

    A 5.12 km tile is far smaller than the 100 km MGRS cell, so the four
    corners plus the center enumerate every cell the tile can touch.
    """
    import mgrs
    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)
    conv = mgrs.MGRS()
    w, s, e, n = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    points = [(w, s), (w, n), (e, s), (e, n), ((w + e) / 2, (s + n) / 2)]
    tiles: set[str] = set()
    for x, y in points:
        lon, lat = to_wgs84.transform(x, y)
        cell = conv.toMGRS(lat, lon, MGRSPrecision=0)
        if isinstance(cell, bytes):
            cell = cell.decode("ascii")
        tiles.add(cell)
    return tiles


# ── WEkEO cache inventory ────────────────────────────────────────────────

def cached_pairs(vpp_cog_dir: str) -> set[tuple[str, int]]:
    """(MGRS, year) pairs with ALL five metric COGs present in the cache."""
    metrics: dict[tuple[str, int], set[str]] = defaultdict(set)
    for path in glob.glob(os.path.join(vpp_cog_dir, "VPP_*.tif")):
        meta = _parse_vpp_filename(os.path.basename(path))
        if meta is None:
            continue
        tile = meta["tileId"]
        tile = tile[1:] if tile.startswith("T") else tile  # T33VWG → 33VWG
        metrics[(tile, int(meta["year"]))].add(meta["metric"].lower())
    return {pair for pair, got in metrics.items() if got >= _REQUIRED_METRICS}


# ── Scan ─────────────────────────────────────────────────────────────────

def derive(data_dir: str, vpp_cog_dir: str) -> dict:
    """Scan ``data_dir`` and return the gap-fill report dict."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    needed: dict[tuple[str, int], set[str]] = defaultdict(set)  # pair → tiles
    skipped: dict[str, list[str]] = defaultdict(list)
    n_missing = 0

    for path in paths:
        name = os.path.basename(path)
        try:
            data = np.load(path, allow_pickle=True)
        except Exception:
            skipped["unreadable"].append(name)
            continue
        try:
            if not _vpp_is_empty(data):
                continue
            n_missing += 1
            year = _tile_year(data)
            if year is None:
                skipped["no_year"].append(name)
                continue
            bbox, _ = resolve_fetch_bbox(name=name, npz_data=data)
            if bbox is None:
                skipped["no_bbox"].append(name)
                continue
            for cell in mgrs_tiles_for_bbox(bbox):
                needed[(cell, year)].add(name)
        finally:
            data.close()

    already = cached_pairs(vpp_cog_dir)
    missing = {pair: tiles for pair, tiles in needed.items() if pair not in already}

    by_year: dict[int, list[str]] = defaultdict(list)
    for cell, year in sorted(missing):
        by_year[year].append(cell)

    commands = [
        "python3 scripts/prefetch_vpp_wekeo.py "
        f"--mgrs-tiles {','.join(cells)} --years {year} "
        f"--dest-dir {vpp_cog_dir}"
        for year, cells in sorted(by_year.items())
    ]
    return {
        "data_dir": data_dir,
        "vpp_cog_dir": vpp_cog_dir,
        "tiles_scanned": len(paths),
        "tiles_missing_vpp": n_missing,
        "skipped": {k: sorted(v) for k, v in skipped.items()},
        "pairs_needed": len(needed),
        "pairs_already_cached": len(needed) - len(missing),
        "pairs_to_fetch": len(missing),
        "missing_by_year": {str(y): cells for y, cells in sorted(by_year.items())},
        "tiles_per_missing_pair": {
            f"{cell}:{year}": len(tiles) for (cell, year), tiles in sorted(missing.items())
        },
        "prefetch_commands": commands,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Derive missing (MGRS, year) VPP gap-fill set from tiles",
    )
    p.add_argument("--data-dir", required=True,
                   help="Tile dir to scan, e.g. /data/unified_v2_512_orphans_staging")
    p.add_argument("--vpp-cog-dir", default="/data/vpp_wekeo",
                   help="WEkEO COG cache to subtract (default /data/vpp_wekeo)")
    p.add_argument("--output", default=None,
                   help="Write the JSON report here as well as stdout")
    args = p.parse_args()

    report = derive(args.data_dir, args.vpp_cog_dir)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(out)
        print(f"\nreport → {out}", file=sys.stderr)

    print(
        f"\n{report['tiles_missing_vpp']}/{report['tiles_scanned']} tiles missing VPP; "
        f"{report['pairs_to_fetch']} (MGRS, year) pairs to fetch "
        f"({report['pairs_already_cached']} already cached)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
