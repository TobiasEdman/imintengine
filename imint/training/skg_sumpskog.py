"""Skogsstyrelsen sumpskog (swamp forest) densification regions.

Queries the Skogsstyrelsen ArcGIS REST service for swamp forest
(sumpskog) polygon density across Sweden, and returns densification
regions for grid cells with high concentrations.

This targets the under-represented ``forest_wetland_*`` LULC classes:
  - forest_wetland_deciduous (0.25 %)
  - forest_wetland_spruce    (0.37 %)
  - forest_wetland_mixed     (0.92 %)
  - forest_wetland_temp      (1.03 %)

Strategy:
  1. Overlay Sweden with a coarse scan grid (25 km).
  2. For each cell, query the ArcGIS endpoint for total sumpskog area.
  3. Cells exceeding a density threshold → densification regions.
  4. These regions are fed to :func:`sampler.densify_grid` for sub-grid
     generation at a finer spacing.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .sampler import generate_grid, filter_land_cells

# ── ArcGIS endpoint ─────────────────────────────────────────────────────

_ARCGIS_URL = (
    "https://geodpags.skogsstyrelsen.se/arcgis/rest/services/"
    "Geodataportal/GeodataportalVisaSumpskog/MapServer/0/query"
)

_SCAN_SPACING_M = 25_000           # 25 km coarse scan grid
_QUERY_TIMEOUT_S = 30              # HTTP timeout per request
_MAX_RETRIES = 2                   # retries on transient errors


# ── Public API ───────────────────────────────────────────────────────────

def generate_sumpskog_densification_regions(
    cache_dir: Path,
    *,
    min_density_pct: float = 3.0,
    scan_spacing_m: int = _SCAN_SPACING_M,
) -> list[dict]:
    """Scan Sweden for sumpskog-rich areas and return densification regions.

    Args:
        cache_dir: Directory for cached scan results.
        min_density_pct: Minimum sumpskog area as % of cell area
            to qualify as densification region (default: 3 %).
        scan_spacing_m: Spacing of the coarse scan grid (default: 25 km).

    Returns:
        List of dicts with ``label`` and ``bbox_3006`` keys,
        compatible with :func:`sampler.densify_grid`.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / "skg_sumpskog_scan.json"

    # Load from cache if exists
    if cache_path.exists():
        print(f"  Sumpskog scan cache hit: {cache_path}")
        scan_data = _load_scan_cache(cache_path)
    else:
        scan_data = _run_scan(cache_dir, cache_path, scan_spacing_m)

    # Filter cells by density threshold
    cell_area_ha = (scan_spacing_m ** 2) / 10_000
    regions: list[dict] = []
    half = scan_spacing_m // 2

    for entry in scan_data:
        density_pct = (entry["total_ha"] / cell_area_ha) * 100
        if density_pct >= min_density_pct:
            e, n = entry["easting"], entry["northing"]
            regions.append({
                "label": f"SKG:sumpskog_{e}_{n}",
                "bbox_3006": (
                    e - half,   # west
                    e + half,   # east
                    n - half,   # south
                    n + half,   # north
                ),
            })

    print(f"  Sumpskog: {len(regions)} regions (density ≥ {min_density_pct}%)"
          f" from {len(scan_data)} scanned cells")
    return regions


# ── Internal helpers ─────────────────────────────────────────────────────

def _load_scan_cache(cache_path: Path) -> list[dict]:
    """Load scan results from cached JSON."""
    with open(cache_path) as f:
        return json.load(f)


def _run_scan(
    cache_dir: Path,
    cache_path: Path,
    scan_spacing_m: int,
) -> list[dict]:
    """Query ArcGIS for sumpskog stats in each coarse grid cell."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate coarse scan grid (land cells only)
    # Use a 1-pixel patch_size for scan purposes — we only need centers
    cells = generate_grid(
        spacing_m=scan_spacing_m,
        patch_size_m=2_560,
        land_filter=True,
    )
    print(f"  Sumpskog scan: {len(cells)} land cells @ {scan_spacing_m/1000:.0f} km")

    half = scan_spacing_m // 2
    scan_data: list[dict] = []
    t_start = time.time()

    for i, cell in enumerate(cells):
        total_ha, count = _query_cell_stats(
            cell.easting, cell.northing, half,
        )
        scan_data.append({
            "easting": cell.easting,
            "northing": cell.northing,
            "count": count,
            "total_ha": round(total_ha, 1),
        })

        # Progress every 50 cells
        if (i + 1) % 50 == 0 or (i + 1) == len(cells):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(cells) - i - 1) / rate
            print(f"    [{i+1}/{len(cells)}] "
                  f"{rate:.1f} cells/s, ETA {eta:.0f}s")

    # Cache results
    tmp = cache_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(scan_data, f)
    tmp.rename(cache_path)
    print(f"  Saved scan results → {cache_path}")

    return scan_data


def _query_cell_stats(
    easting: int,
    northing: int,
    half: int,
) -> tuple[float, int]:
    """Query ArcGIS for sumpskog total area and count within a cell.

    Returns:
        (total_ha, feature_count)
    """
    envelope = json.dumps({
        "xmin": easting - half,
        "ymin": northing - half,
        "xmax": easting + half,
        "ymax": northing + half,
        "spatialReference": {"wkid": 3006},
    })

    params = {
        "where": "1=1",
        "geometry": envelope,
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "3006",
        "outStatistics": json.dumps([
            {
                "statisticType": "sum",
                "onStatisticField": "Areal",
                "outStatisticFieldName": "total_ha",
            },
            {
                "statisticType": "count",
                "onStatisticField": "OBJECTID",
                "outStatisticFieldName": "cnt",
            },
        ]),
        "f": "json",
    }

    url = _ARCGIS_URL + "?" + urllib.parse.urlencode(params)

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = urllib.request.urlopen(url, timeout=_QUERY_TIMEOUT_S)
            data = json.loads(resp.read())
            attrs = data.get("features", [{}])[0].get("attributes", {})
            total_ha = attrs.get("total_ha") or 0.0
            count = attrs.get("cnt") or 0
            return float(total_ha), int(count)
        except Exception:
            if attempt == _MAX_RETRIES:
                return 0.0, 0
            time.sleep(1.0 * (attempt + 1))

    return 0.0, 0
