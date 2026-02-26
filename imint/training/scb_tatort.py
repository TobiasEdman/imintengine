"""SCB Tätort urban densification regions.

Downloads and caches SCB (Statistics Sweden) locality polygon data
from WFS and converts to densification regions compatible with
:func:`sampler.densify_grid`.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from shapely.geometry import shape


# ── WFS endpoint ──────────────────────────────────────────────────────────

SCB_WFS_URL = (
    "https://geodata.scb.se/geoserver/stat/wfs"
    "?service=WFS&REQUEST=GetFeature&version=1.1.0"
    "&TYPENAMES=stat:Tatorter_2018&outputFormat=application/json"
)


# ── Public API ────────────────────────────────────────────────────────────

def generate_scb_densification_regions(
    cache_dir: Path,
    min_population: int = 2_000,
    patch_size_m: int = 2_240,
) -> list[dict]:
    """Download (if needed) and return SCB tätort densification regions.

    Args:
        cache_dir: Directory for cached GeoJSON file.
        min_population: Minimum population for tätort inclusion.
        patch_size_m: Minimum bbox dimension (for padding small tätorter).

    Returns:
        List of dicts with ``label`` and ``bbox_3006`` keys,
        compatible with :func:`sampler.densify_grid`.
    """
    geojson_path = fetch_scb_tatort_geojson(cache_dir)
    return load_tatort_regions(geojson_path, min_population, patch_size_m)


def fetch_scb_tatort_geojson(cache_dir: Path) -> Path:
    """Download SCB Tätorter 2018 GeoJSON, caching locally.

    Returns:
        Path to the cached file.  Skips download if file exists.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / "scb_tatorter_2018.json"

    if cache_path.exists():
        print(f"  SCB tätort cache hit: {cache_path}")
        return cache_path

    print("  Downloading SCB Tätorter 2018 from WFS …")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Atomic download: write to tmp, then rename
    tmp_path = cache_path.with_suffix(".json.tmp")
    urllib.request.urlretrieve(SCB_WFS_URL, tmp_path)
    tmp_path.rename(cache_path)
    print(f"  Saved to {cache_path}")
    return cache_path


def load_tatort_regions(
    geojson_path: Path,
    min_population: int = 2_000,
    patch_size_m: int = 2_240,
) -> list[dict]:
    """Parse SCB tätort GeoJSON into densification region dicts.

    Filters by population.  Pads small bboxes so that
    :func:`sampler.densify_grid` generates at least one cell per tätort.

    Args:
        geojson_path: Path to cached SCB GeoJSON file.
        min_population: Minimum population threshold for inclusion.
        patch_size_m: Patch size in meters — used for bbox padding.

    Returns:
        List of region dicts with ``label`` (str) and
        ``bbox_3006`` (west, east, south, north) tuple.
    """
    with open(geojson_path) as f:
        data = json.load(f)

    min_bbox_size = patch_size_m  # densify_grid needs >= patch_size_m

    regions: list[dict] = []
    for feature in data["features"]:
        props = feature["properties"]
        pop = props.get("bef", 0) or 0
        if pop < min_population:
            continue

        name = props.get("tatort", "unknown")

        # Bbox from polygon geometry (EPSG:3006)
        geom = shape(feature["geometry"])
        minx, miny, maxx, maxy = geom.bounds

        # Pad if smaller than minimum for cell generation
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2

        if (maxx - minx) < min_bbox_size:
            half_w = min_bbox_size / 2
            minx = cx - half_w
            maxx = cx + half_w

        if (maxy - miny) < min_bbox_size:
            half_h = min_bbox_size / 2
            miny = cy - half_h
            maxy = cy + half_h

        regions.append({
            "label": f"SCB:{name}",
            "bbox_3006": (int(minx), int(maxx), int(miny), int(maxy)),
        })

    print(f"  SCB tätort: {len(regions)} regions (pop ≥ {min_population})")
    return regions
