"""
imint/training/sampler.py — Geographic sampling grid across Sweden

Generates a uniform grid of training patch locations in EPSG:3006
(SWEREF99 TM) and provides train/val/test splitting by latitude.

The optional Sweden land mask (``sweden_land_epsg3006.json``) is derived
from Lantmäteriet's "Sverige 5 miljoner" vector dataset (SWEREF99 TM).
When available, grid cells whose centres fall in the sea are discarded
immediately — avoiding expensive NMD/spectral fetches for ocean points.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

# Sweden approximate bounding box in EPSG:3006 (SWEREF99 TM)
SWEDEN_WEST = 260_000
SWEDEN_EAST = 920_000
SWEDEN_SOUTH = 6_130_000
SWEDEN_NORTH = 7_670_000


@dataclass
class GridCell:
    """A single training patch location."""

    easting: int       # Center easting in EPSG:3006
    northing: int      # Center northing in EPSG:3006
    west_3006: int     # Bbox west
    east_3006: int     # Bbox east
    south_3006: int    # Bbox south
    north_3006: int    # Bbox north

    # WGS84 coordinates (set by grid_to_wgs84)
    west_wgs84: float = 0.0
    east_wgs84: float = 0.0
    south_wgs84: float = 0.0
    north_wgs84: float = 0.0
    center_lat: float = 0.0
    center_lon: float = 0.0


def _load_sweden_land_mask(
    geojson_path: str | Path | None = None,
) -> object | None:
    """Load Sweden land polygon from GeoJSON for fast point-in-polygon.

    The GeoJSON must be in EPSG:3006 (SWEREF99 TM).  The file is expected
    at ``data/sweden_land_epsg3006.json`` relative to the project root,
    unless an explicit path is provided.

    Returns:
        A shapely Geometry (union of all land polygons) or None if the
        file is not found or shapely is unavailable.
    """
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union
    except ImportError:
        return None

    if geojson_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        geojson_path = project_root / "data" / "sweden_land_epsg3006.json"

    geojson_path = Path(geojson_path)
    if not geojson_path.exists():
        return None

    with open(geojson_path) as f:
        data = json.load(f)

    polys = []
    for feature in data["features"]:
        polys.append(shape(feature["geometry"]))

    land = unary_union(polys)
    return land


# Module-level cache for the land mask
_SWEDEN_LAND_MASK = None


def _get_land_mask() -> object | None:
    """Get the cached Sweden land mask (loads once on first call)."""
    global _SWEDEN_LAND_MASK
    if _SWEDEN_LAND_MASK is None:
        _SWEDEN_LAND_MASK = _load_sweden_land_mask()
    return _SWEDEN_LAND_MASK


def filter_land_cells(
    cells: list[GridCell],
    return_sea_cells: bool = False,
) -> list[GridCell] | tuple[list[GridCell], list[GridCell]]:
    """Remove grid cells whose centres fall outside Sweden's land area.

    Uses the "Sverige 5 miljoner" vector dataset as a fast land mask.
    Grid cell centres are tested against the EPSG:3006 polygon — no
    coordinate conversion needed.

    If the land mask is unavailable (file missing or no shapely),
    returns the input unchanged.

    Args:
        cells: Grid cells with EPSG:3006 coordinates.
        return_sea_cells: If True, also return cells that were filtered
            out (centres in the sea).

    Returns:
        If *return_sea_cells* is False (default): list of land cells.
        If True: ``(land_cells, sea_cells)`` tuple.
    """
    land = _get_land_mask()
    if land is None:
        print("  WARNING: Sweden land mask not found — skipping land filter")
        if return_sea_cells:
            return cells, []
        return cells

    from shapely.geometry import Point
    from shapely import prepared

    # Use prepared geometry for fast repeated contains() checks
    prep_land = prepared.prep(land)

    land_cells = []
    sea_cells = []
    for cell in cells:
        pt = Point(cell.easting, cell.northing)
        if prep_land.contains(pt):
            land_cells.append(cell)
        elif return_sea_cells:
            sea_cells.append(cell)

    print(f"  Land filter: {len(land_cells)}/{len(cells)} cells on land "
          f"({len(cells) - len(land_cells)} in sea)")

    if return_sea_cells:
        return land_cells, sea_cells
    return land_cells


def generate_grid(
    spacing_m: int = 10_000,
    patch_size_m: int = 2_240,
    land_filter: bool = True,
) -> list[GridCell]:
    """Generate a regular grid of patch locations across Sweden.

    Args:
        spacing_m: Distance between grid centers in meters.
        patch_size_m: Side length of each patch in meters (224px * 10m).
        land_filter: If True, remove cells in the sea using Sweden's
            land polygon. Default True.

    Returns:
        List of GridCell instances covering Sweden.
    """
    half = patch_size_m // 2
    cells = []

    e = SWEDEN_WEST + half
    while e + half <= SWEDEN_EAST:
        n = SWEDEN_SOUTH + half
        while n + half <= SWEDEN_NORTH:
            cells.append(GridCell(
                easting=e,
                northing=n,
                west_3006=e - half,
                east_3006=e + half,
                south_3006=n - half,
                north_3006=n + half,
            ))
            n += spacing_m
        e += spacing_m

    if land_filter:
        cells = filter_land_cells(cells)

    return cells


def grid_to_wgs84(cells: list[GridCell]) -> list[GridCell]:
    """Convert EPSG:3006 bounding boxes to WGS84 (EPSG:4326).

    Uses a simplified inverse SWEREF99 TM projection (accurate to ~10m).
    For training-data sampling, this precision is sufficient.

    Args:
        cells: Grid cells with EPSG:3006 coordinates.

    Returns:
        Same cells with WGS84 fields populated.
    """
    for cell in cells:
        sw = _sweref99_to_wgs84(cell.east_3006, cell.south_3006)
        ne = _sweref99_to_wgs84(cell.west_3006, cell.north_3006)
        center = _sweref99_to_wgs84(cell.easting, cell.northing)

        # Note: in SWEREF99 TM easting maps to longitude, but the
        # west/east bbox edges may swap. Use min/max to be safe.
        cell.west_wgs84 = min(sw[1], ne[1])
        cell.east_wgs84 = max(sw[1], ne[1])
        cell.south_wgs84 = min(sw[0], ne[0])
        cell.north_wgs84 = max(sw[0], ne[0])
        cell.center_lat = center[0]
        cell.center_lon = center[1]

    return cells


def split_by_latitude(
    cells: list[GridCell],
    val_lat_min: float = 64.0,
    val_lat_max: float = 66.0,
    test_lat_min: float = 66.0,
) -> tuple[list[GridCell], list[GridCell], list[GridCell]]:
    """Split grid cells into train/val/test by WGS84 latitude.

    Args:
        cells: Grid cells with WGS84 coordinates populated.
        val_lat_min: Southern boundary of validation zone.
        val_lat_max: Northern boundary of validation zone.
        test_lat_min: Southern boundary of test zone.

    Returns:
        (train, val, test) lists of GridCell.
    """
    train, val, test = [], [], []
    for cell in cells:
        lat = cell.center_lat
        if lat >= test_lat_min:
            test.append(cell)
        elif lat >= val_lat_min:
            val.append(cell)
        else:
            train.append(cell)

    return train, val, test


# ── Grid densification ─────────────────────────────────────────────────────

# Predefined regions with rare NMD classes (EPSG:3006 bounding boxes)
DENSIFICATION_REGIONS = [
    # Major cities (developed classes: buildings, infrastructure, roads)
    {"label": "Stockholm",   "bbox_3006": (665_000, 695_000, 6_560_000, 6_600_000)},
    {"label": "Gothenburg",  "bbox_3006": (310_000, 340_000, 6_390_000, 6_420_000)},
    {"label": "Malmö",       "bbox_3006": (370_000, 400_000, 6_155_000, 6_185_000)},
    {"label": "Uppsala",     "bbox_3006": (640_000, 665_000, 6_620_000, 6_650_000)},
    {"label": "Linköping",   "bbox_3006": (500_000, 530_000, 6_475_000, 6_505_000)},
    {"label": "Örebro",      "bbox_3006": (480_000, 510_000, 6_545_000, 6_570_000)},
    {"label": "Umeå",        "bbox_3006": (720_000, 750_000, 7_070_000, 7_095_000)},
    {"label": "Luleå",       "bbox_3006": (790_000, 820_000, 7_275_000, 7_300_000)},
    # Southern deciduous belt (deciduous forest, cropland, varied land use)
    {"label": "Skåne",       "bbox_3006": (350_000, 480_000, 6_130_000, 6_230_000)},
    {"label": "Blekinge",    "bbox_3006": (430_000, 530_000, 6_215_000, 6_265_000)},
    {"label": "Halland",     "bbox_3006": (310_000, 380_000, 6_260_000, 6_390_000)},
    # Coastal strips (sea water, coastal habitats)
    {"label": "West coast",  "bbox_3006": (270_000, 320_000, 6_300_000, 6_530_000)},
    {"label": "Gotland",     "bbox_3006": (630_000, 690_000, 6_350_000, 6_450_000)},
    {"label": "Öland",       "bbox_3006": (510_000, 540_000, 6_280_000, 6_360_000)},
    # Mountain/alpine areas (bare land, open vegetated)
    {"label": "Alpine north","bbox_3006": (370_000, 520_000, 7_300_000, 7_600_000)},
    {"label": "Alpine mid",  "bbox_3006": (380_000, 480_000, 6_950_000, 7_150_000)},
]


def generate_densification_regions() -> list[dict]:
    """Return the predefined densification regions.

    These are areas in Sweden known to contain rare NMD classes
    (urban, coastal, deciduous, alpine).

    Returns:
        List of dicts with 'label' and 'bbox_3006' keys.
    """
    return DENSIFICATION_REGIONS


def densify_grid(
    base_cells: list[GridCell],
    densification_regions: list[dict] | None = None,
    densify_spacing_m: int = 5_000,
    patch_size_m: int = 2_240,
    min_distance_m: int = 2_000,
) -> list[GridCell]:
    """Add extra grid points in regions rich in rare classes.

    Generates a finer sub-grid within each densification region and
    merges with the base grid, deduplicating points that are too close.

    Args:
        base_cells: Original uniform grid cells.
        densification_regions: List of region dicts. Uses defaults if None.
        densify_spacing_m: Spacing for the finer sub-grid (default: 5km).
        patch_size_m: Patch size in meters.
        min_distance_m: Minimum distance between any two grid centers
            (to avoid near-duplicate tiles).

    Returns:
        Merged list of GridCell (original + densified).
    """
    if densification_regions is None:
        densification_regions = DENSIFICATION_REGIONS

    # Build spatial index of existing centers for deduplication
    existing = set()
    for cell in base_cells:
        # Round to nearest km for quick lookup
        key = (cell.easting // 1000, cell.northing // 1000)
        existing.add(key)

    half = patch_size_m // 2
    new_cells = []

    for region in densification_regions:
        west, east, south, north = region["bbox_3006"]

        # Clip to Sweden bounds
        west = max(west, SWEDEN_WEST + half)
        east = min(east, SWEDEN_EAST - half)
        south = max(south, SWEDEN_SOUTH + half)
        north = min(north, SWEDEN_NORTH - half)

        if west >= east or south >= north:
            continue

        e = west + half
        while e + half <= east:
            n = south + half
            while n + half <= north:
                key = (e // 1000, n // 1000)

                # Check minimum distance to existing points
                too_close = False
                for de in range(-2, 3):
                    for dn in range(-2, 3):
                        if (key[0] + de, key[1] + dn) in existing:
                            too_close = True
                            break
                    if too_close:
                        break

                if not too_close:
                    new_cells.append(GridCell(
                        easting=e,
                        northing=n,
                        west_3006=e - half,
                        east_3006=e + half,
                        south_3006=n - half,
                        north_3006=n + half,
                    ))
                    existing.add(key)

                n += densify_spacing_m
            e += densify_spacing_m

    return base_cells + new_cells


# ── Coordinate conversion ─────────────────────────────────────────────────

def _sweref99_to_wgs84(easting: float, northing: float) -> tuple[float, float]:
    """Convert SWEREF99 TM (EPSG:3006) to WGS84 (lat, lon).

    Simplified inverse Transverse Mercator. Accurate to ~10m across Sweden.
    """
    # SWEREF99 TM parameters
    a = 6_378_137.0             # Semi-major axis (GRS80)
    f = 1 / 298.257222101       # Flattening
    lat0 = 0.0                  # Origin latitude (radians)
    lon0 = math.radians(15.0)   # Central meridian 15E
    k0 = 0.9996                 # Scale factor
    fn = 0.0                    # False northing
    fe = 500_000.0              # False easting

    e2 = 2 * f - f * f
    e_prime2 = e2 / (1 - e2)
    n = f / (2 - f)
    n2, n3, n4 = n * n, n * n * n, n * n * n * n

    # Meridional arc
    A = (a / (1 + n)) * (1 + n2 / 4 + n4 / 64)

    xi = (northing - fn) / (k0 * A)
    eta = (easting - fe) / (k0 * A)

    # Coefficients for inverse
    d1 = n / 2 - 2 * n2 / 3 + 37 * n3 / 96
    d2 = n2 / 48 + n3 / 15
    d3 = 17 * n3 / 480

    xi_prime = xi - (
        d1 * math.sin(2 * xi) * math.cosh(2 * eta)
        + d2 * math.sin(4 * xi) * math.cosh(4 * eta)
        + d3 * math.sin(6 * xi) * math.cosh(6 * eta)
    )
    eta_prime = eta - (
        d1 * math.cos(2 * xi) * math.sinh(2 * eta)
        + d2 * math.cos(4 * xi) * math.sinh(4 * eta)
        + d3 * math.cos(6 * xi) * math.sinh(6 * eta)
    )

    chi = math.asin(math.sin(xi_prime) / math.cosh(eta_prime))

    lat = chi + (
        (e2 / 2 + 5 * e2 ** 2 / 24 + e2 ** 3 / 12) * math.sin(2 * chi)
        + (7 * e2 ** 2 / 48 + 29 * e2 ** 3 / 240) * math.sin(4 * chi)
        + (7 * e2 ** 3 / 120) * math.sin(6 * chi)
    )
    lon = lon0 + math.atan2(math.sinh(eta_prime), math.cos(xi_prime))

    return math.degrees(lat), math.degrees(lon)


# ── Sea cell filtering (Swedish waters) ──────────────────────────────────

_SFV_TERRITORIAL_URL = (
    "https://www.sjofartsverket.se/globalassets/tjanster/sjokort/"
    "sjoterritoriets_gransterritorialgrans.zip"
)

_SWEDISH_TERRITORIAL_WATERS = None


def _wgs84_to_sweref99(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Convert WGS84 (lat, lon) to SWEREF99 TM (easting, northing).

    Simplified forward Transverse Mercator, accurate to ~10 m.
    """
    a = 6_378_137.0
    f = 1 / 298.257222101
    lon0 = math.radians(15.0)
    k0 = 0.9996
    fe = 500_000.0

    e2 = 2 * f - f * f
    n = f / (2 - f)
    n2, n3, n4 = n * n, n ** 3, n ** 4
    A = (a / (1 + n)) * (1 + n2 / 4 + n4 / 64)

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    t = math.sinh(
        math.atanh(math.sin(lat))
        - 2 * math.sqrt(e2) / (1 + e2)
        * math.atanh(2 * math.sqrt(e2) / (1 + e2) * math.sin(lat))
    )
    xi_p = math.atan2(t, math.cos(lon - lon0))
    eta_p = math.atanh(math.sin(lon - lon0) / math.sqrt(1 + t * t))

    a1 = n / 2 - 2 * n2 / 3 + 5 * n3 / 16
    a2 = 13 * n2 / 48 - 3 * n3 / 5
    a3 = 61 * n3 / 240

    northing = k0 * A * (
        xi_p
        + a1 * math.sin(2 * xi_p) * math.cosh(2 * eta_p)
        + a2 * math.sin(4 * xi_p) * math.cosh(4 * eta_p)
        + a3 * math.sin(6 * xi_p) * math.cosh(6 * eta_p)
    )
    easting = fe + k0 * A * (
        eta_p
        + a1 * math.cos(2 * xi_p) * math.sinh(2 * eta_p)
        + a2 * math.cos(4 * xi_p) * math.sinh(4 * eta_p)
        + a3 * math.cos(6 * xi_p) * math.sinh(6 * eta_p)
    )
    return easting, northing


def _build_territorial_waters(cache_dir: Path) -> object | None:
    """Build Swedish territorial waters polygon from Sjöfartsverket data.

    Downloads the official territorial boundary point shapefile, converts
    the turning-point sequences to polygons in EPSG:3006, and subtracts
    the land mask to produce a territorial-waters-only geometry.

    The result is cached as GeoJSON for fast reloads.
    """
    try:
        import shapefile as pyshp
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError as exc:
        print(f"  WARNING: missing dependency for territorial waters: {exc}")
        return None

    cache_dir = Path(cache_dir)
    geojson_cache = cache_dir / "swedish_territorial_waters.json"

    # Fast path: load from cache
    if geojson_cache.exists():
        from shapely.geometry import shape
        with open(geojson_cache) as f:
            data = json.load(f)
        polys = [shape(feat["geometry"]) for feat in data["features"]]
        return unary_union(polys)

    # ── Download shapefile ────────────────────────────────────────────
    zip_path = cache_dir / "territorialgrans.zip"
    if not zip_path.exists():
        import urllib.request
        print("  Downloading Sjöfartsverket territorial boundary …")
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = zip_path.with_suffix(".zip.tmp")
        urllib.request.urlretrieve(_SFV_TERRITORIAL_URL, tmp)
        tmp.rename(zip_path)
        print(f"  Saved to {zip_path}")

    # ── Extract ───────────────────────────────────────────────────────
    import zipfile
    extract_dir = cache_dir / "territorialgrans"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    # ── Find the polygon shapefile (not _linje) ──────────────────────
    import glob as _glob

    shp_files = _glob.glob(
        str(extract_dir / "**" / "*.shp"), recursive=True,
    )
    shp_path = next(
        (f for f in shp_files if "_linje" not in f), None,
    )
    if shp_path is None:
        print("  WARNING: territorial boundary .shp not found")
        return None

    # ── Read boundary points ──────────────────────────────────────────
    from collections import defaultdict

    sf = pyshp.Reader(shp_path, encoding="latin-1")
    segments: dict[str, list] = defaultdict(list)
    for rec, _shape in zip(sf.iterRecords(), sf.iterShapes()):
        seg = rec["Delsträcka"]
        nr = rec["Löpnr_dels"]
        lon = rec["Longitud_d"]
        lat = rec["Latitud_de"]
        segments[seg].append((nr, lon, lat))

    for seg in segments:
        segments[seg].sort(key=lambda x: x[0])

    # ── Build polygons in EPSG:3006 ───────────────────────────────────
    polys = []
    for _seg_name, pts in segments.items():
        coords = [_wgs84_to_sweref99(lat, lon) for _, lon, lat in pts]
        coords.append(coords[0])  # close ring
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        polys.append(poly)

    territorial_zone = unary_union(polys)

    # ── Subtract land → territorial waters ────────────────────────────
    land = _get_land_mask()
    if land is not None:
        territorial_waters = territorial_zone.difference(land)
    else:
        territorial_waters = territorial_zone

    print(f"  Territorial waters: {territorial_waters.area / 1e6:.0f} km²")

    # ── Cache as GeoJSON ──────────────────────────────────────────────
    from shapely.geometry import mapping

    features = []
    if territorial_waters.geom_type == "MultiPolygon":
        for geom in territorial_waters.geoms:
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {},
            })
    else:
        features.append({
            "type": "Feature",
            "geometry": mapping(territorial_waters),
            "properties": {},
        })

    geojson = {"type": "FeatureCollection", "features": features}
    tmp = geojson_cache.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(geojson, f)
    tmp.rename(geojson_cache)
    print(f"  Cached territorial waters → {geojson_cache}")

    return territorial_waters


def _get_territorial_waters(cache_dir: Path | None = None) -> object | None:
    """Get the cached territorial waters mask (loads once)."""
    global _SWEDISH_TERRITORIAL_WATERS
    if _SWEDISH_TERRITORIAL_WATERS is None:
        if cache_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            cache_dir = project_root / "data" / "cache"
        _SWEDISH_TERRITORIAL_WATERS = _build_territorial_waters(cache_dir)
    return _SWEDISH_TERRITORIAL_WATERS


def filter_sea_cells_swedish_waters(
    sea_cells: list[GridCell],
    max_distance_m: int = 5_000,
    cache_dir: Path | None = None,
) -> list[GridCell]:
    """Keep only sea cells that are in Swedish waters near the coast.

    Two-step filter:

    1. **Distance** — cell centre must be within *max_distance_m* of
       Swedish land (EPSG:3006 distance in meters).
    2. **Territorial** — cell centre must fall inside the Swedish
       territorial waters polygon derived from Sjöfartsverket's
       official boundary data.

    This prevents cells in Norwegian, Danish, or Finnish waters.

    Args:
        sea_cells: Cells whose centres are NOT on Swedish land.
        max_distance_m: Maximum distance from land in meters.
        cache_dir: Cache directory for downloaded/cached data.

    Returns:
        Filtered list of sea cells in Swedish coastal waters.
    """
    if not sea_cells:
        return []

    land = _get_land_mask()
    if land is None:
        print("  WARNING: land mask unavailable — returning all sea cells")
        return sea_cells

    from shapely.geometry import Point

    # Step 1: distance from Swedish land
    near_coast = []
    for cell in sea_cells:
        pt = Point(cell.easting, cell.northing)
        if land.distance(pt) <= max_distance_m:
            near_coast.append(cell)

    print(f"  Sea distance filter: {len(near_coast)}/{len(sea_cells)} "
          f"within {max_distance_m / 1000:.0f} km of land")

    # Step 2: Swedish territorial waters (Sjöfartsverket boundary)
    tw = _get_territorial_waters(cache_dir)
    if tw is None:
        print("  WARNING: territorial waters unavailable — skipping filter")
        return near_coast

    from shapely import prepared

    prep_tw = prepared.prep(tw)
    swedish = []
    for cell in near_coast:
        pt = Point(cell.easting, cell.northing)
        if prep_tw.contains(pt):
            swedish.append(cell)

    print(f"  Territorial filter: {len(swedish)}/{len(near_coast)} "
          f"in Swedish waters")
    return swedish
