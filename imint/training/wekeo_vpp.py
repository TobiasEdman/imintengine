"""WEkEO HR-VPP vegetation phenology fetching — CDSE fallback.

Fallback for :mod:`imint.training.cdse_vpp` when the CDSE Sentinel Hub
processing-unit quota is exhausted. Same HR-VPP source data (Copernicus
Land Monitoring Service), accessed via the WEkEO Harmonised Data Access
(HDA) API instead of the CDSE Sentinel Hub Process API.

Two stages — bulk-prefetch, then local read:

  1. :func:`prefetch_vpp_cogs` — bulk-download the per-MGRS-tile VPP COG
     GeoTIFFs to a local directory (run once, e.g. on a k8s pod). Writes
     an ``index.json`` recording each COG's EPSG:4326 footprint.
  2. :func:`fetch_vpp_tiles_local` — read an EPSG:3006 tile bbox out of
     the prefetched COGs. Identical return contract to
     :func:`cdse_vpp.fetch_vpp_tiles` — quota-independent once the COGs
     are on disk.

WEkEO HDA access::

    dataset_id : EO:EEA:DAT:CLMS_HRVPP_VPP   (UTM, Sentinel-2/MGRS tiling)
    query keys : productType, tileId, itemsPerPage, startIndex
    auth       : WEkEO credentials — WEKEO_USERNAME / WEKEO_PASSWORD env
                 vars, or the hda client's own ~/.hdarc

The UTM VPP product is the MGRS-tiled one, matching the Sentinel-2 tile
grid the CDSE BYOC collection is built on.

License: Copernicus Open Access
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

# ── WEkEO HDA config ─────────────────────────────────────────────────────
_HDA_DATASET_ID = "EO:EEA:DAT:CLMS_HRVPP_VPP"

# HDA productType -> result band name (lowercase, matches cdse_vpp keys).
_VPP_PRODUCT_TYPES: dict[str, str] = {
    "SOSD": "sosd",
    "EOSD": "eosd",
    "LENGTH": "length",
    "MAXV": "maxv",
    "MINV": "minv",
}

# PPI bands — INT16 stored, scale by 0.0001 to physical PPI (same as CDSE).
_PPI_BANDS = {"maxv", "minv"}

# HR-VPP VPP filename: VPP_<year>_S2_<tile>-010m_V<ver>_s<season>_<METRIC>.tif
_VPP_FILENAME_RE = re.compile(
    r"VPP_(?P<year>\d{4})_S2_(?P<tile>[0-9A-Z]+)-0?\d+m_"
    r"V\d+_s(?P<season>\d)_(?P<metric>[A-Z]+)\.tif$"
)

_INDEX_NAME = "index.json"


# ── Stage 1: bulk prefetch ───────────────────────────────────────────────

def prefetch_vpp_cogs(
    tile_ids: list[str],
    years: list[int],
    dest_dir: Path | str,
    *,
    season: int = 1,
    product_types: list[str] | None = None,
) -> dict[str, dict]:
    """Bulk-download HR-VPP VPP COGs from WEkEO HDA to ``dest_dir``.

    Args:
        tile_ids: Sentinel-2 MGRS tile identifiers, e.g. ``["33VWJ", ...]``.
        years: Product years to keep (e.g. ``[2021]``).
        dest_dir: Local directory for the COGs + ``index.json``.
        season: HR-VPP growing season (1 = primary). CDSE pipeline uses 1.
        product_types: HDA productTypes to fetch; defaults to the five the
            training pipeline needs (SOSD, EOSD, LENGTH, MAXV, MINV).

    Returns:
        The ``index.json`` content — ``{filename: {metric, tileId, year,
        season, bounds_4326}}`` — covering every COG now in ``dest_dir``.

    Idempotent: COGs already present are not re-downloaded.
    """
    from hda import Client, Configuration

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    types = product_types or list(_VPP_PRODUCT_TYPES)
    years_set = {int(y) for y in years}

    user = os.environ.get("WEKEO_USERNAME")
    password = os.environ.get("WEKEO_PASSWORD")
    config = (
        Configuration(user=user, password=password)
        if user and password
        else Configuration()  # falls back to ~/.hdarc
    )
    client = Client(config=config)

    index = _load_index(dest)
    for tile in tile_ids:
        for ptype in types:
            matches = client.search({
                "dataset_id": _HDA_DATASET_ID,
                "productType": ptype,
                "tileId": tile,
                "itemsPerPage": 200,
                "startIndex": 0,
            })
            for result in matches:
                fname = _result_filename(result)
                meta = _parse_vpp_filename(fname)
                if meta is None:
                    continue
                if meta["year"] not in years_set or meta["season"] != season:
                    continue
                target = dest / fname
                if not target.exists():
                    result.download(str(dest))
                if fname not in index:
                    index[fname] = {**meta, "bounds_4326": _cog_bounds_4326(target)}

    _save_index(dest, index)
    return index


# ── Stage 2: local read (CDSE-fallback path) ─────────────────────────────

def fetch_vpp_tiles_local(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    size_px: int | tuple[int, int] = 256,
    vpp_cog_dir: Path | str,
    year: int = 2021,
    season: int = 1,
) -> dict[str, np.ndarray]:
    """Read HR-VPP phenology for a tile bbox from prefetched WEkEO COGs.

    Drop-in replacement for :func:`cdse_vpp.fetch_vpp_tiles` — identical
    return contract — for use once :func:`prefetch_vpp_cogs` has populated
    ``vpp_cog_dir``.

    Args:
        west, south, east, north: Bounding box in EPSG:3006 (metres).
        size_px: Output size — int for square or (H, W) tuple.
        vpp_cog_dir: Directory populated by :func:`prefetch_vpp_cogs`.
        year: VPP product year.
        season: HR-VPP growing season (1 = primary).

    Returns:
        Dict mapping band names to (H, W) float32 arrays — ``sosd, eosd,
        length, maxv, minv``. PPI bands (maxv, minv) scaled to physical
        values; nodata pixels are 0.
    """
    h_px, w_px = (size_px, size_px) if isinstance(size_px, int) else size_px
    cog_dir = Path(vpp_cog_dir)
    index = _load_index(cog_dir)
    if not index:
        raise FileNotFoundError(
            f"No WEkEO VPP {_INDEX_NAME} in {cog_dir}. Run prefetch_vpp_cogs "
            f"first to populate the fallback cache."
        )

    bounds_4326 = _bbox_3006_to_4326(west, south, east, north)

    result: dict[str, np.ndarray] = {}
    for ptype, band in _VPP_PRODUCT_TYPES.items():
        cogs = [
            cog_dir / fname
            for fname, meta in index.items()
            if meta["metric"] == ptype
            and meta["year"] == year
            and meta["season"] == season
            and _bounds_overlap(meta["bounds_4326"], bounds_4326)
        ]
        arr = _read_mosaic_3006(
            cogs, west, south, east, north, h_px, w_px,
        )
        # Scaling — mirror cdse_vpp.fetch_vpp_tiles exactly.
        if band in _PPI_BANDS:
            arr = np.clip(arr * 0.0001, 0.0, None)
        else:
            arr = np.clip(arr, 0.0, None)
        result[band] = arr.astype(np.float32)

    return result


# ── Internal helpers ─────────────────────────────────────────────────────

def _bbox_3006_to_4326(
    west: float, south: float, east: float, north: float,
) -> tuple[float, float, float, float]:
    """Convert an EPSG:3006 bbox to EPSG:4326 (lon/lat) bounds."""
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    return transform_bounds(
        CRS.from_epsg(3006), CRS.from_epsg(4326), west, south, east, north,
    )


def _bounds_overlap(
    a: tuple[float, float, float, float] | list,
    b: tuple[float, float, float, float],
) -> bool:
    """True if two (w, s, e, n) bounds intersect."""
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _read_mosaic_3006(
    cogs: list[Path],
    west: float, south: float, east: float, north: float,
    h_px: int, w_px: int,
) -> np.ndarray:
    """Read + mosaic COGs into an (h_px, w_px) array on the EPSG:3006 bbox.

    Each COG is reprojected from its native UTM CRS straight onto the
    output grid — a WarpedVRT whose transform/size is exactly the
    EPSG:3006 bbox at (h_px, w_px) — with nearest resampling (phenology
    metrics are categorical/date-like — no interpolation). COGs are
    composited in order; later COGs only fill nodata gaps.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds as transform_from_bounds
    from rasterio.vrt import WarpedVRT

    out = np.zeros((h_px, w_px), dtype=np.float32)
    filled = np.zeros((h_px, w_px), dtype=bool)
    if not cogs:
        return out

    dst_transform = transform_from_bounds(west, south, east, north, w_px, h_px)
    for cog in cogs:
        with rasterio.open(cog) as src:
            with WarpedVRT(
                src, crs="EPSG:3006", transform=dst_transform,
                width=w_px, height=h_px, resampling=Resampling.nearest,
            ) as vrt:
                data = vrt.read(1).astype(np.float32)
                nodata = vrt.nodata
        valid = data != 0 if nodata is None else (data != nodata) & (data != 0)
        take = valid & ~filled
        out[take] = data[take]
        filled |= take
        if filled.all():
            break
    return out


def _parse_vpp_filename(fname: str) -> dict | None:
    """Parse a VPP COG filename into {metric, tileId, year, season}."""
    m = _VPP_FILENAME_RE.search(fname)
    if m is None:
        return None
    return {
        "metric": m.group("metric"),
        "tileId": m.group("tile"),
        "year": int(m.group("year")),
        "season": int(m.group("season")),
    }


def _result_filename(result) -> str:
    """Extract the product filename from an hda search result.

    The hda client's result objects expose the filename under different
    attributes across versions; check the known ones, then fall back to
    the string form.
    """
    for attr in ("results", "properties"):
        props = getattr(result, attr, None)
        if isinstance(props, dict):
            for key in ("title", "id", "filename"):
                val = props.get(key)
                if isinstance(val, str) and val.endswith(".tif"):
                    return os.path.basename(val)
    for key in ("title", "id", "filename"):
        val = getattr(result, key, None)
        if isinstance(val, str) and val.endswith(".tif"):
            return os.path.basename(val)
    return os.path.basename(str(result))


def _cog_bounds_4326(path: Path) -> list[float]:
    """Return a COG's footprint as EPSG:4326 (w, s, e, n) bounds."""
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(path) as src:
        return list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))


def _load_index(cog_dir: Path) -> dict[str, dict]:
    """Load the COG footprint index, or an empty dict if absent."""
    index_path = cog_dir / _INDEX_NAME
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return {}


def _save_index(cog_dir: Path, index: dict[str, dict]) -> None:
    """Atomically write the COG footprint index."""
    index_path = cog_dir / _INDEX_NAME
    tmp = index_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    tmp.rename(index_path)
