"""Generate a Sentinel-1 radar sub-tab of the marine showcase (one area per run).

Two presets (``--area``), both under the "⚓ Marin" tab:

* ``skagen``: the Skagen anchorage (Danish waters).
* ``oresund``: the "Sjöfart — Öresund" sub-tab's own area (``COORDS`` of
  ``generate_marine_commercial_showcase.py``), so the radar sub-tab lines up
  with the optical one pixel for pixel.

Each preset pins its target and historical Sentinel-1 IW GRDH scenes. They
are fetched from Microsoft Planetary Computer (``imint.fetch.fetch_s1_vessel_scenes``:
free and anonymous, no PU billing, no requester-pays) with a margin around the
display extent. The margin is chosen so that AI2's UTM model grid covers the
whole EPSG:3006 display grid, and so that the detector and the 128 px
attribute windows see real radar at the edges. The ``ai2_vessels_s1``
analyzer runs at ``CANDIDATE_THRESHOLD`` (0.3). Each detection is labelled
against AI2's production threshold, ``SCORE_THRESHOLD`` (0.7). Only detections
inside the display extent are kept.

Per area, the display grid is ``imint.fetch._to_nmd_grid(coords)`` (EPSG:3006,
10 m), the convention of the rest of the showcase. The script writes:

* ``rgb.png``: the target scene's VV.
* ``baseline.png``: the historical scene's VV, with the same fixed dB stretch.
* ``sjokort.png``: a nautical chart on the same grid. Öresund uses the
  Sjöfartsverket S-57 cells rendered by ``imint.fetch.render_sjokort_png``.
  Skagen uses OpenSeaMap seamarks over the OSM basemap. Those tiles are read
  from the marine-awareness repository's committed offline tile trees, through
  ``git show``, so no tile server is contacted.
* The detections in display pixel space.
* ``showcase_summary.json`` and ``MANIFEST.json``.

A detection is a candidate, not a confirmed vessel: no AIS is matched here.

Usage (pystac-client + planetary-computer must be importable)::

    .venv/bin/python scripts/generate_marine_radar_showcase.py --area skagen
    .venv/bin/python scripts/generate_marine_radar_showcase.py --area oresund \\
        --enc-dir outputs/sjokort_oresund/unique_enc
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from imint.analyzers.ai2_vessels_s1 import SCORE_THRESHOLD
from imint.fetch import S1_VESSEL_SOURCE
from scripts.generate_marine_commercial_showcase import COORDS as ORESUND_COORDS

DISPLAY_CRS = "EPSG:3006"
DISPLAY_GSD_M = 10
# Fixed display stretch of VV in dB of the uncalibrated GRD amplitude
# (20·log10 DN), shared by both areas and by target and historical scenes so
# that Efter/Före compare like for like. Open water is ~29–36 dB, land ~45–57 dB:
# water renders dark grey, land keeps its texture and ships stand out white.
VV_DB_RANGE = (28.0, 58.0)
# The analyzer runs at this threshold; detections below SCORE_THRESHOLD (AI2's
# production threshold, 0.7) are shown as candidates.
CANDIDATE_THRESHOLD = 0.3
OSM_ATTRIBUTION = ("© OpenStreetMap contributors (ODbL), seamarks © OpenSeaMap (CC BY-SA)")
OSM_BASE_ZOOM = 12      # deepest basemap zoom in marine-awareness (~20 m/px at 57.7° N)
OSM_SEAMARK_ZOOM = 13   # seamark overlay, ~10 m/px at 57.7° N
WEB_MERCATOR_HALF = 20037508.342789244


@dataclasses.dataclass(frozen=True)
class Area:
    key: str
    name: str
    coords: dict
    target_date: str
    target_scene_id: str
    historical_scene_id: str
    margin_m: float   # fetch margin: AI2's UTM grid must cover the whole display grid
    chart: str        # "s57" | "osm"


AREAS = {
    "skagen": Area(
        key="skagen", name="Skagen anchorage",
        coords={"west": 10.58, "south": 57.64, "east": 10.95, "north": 57.82},
        target_date="2026-06-28",
        target_scene_id="S1D_IW_GRDH_1SDV_20260628T053135_20260628T053200_003432_0060CF",
        historical_scene_id="S1D_IW_GRDH_1SDV_20260523T053133_20260523T053158_002907_004F80",
        # EPSG:3006 is ~3.6° rotated against UTM 32N here; 1000 m leaves 0.4 %
        # of the display uncovered, 1500 m none.
        margin_m=1500.0, chart="osm"),
    "oresund": Area(
        key="oresund", name="Öresund, Helsingborg–Helsingør",
        coords=dict(ORESUND_COORDS),
        target_date="2025-07-19",
        target_scene_id="S1C_IW_GRDH_1SDV_20250719T165231_20250719T165256_003292_006A41",
        historical_scene_id="S1C_IW_GRDH_1SDV_20250601T165228_20250601T165253_002592_005606",
        margin_m=1000.0, chart="s57"),
}


def expand_bbox(coords: dict, margin_m: float) -> dict:
    """WGS84 bbox grown by ``margin_m`` on every side (local metres per degree)."""
    lat_c = math.radians((coords["south"] + coords["north"]) / 2)
    dlat = margin_m / 111_320.0
    dlon = margin_m / (111_320.0 * math.cos(lat_c))
    return {"west": round(coords["west"] - dlon, 6), "south": round(coords["south"] - dlat, 6),
            "east": round(coords["east"] + dlon, 6), "north": round(coords["north"] + dlat, 6)}


# ── Display grid (the showcase's NMD-snapped EPSG:3006 grid) ─────────────

def display_grid(coords: dict):
    """(affine transform, (height, width), snapped bounds) of the display grid."""
    from rasterio.transform import from_origin

    from imint.fetch import _to_nmd_grid

    g = _to_nmd_grid(coords)
    if g["crs"] != DISPLAY_CRS:
        raise ValueError(f"expected {DISPLAY_CRS}, got {g['crs']}")
    width = round((g["east"] - g["west"]) / DISPLAY_GSD_M)
    height = round((g["north"] - g["south"]) / DISPLAY_GSD_M)
    return from_origin(g["west"], g["north"], DISPLAY_GSD_M, DISPLAY_GSD_M), (height, width), g


def reproject_to_display(band: np.ndarray, src_transform, src_crs: str, dst_transform,
                         shape: tuple, src_nodata=0) -> np.ndarray:
    """One band onto the display grid (bilinear); NaN = no data."""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    out = np.full(shape, np.nan, dtype=np.float32)
    reproject(source=np.asarray(band, dtype=np.float32), destination=out,
              src_transform=src_transform, src_crs=src_crs, src_nodata=src_nodata,
              dst_transform=dst_transform, dst_crs=DISPLAY_CRS, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    return out


def save_vv_db_png(vv_dn: np.ndarray, path: Path) -> dict:
    """Grey+alpha VV image: 20·log10(DN) dB with the fixed ``VV_DB_RANGE`` stretch.

    No-data (NaN or DN ≤ 0) is written fully transparent.
    """
    from PIL import Image

    valid = np.isfinite(vv_dn) & (vv_dn > 0)
    db = np.full(vv_dn.shape, np.nan, dtype=np.float32)
    db[valid] = 20.0 * np.log10(vv_dn[valid])
    lo, hi = VV_DB_RANGE
    grey = np.zeros(vv_dn.shape, dtype=np.uint8)
    grey[valid] = np.round(np.clip((db[valid] - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    Image.fromarray(np.dstack([grey, alpha]), mode="LA").save(path, optimize=True)
    print(f"    saved: {path}")
    pct = np.percentile(db[valid], [1, 50, 99]) if valid.any() else [np.nan] * 3
    return {"valid_frac": round(float(valid.mean()), 6),
            "db_p01_p50_p99": [round(float(v), 2) for v in pct],
            "clipped_low_frac": round(float((db[valid] < lo).mean()), 4),
            "clipped_high_frac": round(float((db[valid] > hi).mean()), 4)}


# ── Nautical charts on the display grid ───────────────────────────────────

def render_s57_chart(enc_dir: Path, coords: dict, bounds: dict, shape: tuple,
                     path: Path) -> dict:
    """Sjöfartsverket S-57 cells rendered on the display grid.

    ``render_sjokort_png`` takes its projected bounds from a ``bands_meta.json``
    beside the output, and without one falls back to the projected WGS84
    corners, which are not the snapped grid (Öresund: ~300 m off at N and S).
    So the chart is rendered in a temporary directory that carries the
    display grid's bounds, then moved into place.
    """
    from PIL import Image

    from imint.fetch import render_sjokort_png

    cells = sorted(enc_dir.glob("*.000"))
    if not cells:
        raise FileNotFoundError(f"no S-57 cells (*.000) in {enc_dir}")
    height, width = shape
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "bands").mkdir()
        meta = {"geo": {"crs": DISPLAY_CRS, "bounds_projected":
                        {k: bounds[k] for k in ("west", "south", "east", "north")}}}
        (tmp / "bands" / "display_bands_meta.json").write_text(json.dumps(meta))
        render_sjokort_png(cells, dict(coords), tmp / "sjokort.png", width, height)
        with Image.open(tmp / "sjokort.png") as im:
            if im.size != (width, height):
                raise RuntimeError(f"chart is {im.size}, display grid is {(width, height)}")
        shutil.move(tmp / "sjokort.png", path)
    print(f"    saved: {path}")
    return {"source": "Sjöfartsverket S-57 ENC cells via SLU GET",
            "attribution": "Sjökortsdata © Sjöfartsverket",
            "renderer": "imint.fetch.render_sjokort_png",
            "cells": [{"file": c.name, "sha256": sha256_file(c)} for c in cells]}


def _tile_x(lon: float, z: int) -> int:
    return int((lon + 180.0) / 360.0 * 2 ** z)


def _tile_y(lat: float, z: int) -> int:
    r = math.radians(lat)
    return int((1.0 - math.asinh(math.tan(r)) / math.pi) / 2.0 * 2 ** z)


def render_osm_chart(repo: Path, ref: str, bounds: dict, dst_transform, shape: tuple,
                     path: Path) -> dict:
    """OpenSeaMap seamarks over the OSM basemap, reprojected onto the display grid.

    The tiles come from marine-awareness's committed offline trees
    (``dashboard/tiles/{z}/{x}/{y}.png`` and ``dashboard/tiles_seamark/...``),
    read through ``git show <commit>:<path>``: read-only, and no tile server
    is contacted. The basemap (z12) is upsampled 2x onto the seamark zoom
    (z13), the seamarks are alpha-composited over it in Web Mercator, and the
    opaque mosaic is warped once to EPSG:3006.

    Raises:
        FileNotFoundError: a needed tile is not in the repository at ``ref``.
    """
    from PIL import Image
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    def git(*args: str, binary: bool = False):
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
        return out.stdout if binary else out.stdout.decode().strip()

    commit = git("rev-parse", ref)
    w, s, e, n = transform_bounds(DISPLAY_CRS, "EPSG:4326", bounds["west"], bounds["south"],
                                  bounds["east"], bounds["north"], densify_pts=101)
    zb, zs = OSM_BASE_ZOOM, OSM_SEAMARK_ZOOM
    xb = range(_tile_x(w, zb), _tile_x(e, zb) + 1)
    yb = range(_tile_y(n, zb), _tile_y(s, zb) + 1)
    # Seamark tiles over exactly the basemap tiles' footprint (2x per axis).
    xs = range(2 * xb.start, 2 * xb.stop)
    ys = range(2 * yb.start, 2 * yb.stop)
    layers = {"base": ("dashboard/tiles", zb, xb, yb), "seamark": ("dashboard/tiles_seamark", zs, xs, ys)}

    listing = {}
    for root, z, _, _ in layers.values():
        for line in git("ls-tree", "-r", commit, f"{root}/{z}/").splitlines():
            meta, tile_path = line.split("\t")
            listing[tile_path] = meta.split()[2]
    missing = [f"{root}/{z}/{x}/{y}.png" for root, z, xr, yr in layers.values()
               for x in xr for y in yr if f"{root}/{z}/{x}/{y}.png" not in listing]
    if missing:
        raise FileNotFoundError(f"{len(missing)} tiles not in {repo}@{commit[:12]}: {missing[:5]}")

    def mosaic(root: str, z: int, xr: range, yr: range) -> Image.Image:
        im = Image.new("RGBA", (256 * len(xr), 256 * len(yr)))
        for i, x in enumerate(xr):
            for j, y in enumerate(yr):
                data = git("show", f"{commit}:{root}/{z}/{x}/{y}.png", binary=True)
                with Image.open(io.BytesIO(data)) as tile:
                    im.paste(tile.convert("RGBA"), (256 * i, 256 * j))
        return im

    base = mosaic(*layers["base"])
    base = base.resize((base.width * 2, base.height * 2), Image.BICUBIC)
    seamark = mosaic(*layers["seamark"])
    chart = Image.alpha_composite(base, seamark).convert("RGB")
    res = 2 * WEB_MERCATOR_HALF / (256 * 2 ** zs)
    src_transform = from_origin(-WEB_MERCATOR_HALF + xs.start * 256 * res,
                                WEB_MERCATOR_HALF - ys.start * 256 * res, res, res)
    rgb = np.asarray(chart)
    out = np.dstack([reproject_to_display(rgb[..., b], src_transform, "EPSG:3857",
                                          dst_transform, shape, src_nodata=None)
                     for b in range(3)])
    if np.isnan(out).any():
        raise RuntimeError("chart mosaic does not cover the display grid")
    Image.fromarray(np.round(out).astype(np.uint8), mode="RGB").save(path, optimize=True)
    print(f"    saved: {path} ({len(xb) * len(yb)} base z{zb} + {len(xs) * len(ys)} "
          f"seamark z{zs} tiles from {repo.name}@{commit[:12]})")
    tiles = sorted(f"{root}/{z}/{x}/{y}.png" for root, z, xr, yr in layers.values()
                   for x in xr for y in yr)
    return {"source": ("OpenStreetMap basemap + OpenSeaMap seamarks, offline tiles committed "
                       "in marine-awareness (scripts/prefetch_osm_tiles.py)"),
            "attribution": OSM_ATTRIBUTION,
            "licence": ("OSM data ODbL; OpenSeaMap tiles CC BY-SA 2.0; this derived chart "
                        "image is shared under CC BY-SA"),
            "repo": repo.name, "ref": ref, "commit": commit,
            "zooms": {"base": zb, "seamark": zs},
            "tiles": {"base": len(xb) * len(yb), "seamark": len(xs) * len(ys)},
            "tile_blobs_sha256": "sha256:" + hashlib.sha256(
                "\n".join(f"{t} {listing[t]}" for t in tiles).encode()).hexdigest(),
            "tile_blobs_sha256_of": "sorted '<path> <git blob sha>' lines, newline-joined",
            "resampling": "base bicubic x2 to z13, composite, then bilinear to EPSG:3006"}


# ── Detections in display pixel space ─────────────────────────────────────

def detections_to_display(fc: dict, regions: list[dict], dst_transform,
                          shape: tuple) -> list[dict]:
    """Pixel-space features (``[col, row]``, row 0 = top) inside the display extent.

    The WGS84 box ring of each ``s1_vessel_feature_collection`` feature is
    projected to EPSG:3006 and through the inverse display transform — the
    same pixel convention as ``mc_ai2_vessels.geojson``. A detection is kept
    when its AI2 point (the analyzer's lon/lat) falls inside the extent.
    ``regions`` are the analyzer's regions; the writer kept those inside the
    FeatureCollection's bbox, in order, so they pair with its features.
    ``above_threshold`` marks AI2's production threshold (``SCORE_THRESHOLD``);
    features are ordered by ascending score so the strongest draw on top.
    """
    from pyproj import Transformer

    from imint.exporters.export import s1_regions_in_bbox

    regions = s1_regions_in_bbox(regions, fc["properties"]["bbox"])
    to_3006 = Transformer.from_crs("EPSG:4326", DISPLAY_CRS, always_xy=True).transform
    inv = ~dst_transform
    height, width = shape

    def to_px(lon: float, lat: float) -> tuple[float, float]:
        return inv * to_3006(lon, lat)

    out = []
    for region, feat in zip(regions, fc["features"], strict=True):
        col, row = to_px(region["lon"], region["lat"])
        if not (0 <= col < width and 0 <= row < height):
            continue
        ring = [[round(c, 2), round(r, 2)] for c, r in
                (to_px(lon, lat) for lon, lat in feat["geometry"]["coordinates"][0])]
        props = {"label": "vessel", **feat["properties"],
                 "above_threshold": feat["properties"]["score"] >= SCORE_THRESHOLD,
                 "lon": round(region["lon"], 6), "lat": round(region["lat"], 6),
                 "point_px": [round(col, 2), round(row, 2)]}
        out.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": props})
    out.sort(key=lambda f: f["properties"]["score"])
    return out


def tier_stats(features: list[dict]) -> dict:
    """Count, score range and predicted-length median/range of one tier."""
    scores = [f["properties"]["score"] for f in features]
    lengths = [f["properties"]["pred_length_m"] for f in features
               if f["properties"]["pred_length_m"] is not None]
    return {"count": len(features),
            "score_range": [min(scores), max(scores)] if scores else None,
            "pred_length_m_median": round(float(np.median(lengths)), 1) if lengths else None,
            "pred_length_m_range": [round(min(lengths), 1), round(max(lengths), 1)]
            if lengths else None}


# ── Provenance helpers ────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def git_state() -> tuple[str, bool]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()

    # Untracked files are not code the run used (process_files hashes those).
    return (run("rev-parse", "--short", "HEAD"),
            bool(run("status", "--porcelain", "--untracked-files=no")))


def repo_relative(path: Path) -> str:
    """``path`` relative to the repo root when inside it, else absolute."""
    path = path.resolve()
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def scene_record(scene) -> dict:
    keep = ("scene_id", "safe_name", "start_utc", "end_utc", "platform",
            "orbit_direction", "relative_orbit", "absolute_orbit")
    return {k: v for k, v in dataclasses.asdict(scene).items() if k in keep}


# ── Main ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--area", required=True, choices=sorted(AREAS))
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / "imint_cache" / "s1",
                    help="radar window + model input cache (default: ~/imint_cache/s1)")
    ap.add_argument("--output-dir", type=Path,
                    help="working output dir (default: outputs/showcase/marine_radar_<area>)")
    ap.add_argument("--enc-dir", type=Path,
                    default=PROJECT_ROOT / "outputs" / "sjokort_oresund" / "unique_enc",
                    help="S-57 cells for the s57 chart (default: outputs/sjokort_oresund/unique_enc)")
    ap.add_argument("--marine-awareness-repo", type=Path,
                    default=PROJECT_ROOT.parent / "marine-awareness",
                    help="repository with the offline OSM/OpenSeaMap tiles (read-only)")
    ap.add_argument("--marine-awareness-ref", default="origin/main",
                    help="git ref of the tiles (default: origin/main)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    # Record the code state before this run writes its own outputs, which
    # would otherwise always mark the tree dirty.
    sha, dirty = git_state()

    from imint.analyzers.ai2_vessels_s1 import (
        DEFAULT_ATTR_CKPT, DEFAULT_DETECT_CKPT, AI2S1VesselAnalyzer)
    from imint.exporters.export import s1_vessel_feature_collection, save_s1_vessel_geojson
    from imint.fetch import fetch_s1_vessel_scenes

    area = AREAS[args.area]
    out_dir: Path = args.output_dir or PROJECT_ROOT / "outputs" / "showcase" / f"marine_radar_{area.key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = PROJECT_ROOT / "docs" / "showcase" / f"marine_radar_{area.key}"
    docs_data = PROJECT_ROOT / "docs" / "data"
    geojson_name = f"mr_{area.key}_vessels.geojson"
    wgs84_name = f"mr_{area.key}_vessels_wgs84.geojson"
    coords = dict(area.coords)
    fetch_coords = expand_bbox(coords, area.margin_m)
    dst_transform, shape, bounds = display_grid(coords)
    height, width = shape

    print(f"\n{'=' * 60}\n  Marine Radar Showcase — {area.name}\n"
          f"  Display area: {coords}\n  Fetch area:   {fetch_coords}\n"
          f"  Display grid: {DISPLAY_CRS} {width}x{height} px\n{'=' * 60}\n")

    # ── 1. Fetch target + historical (cached) ────────────────────────
    print("[1/5] Fetching radar windows (MPC, cached)...")
    scenes = fetch_s1_vessel_scenes(fetch_coords, area.target_date, cache_dir=args.cache_dir)
    for role, got, want in (("target", scenes.target_scene.scene_id, area.target_scene_id),
                            ("historical", scenes.historical_scene.scene_id,
                             area.historical_scene_id)):
        if got != want:
            raise RuntimeError(f"fetch picked {role} {got}, the preset pins {want}")
    print(f"  target {scenes.target_scene.scene_id}\n  historical "
          f"{scenes.historical_scene.scene_id}\n  model grid {scenes.geo.crs} "
          f"{scenes.geo.shape[1]}x{scenes.geo.shape[0]} px")

    # ── 2. Detect ────────────────────────────────────────────────────
    print(f"\n[2/5] Running ai2_vessels_s1 at score threshold {CANDIDATE_THRESHOLD}...")
    result = AI2S1VesselAnalyzer({"device": args.device, "predict_attributes": True,
                                  "score_threshold": CANDIDATE_THRESHOLD}).run(
        None, bands=scenes.bands(), date=area.target_date, coords=fetch_coords, geo=scenes.geo)
    if not result.success:
        print(f"analyzer failed: {result.error}", file=sys.stderr)
        return 1
    md = result.metadata
    print(f"  {md['patches']} patches, {md['raw_detections']} raw, {md['after_nms']} after "
          f"NMS, {md['final_detections']} after infra filter "
          f"(applied={md['infra_filter'].get('applied')})")
    fc = s1_vessel_feature_collection(
        result, scenes.geo, date=area.target_date,
        bbox=[fetch_coords[k] for k in ("west", "south", "east", "north")],
        scene_id=scenes.target_scene.scene_id,
        historical_scene_id=scenes.historical_scene.scene_id,
        scene_time_utc=scenes.scene_time_utc, scene_start_utc=scenes.target_scene.start_utc,
        scene_end_utc=scenes.target_scene.end_utc, source=S1_VESSEL_SOURCE)
    save_s1_vessel_geojson(fc, out_dir / wgs84_name)

    # ── 3. Radar images + pixel-space detections on the display grid ─
    print(f"\n[3/5] Reprojecting to the {DISPLAY_CRS} display grid...")
    image_stats = {}
    for name, stack in (("rgb.png", scenes.target), ("baseline.png", scenes.historical)):
        vv = reproject_to_display(stack[0], scenes.geo.transform, scenes.geo.crs,
                                  dst_transform, shape)
        image_stats[name] = save_vv_db_png(vv, out_dir / name)
    features = detections_to_display(fc, result.outputs["regions"], dst_transform, shape)
    strong = [f for f in features if f["properties"]["above_threshold"]]
    weak = [f for f in features if not f["properties"]["above_threshold"]]
    in_bbox = sum(coords["west"] <= f["properties"]["lon"] <= coords["east"]
                  and coords["south"] <= f["properties"]["lat"] <= coords["north"]
                  for f in strong)
    tab_fc = {
        "type": "FeatureCollection",
        "properties": {
            "coordinates": (f"pixel [col, row] of the {DISPLAY_CRS} display grid "
                            f"{width}x{height} px, row 0 = top"),
            "area": area.key,
            "scene_id": scenes.target_scene.scene_id,
            "scene_time_utc": scenes.scene_time_utc,
            "historical_scene_id": scenes.historical_scene.scene_id,
            "score_floor": CANDIDATE_THRESHOLD,
            "production_threshold": SCORE_THRESHOLD,
            "count": len(features),
            "count_above_threshold": len(strong),
            "note": ("Detections are candidates, not confirmed vessels; no AIS matched. "
                     "above_threshold marks AI2's production threshold."),
        },
        "features": features,
    }
    with open(out_dir / geojson_name, "w") as f:
        json.dump(tab_fc, f, indent=1)
    print(f"    saved: {out_dir / geojson_name} ({len(strong)} >= {SCORE_THRESHOLD}, "
          f"{len(weak)} in [{CANDIDATE_THRESHOLD}, {SCORE_THRESHOLD}) in the display extent; "
          f"{md['final_detections']} in the fetch window)")

    # ── 4. Nautical chart on the display grid ────────────────────────
    print("\n[4/5] Rendering the nautical chart on the display grid...")
    if area.chart == "s57":
        chart = render_s57_chart(args.enc_dir, coords, bounds, shape, out_dir / "sjokort.png")
    else:
        chart = render_osm_chart(args.marine_awareness_repo, args.marine_awareness_ref,
                                 bounds, dst_transform, shape, out_dir / "sjokort.png")

    # ── 5. Summary, manifest, copy to docs ───────────────────────────
    print("\n[5/5] Writing summary + manifest, copying to docs...")
    scene_info = {role: {"scene_id": sc.scene_id, "start_utc": sc.start_utc,
                         "platform": sc.platform, "orbit_direction": sc.orbit_direction,
                         "relative_orbit": sc.relative_orbit}
                  for role, sc in (("target", scenes.target_scene),
                                   ("historical", scenes.historical_scene))}
    summary = {
        "area": area.key,
        "area_name": area.name,
        "scenes": scene_info,
        "target_scene_time_utc": scenes.scene_time_utc,
        "coords": coords,
        "coords_fetch": fetch_coords,
        "fetch_margin_m": area.margin_m,
        "display_grid": {"crs": DISPLAY_CRS, "transform": list(dst_transform)[:6],
                         "width": width, "height": height},
        "model_grid": {"crs": scenes.geo.crs, "transform": list(scenes.geo.transform)[:6],
                       "width": scenes.geo.shape[1], "height": scenes.geo.shape[0]},
        "score_floor": CANDIDATE_THRESHOLD,
        "production_threshold": SCORE_THRESHOLD,
        "detections_above_threshold": tier_stats(strong),
        "candidates_below_threshold": tier_stats(weak),
        "detections_above_threshold_in_requested_bbox": in_bbox,
        "detections_in_fetch_window": md["final_detections"],
        "display": {"band": "VV", "scale": "20*log10(GRD DN), uncalibrated",
                    "stretch_db": list(VV_DB_RANGE), "resampling": "bilinear",
                    **image_stats},
        "chart": chart,
        "detector": {k: md[k] for k in ("score_threshold", "patches", "raw_detections",
                                        "after_nms", "final_detections", "infra_filter",
                                        "attr_crops_padded", "water_mask", "valid_frac_target",
                                        "valid_frac_historical", "seconds_total")},
    }
    with open(out_dir / "showcase_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    docs_files = ("rgb.png", "baseline.png", "sjokort.png", "showcase_summary.json",
                  "MANIFEST.json")
    stack = np.concatenate([scenes.target, scenes.historical]).astype(np.uint16)
    manifest = {
        "description": (f"Sentinel-1 radar showcase ({area.name}): VV images of the target "
                        "and historical scenes, a nautical chart and AI2 Sentinel-1 vessel "
                        f"detections (score >= {CANDIDATE_THRESHOLD}) on the {DISPLAY_CRS} "
                        "display grid. Detections are candidates, not confirmed vessels; "
                        "no AIS matched."),
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "produced_by": "scripts/generate_marine_radar_showcase.py",
        "git_sha": sha,
        "git_dirty": dirty,
        "run_args": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     "preset": dataclasses.asdict(area), "coords_fetch": fetch_coords},
        "source": S1_VESSEL_SOURCE,
        "scenes": {"target": scene_record(scenes.target_scene),
                   "historical": scene_record(scenes.historical_scene)},
        "fetch_provenance": {k: scenes.provenance[k] for k in
                             ("chain_version", "window_epsg", "aoi_px", "valid_frac_target",
                              "valid_frac_historical", "rasterio", "gdal")},
        "input_data_sha256": "sha256:" + hashlib.sha256(stack.tobytes()).hexdigest(),
        "input_data_hash_of": "uint16 [target_vv, target_vh, historical_vv, historical_vh] "
                              "model input stack, C order",
        "models": {
            "licence": "AI2 rslearn_projects weights + code Apache-2.0, annotations CC-BY 4.0",
            "detector": {"file": Path(DEFAULT_DETECT_CKPT).name,
                         "sha256": sha256_file(Path(DEFAULT_DETECT_CKPT))},
            "attribute": {"file": Path(DEFAULT_ATTR_CKPT).name,
                          "sha256": sha256_file(Path(DEFAULT_ATTR_CKPT))},
        },
        "infra_filter": md["infra_filter"],
        "chart": chart,
        "process_files": [{"path": p, "sha256": sha256_file(PROJECT_ROOT / p)} for p in (
            "scripts/generate_marine_radar_showcase.py",
            "scripts/generate_marine_commercial_showcase.py",
            "imint/analyzers/ai2_vessels_s1.py",
            "imint/fetch.py",
            "imint/exporters/export.py")],
        "outputs": {repo_relative(p): sha256_file(p) for p in (
            *(out_dir / n for n in docs_files[:-1]),
            out_dir / geojson_name, out_dir / wgs84_name)},
        "outputs_copied_to": [repo_relative(docs_dir / n) for n in docs_files]
                             + [repo_relative(docs_data / geojson_name)],
    }
    with open(out_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    docs_dir.mkdir(parents=True, exist_ok=True)
    for name in docs_files:
        shutil.copy2(out_dir / name, docs_dir / name)
        print(f"  Copied: {name}")
    shutil.copy2(out_dir / geojson_name, docs_data / geojson_name)
    print(f"  Copied: {geojson_name}")

    s, c = summary["detections_above_threshold"], summary["candidates_below_threshold"]
    print(f"\n{'=' * 60}\n  Done ({area.key}): {s['count']} detections >= {SCORE_THRESHOLD} "
          f"(median predicted length {s['pred_length_m_median']} m), {c['count']} candidates "
          f"in [{CANDIDATE_THRESHOLD}, {SCORE_THRESHOLD})\n  Images {width}w x {height}h "
          f"(tab-data.js imgW: {width}, imgH: {height})\n{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
