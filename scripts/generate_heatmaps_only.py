"""Generate vessel heatmaps for Öresund using CDSE Sentinel Hub HTTP API.

Uses _stac_available_dates() for date discovery and direct HTTP POST to
Sentinel Hub Process API for per-date Sentinel-2 fetching (no openEO).

For AI2 model, fetches all 9 spectral bands + SCL (10-band TIFF).
For YOLO, uses the standard 6-band fetch via fetch_s2_scene().

Usage:
    .venv/bin/python scripts/generate_heatmaps_only.py
    .venv/bin/python scripts/generate_heatmaps_only.py --analyzer yolo
    .venv/bin/python scripts/generate_heatmaps_only.py --analyzer ai2
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Öresund area ───────────────────────────────────────────────────
COORDS_WGS84 = {
    "west":  12.560,
    "south": 56.010,
    "east":  12.700,
    "north": 56.060,
}
HEATMAP_START = "2025-06-01"
HEATMAP_END = "2025-08-31"

# Band name mappings
_PRITHVI_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
# AI2 requires these 9 bands (B8A mapped to B08 for model compatibility)
_AI2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
# Sentinel Hub band names for the 10-band fetch (9 spectral + SCL)
_SH_AI2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "SCL"]

_SCL_CLOUD_CLASSES = frozenset({3, 8, 9, 10})
_REQUEST_TIMEOUT_S = 90
_MAX_RETRIES = 3


def _build_ai2_evalscript() -> str:
    """Evalscript that returns 9 spectral bands + SCL (10 bands total)."""
    return """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "SCL"],
      units: "DN"
    }],
    output: {
      bands: 10,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  return [
    sample.B02 / 10000,
    sample.B03 / 10000,
    sample.B04 / 10000,
    sample.B05 / 10000,
    sample.B06 / 10000,
    sample.B07 / 10000,
    sample.B8A / 10000,
    sample.B11 / 10000,
    sample.B12 / 10000,
    sample.SCL
  ];
}
"""


def _fetch_ai2_scene(
    west: float, south: float, east: float, north: float,
    w_px: int, h_px: int,
    date: str,
    cloud_threshold: float = 0.3,
) -> tuple[dict, np.ndarray, float] | None:
    """Fetch 9 spectral bands + SCL via Sentinel Hub HTTP for AI2 model.

    Returns (bands_dict, scl, cloud_fraction) or None if rejected.
    """
    from imint.training.cdse_vpp import _get_token, _token_lock, _SH_PROCESS_URL
    import rasterio
    from rasterio.io import MemoryFile

    token = _get_token()

    request_body = {
        "input": {
            "bounds": {
                "bbox": [west, south, east, north],
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/3006"
                },
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{date}T00:00:00Z",
                        "to": f"{date}T23:59:59Z",
                    },
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {
            "width": w_px,
            "height": h_px,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff"},
            }],
        },
        "evalscript": _build_ai2_evalscript(),
    }

    body_bytes = json.dumps(request_body).encode()

    for attempt in range(_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                _SH_PROCESS_URL,
                data=body_bytes,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                    "Accept": "image/tiff",
                },
            )
            resp = urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S)
            data = resp.read()

            if data[:4] not in (b"II*\x00", b"MM\x00*"):
                try:
                    err = json.loads(data)
                    msg = err.get("error", {}).get("message", str(err)[:300])
                    print(f"    SH error: {msg}")
                except Exception:
                    print(f"    Unexpected response ({len(data)} bytes)")
                return None

            break

        except urllib.error.HTTPError as e:
            if e.code == 401:
                import imint.training.cdse_vpp as _vpp_mod
                with _token_lock:
                    _vpp_mod._cached_token = None
                    _vpp_mod._token_expires = 0.0
                token = _get_token()
                continue
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", 10))
                time.sleep(retry_after)
                continue
            if attempt < _MAX_RETRIES:
                time.sleep(2.0 * (attempt + 1))
                continue
            print(f"    HTTP {e.code}")
            return None
        except Exception as e:
            if attempt < _MAX_RETRIES:
                time.sleep(2.0 * (attempt + 1))
                continue
            print(f"    Fetch error: {e}")
            return None
    else:
        return None

    # Parse 10-band TIFF
    try:
        with MemoryFile(data) as memfile:
            with memfile.open() as ds:
                raw = ds.read()  # (10, H, W)
    except Exception as e:
        print(f"    TIFF parse error: {e}")
        return None

    if raw.shape[0] < 10:
        print(f"    Only {raw.shape[0]} bands returned (need 10)")
        return None

    # Split: spectral (0-8) and SCL (9)
    scl = raw[9].astype(np.uint8)

    # Cloud check
    cloud_mask = np.isin(scl, list(_SCL_CLOUD_CLASSES))
    cloud_frac = float(cloud_mask.sum()) / max(scl.size, 1)
    if cloud_frac > cloud_threshold:
        return None

    # Nodata check
    if float((raw[0] == 0).mean()) > 0.10:
        return None

    # Build bands dict — map B8A to B08 for AI2 compatibility
    bands = {}
    sh_names = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
    imint_names = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
    for i, (sh_name, imint_name) in enumerate(zip(sh_names, imint_names)):
        bands[imint_name] = raw[i].astype(np.float32)

    return bands, scl, cloud_frac


def _spectral_to_bands_dict(spectral: np.ndarray) -> dict:
    """Convert (6, H, W) spectral array to IMINT bands dict."""
    return {name: spectral[i] for i, name in enumerate(_PRITHVI_BANDS)}


def _make_rgb(bands: dict, scl: np.ndarray | None = None) -> np.ndarray:
    """Create (H, W, 3) RGB from bands dict."""
    from imint.utils import bands_to_rgb
    return bands_to_rgb(bands, scl=scl)


def main():
    import argparse
    from imint.config.env import load_env
    load_env("dev")

    parser = argparse.ArgumentParser(description="Generate vessel heatmaps via CDSE HTTP")
    parser.add_argument(
        "--analyzer", default="both", choices=["yolo", "ai2", "both"],
        help="Which analyzer to run (default: both)",
    )
    parser.add_argument(
        "--cloud-threshold", type=float, default=0.3,
        help="Max AOI cloud fraction (default: 0.3)",
    )
    parser.add_argument(
        "--scene-cloud-max", type=float, default=50.0,
        help="Max scene cloud %% for STAC filter (default: 50)",
    )
    parser.add_argument(
        "--sigma", type=float, default=5.0,
        help="Gaussian smoothing sigma in pixels (default: 5.0)",
    )
    args = parser.parse_args()

    from scipy.ndimage import gaussian_filter
    from imint.fetch import _stac_available_dates, _to_nmd_grid
    from imint.training.cdse_s2 import fetch_s2_scene
    from imint.exporters.export import save_vessel_heatmap_png

    out_dir = str(PROJECT_ROOT / "outputs" / "showcase" / "marine_commercial")
    os.makedirs(out_dir, exist_ok=True)

    need_ai2 = args.analyzer in ("ai2", "both")
    need_yolo = args.analyzer in ("yolo", "both")

    # Convert WGS84 → EPSG:3006 for Sentinel Hub Process API
    proj = _to_nmd_grid(COORDS_WGS84)
    west, south, east, north = proj["west"], proj["south"], proj["east"], proj["north"]
    w_px = int((east - west) / 10)
    h_px = int((north - south) / 10)

    print(f"\n{'='*60}")
    print(f"  Vessel Heatmap — Öresund (CDSE Sentinel Hub HTTP)")
    print(f"  Period: {HEATMAP_START} to {HEATMAP_END}")
    print(f"  WGS84: {COORDS_WGS84}")
    print(f"  EPSG:3006: W={west} S={south} E={east} N={north}")
    print(f"  Image size: {w_px}x{h_px} px (10m)")
    print(f"  Analyzers: {args.analyzer}")
    print(f"{'='*60}\n")

    # ── Discover available dates via STAC ───────────────────────────
    print("[STAC] Querying available dates...")
    stac_dates = _stac_available_dates(
        COORDS_WGS84, HEATMAP_START, HEATMAP_END,
        scene_cloud_max=args.scene_cloud_max,
    )
    print(f"[STAC] {len(stac_dates)} dates with scene cloud < {args.scene_cloud_max}%\n")

    if not stac_dates:
        print("No dates available — nothing to do.")
        return

    # ── Fetch all dates via Sentinel Hub HTTP ──────────────────────
    # If AI2 is needed, fetch 9-band version (also usable for YOLO)
    # Otherwise fetch 6-band version for YOLO only
    yolo_cache = []   # list of (date, bands_dict, scl, cloud_frac)
    ai2_cache = []    # list of (date, bands_dict, scl, cloud_frac)

    for i, (date_str, scene_cloud) in enumerate(stac_dates):
        print(f"[{i+1}/{len(stac_dates)}] Fetching {date_str} (scene cloud {scene_cloud:.1f}%)...")

        if need_ai2:
            # Fetch 9 bands — usable for both AI2 and YOLO
            result = _fetch_ai2_scene(
                west, south, east, north, w_px, h_px,
                date=date_str,
                cloud_threshold=args.cloud_threshold,
            )
            if result is None:
                print(f"    Rejected (cloud/nodata)")
                continue
            bands, scl, cloud_frac = result
            print(f"    OK  bands={len(bands)}  AOI cloud={cloud_frac:.1%}")
            ai2_cache.append((date_str, bands, scl, cloud_frac))
            # YOLO can use the same data (it only needs B02, B03, B04 for RGB)
            if need_yolo:
                yolo_cache.append((date_str, bands, scl, cloud_frac))
        else:
            # YOLO only — 6-band fetch is sufficient
            result = fetch_s2_scene(
                west, south, east, north,
                date=date_str,
                size_px=(h_px, w_px),
                cloud_threshold=args.cloud_threshold,
                haze_threshold=1.0,
            )
            if result is None:
                print(f"    Rejected (cloud/nodata)")
                continue
            spectral, scl, cloud_frac = result
            bands = _spectral_to_bands_dict(spectral)
            print(f"    OK  shape={spectral.shape}  AOI cloud={cloud_frac:.1%}")
            yolo_cache.append((date_str, bands, scl, cloud_frac))

    total_cached = len(ai2_cache) if need_ai2 else len(yolo_cache)
    print(f"\n{total_cached} usable dates fetched.\n")

    if total_cached == 0:
        print("No cloud-free dates — cannot build heatmap.")
        return

    # ── YOLO heatmap ───────────────────────────────────────────────
    if need_yolo:
        print(f"\n{'='*60}")
        print(f"  Building YOLO heatmap")
        print(f"{'='*60}\n")

        from imint.analyzers.marine_vessels import MarineVesselAnalyzer
        analyzer = MarineVesselAnalyzer(config={
            "confidence": 0.286,
            "chip_size": 320,
            "overlap_ratio": 0.2,
            "water_filter": True,
            "max_bbox_m": 750,
        })

        heatmap = None
        total_det = 0

        for date_str, bands, scl, cloud_frac in yolo_cache:
            rgb = _make_rgb(bands, scl=scl)
            result = analyzer.run(rgb, bands=bands, scl=scl)
            regions = result.outputs.get("regions", []) if result.success else []
            n = len(regions)
            total_det += n
            print(f"    {date_str}  cloud={cloud_frac:.1%}  vessels={n}")

            if heatmap is None:
                heatmap = np.zeros(rgb.shape[:2], dtype=np.float32)
            for r in regions:
                bb = r["bbox"]
                cy = (bb["y_min"] + bb["y_max"]) // 2
                cx = (bb["x_min"] + bb["x_max"]) // 2
                if 0 <= cy < heatmap.shape[0] and 0 <= cx < heatmap.shape[1]:
                    heatmap[cy, cx] += 1.0

        print(f"\n    {len(yolo_cache)} dates, {total_det} total detections")

        if heatmap is not None and heatmap.max() > 0:
            heatmap = gaussian_filter(heatmap, sigma=args.sigma)
            hm_path = os.path.join(out_dir, "vessel_heatmap_clean.png")
            save_vessel_heatmap_png(heatmap, hm_path)
            print(f"    Saved: {hm_path} ({os.path.getsize(hm_path)/1024:.0f} KB)")

    # ── AI2 heatmap ────────────────────────────────────────────────
    if need_ai2:
        print(f"\n{'='*60}")
        print(f"  Building AI2 heatmap")
        print(f"{'='*60}\n")

        from imint.analyzers.ai2_vessels import AI2VesselAnalyzer
        analyzer = AI2VesselAnalyzer(config={
            "predict_attributes": True,
            "water_filter": True,
            "max_bbox_m": 750,
            "device": "cpu",
        })

        heatmap = None
        total_det = 0

        for date_str, bands, scl, cloud_frac in ai2_cache:
            rgb = _make_rgb(bands, scl=scl)
            result = analyzer.run(rgb, bands=bands, scl=scl)
            regions = result.outputs.get("regions", []) if result.success else []
            n = len(regions)
            total_det += n
            print(f"    {date_str}  cloud={cloud_frac:.1%}  vessels={n}")
            if not result.success:
                print(f"      error: {result.error}")

            if heatmap is None:
                heatmap = np.zeros(rgb.shape[:2], dtype=np.float32)
            for r in regions:
                bb = r["bbox"]
                cy = (bb["y_min"] + bb["y_max"]) // 2
                cx = (bb["x_min"] + bb["x_max"]) // 2
                if 0 <= cy < heatmap.shape[0] and 0 <= cx < heatmap.shape[1]:
                    heatmap[cy, cx] += 1.0

        print(f"\n    {len(ai2_cache)} dates, {total_det} total detections")

        if heatmap is not None and heatmap.max() > 0:
            heatmap = gaussian_filter(heatmap, sigma=args.sigma)
            hm_path = os.path.join(out_dir, "ai2_vessel_heatmap_clean.png")
            save_vessel_heatmap_png(heatmap, hm_path)
            print(f"    Saved: {hm_path} ({os.path.getsize(hm_path)/1024:.0f} KB)")

    # ── Copy to docs ───────────────────────────────────────────────
    docs_showcase = PROJECT_ROOT / "docs" / "showcase" / "marine_commercial"
    os.makedirs(str(docs_showcase), exist_ok=True)

    for fname in ["vessel_heatmap_clean.png", "ai2_vessel_heatmap_clean.png"]:
        src = os.path.join(out_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, str(docs_showcase / fname))
            sz = os.path.getsize(src)
            print(f"  Copied to docs: {fname} ({sz/1024:.0f} KB)")

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
