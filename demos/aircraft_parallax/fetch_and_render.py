"""Fetch L1C SAFE for 2026-01-11 ~10:43 UTC over Hisingen and render RGB.

AOI: 5x5 km around 57.71818 N, 11.66559 E (Sa"ve / Hisingen, Gothenburg).
Date: 2026-01-11 (begaerd passage 10:43:19 UTC; faktisk passage matchas via
GCP product_id-listing).

Produces:
  outputs/showcase/aircraft_parallax/rgb.png    (5x5 km RGB, 2-98 percentile)
  docs/showcase/aircraft_parallax/rgb.png       (mirror for dashboard)
  outputs/showcase/aircraft_parallax/MANIFEST.json
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_REPO = Path("/Users/tobiasedman/Developer/ImintEngine")
sys.path.insert(0, str(MAIN_REPO))

from imint.fetch import fetch_l1c_safe_from_gcp  # noqa: E402

# --- Config -----------------------------------------------------------------

CENTER_LAT = 57.71818
CENTER_LON = 11.66559
HALF_KM = 2.5  # 5x5 km AOI

# Approximate degree offsets for 2.5 km half-width at lat 57.7
DLAT = HALF_KM / 111.32
DLON = HALF_KM / (111.32 * math.cos(math.radians(CENTER_LAT)))

BBOX_WGS84 = {
    "west":  CENTER_LON - DLON,
    "south": CENTER_LAT - DLAT,
    "east":  CENTER_LON + DLON,
    "north": CENTER_LAT + DLAT,
}

DATE = "2026-01-11"

OUT_DIR = REPO_ROOT / "outputs" / "showcase" / "aircraft_parallax"
DOCS_DIR = REPO_ROOT / "docs" / "showcase" / "aircraft_parallax"
SAFE_CACHE = REPO_ROOT / "demos" / "aircraft_parallax" / "cache_l1c"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
SAFE_CACHE.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print(f"AOI bbox WGS84: {BBOX_WGS84}")
    print(f"Date: {DATE}")
    print(f"Center: {CENTER_LAT} N, {CENTER_LON} E (Sa\u0308ve / Hisingen)")

    # Step 1: Fetch L1C SAFE from GCP public bucket
    print("\n[1/3] Fetching L1C SAFE from Google Cloud public bucket...")
    safe = fetch_l1c_safe_from_gcp(
        date=DATE,
        coords=BBOX_WGS84,
        dest_dir=SAFE_CACHE,
        cloud_max=100.0,            # accept any cloudiness
        preferred_utm_zone=32,      # Hisingen falls inside UTM 32N
    )
    print(f"  SAFE: {safe.name}")

    # Step 2: Read B02/B03/B04/B08 (10 m bands) and crop to AOI
    print("\n[2/3] Reading B02/B03/B04/B08 and cropping to AOI...")
    granule_dir = next((safe / "GRANULE").glob("*"))
    img_data = granule_dir / "IMG_DATA"

    bands = {}
    for band_id in ("B02", "B03", "B04", "B08"):
        jp2_candidates = list(img_data.glob(f"*_{band_id}.jp2"))
        if not jp2_candidates:
            print(f"  ERROR: no JP2 for {band_id} in {img_data}")
            return 1
        jp2 = jp2_candidates[0]
        with rasterio.open(jp2) as src:
            utm_bounds = transform_bounds(
                "EPSG:4326", src.crs,
                BBOX_WGS84["west"], BBOX_WGS84["south"],
                BBOX_WGS84["east"], BBOX_WGS84["north"],
            )
            win = from_bounds(*utm_bounds, transform=src.transform)
            arr = src.read(1, window=win)
            bands[band_id] = arr.astype(np.float32)
            print(f"  {band_id}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")

    # Step 3: Stack as RGB and render with 2-98 percentile stretch
    print("\n[3/3] Rendering RGB PNG with 2-98 percentile stretch...")
    h_min = min(b.shape[0] for b in bands.values())
    w_min = min(b.shape[1] for b in bands.values())
    r = bands["B04"][:h_min, :w_min]
    g = bands["B03"][:h_min, :w_min]
    b = bands["B02"][:h_min, :w_min]

    rgb = np.stack([r, g, b], axis=-1)

    # Per-channel 2-98 percentile stretch (keep colour balance robust)
    valid = rgb > 0
    out = np.zeros_like(rgb, dtype=np.uint8)
    for ch in range(3):
        ch_arr = rgb[..., ch]
        v_mask = valid[..., ch]
        if v_mask.sum() == 0:
            continue
        lo, hi = np.percentile(ch_arr[v_mask], [2, 98])
        if hi <= lo:
            continue
        norm = np.clip((ch_arr - lo) / (hi - lo), 0, 1)
        out[..., ch] = (norm * 255).astype(np.uint8)

    # Write PNG via PIL (already in repo deps)
    from PIL import Image
    img = Image.fromarray(out, mode="RGB")
    rgb_path = OUT_DIR / "rgb.png"
    img.save(rgb_path)
    print(f"  RGB written: {rgb_path}  ({img.width}x{img.height})")

    # Mirror to docs/
    docs_rgb = DOCS_DIR / "rgb.png"
    img.save(docs_rgb)
    print(f"  RGB mirrored: {docs_rgb}")

    # MANIFEST
    manifest = {
        "produced_at": datetime.utcnow().isoformat() + "Z",
        "safe_archive": safe.name,
        "date": DATE,
        "center_lat": CENTER_LAT,
        "center_lon": CENTER_LON,
        "aoi_half_km": HALF_KM,
        "bbox_wgs84": BBOX_WGS84,
        "image_size_px": [img.width, img.height],
        "bands_used": ["B04 (red, 665 nm)", "B03 (green, 560 nm)", "B02 (blue, 490 nm)"],
        "stretch": "per-channel 2-98 percentile",
        "purpose": "aircraft_parallax showcase tab — RGB for visual inspection of push-broom band offsets",
    }
    manifest_path = OUT_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  MANIFEST: {manifest_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
