"""Render each band as a single high-resolution grayscale PNG (no overlay, no
crosshair) at native pixel scale for the interactive picker.

Each PNG corresponds to the 120x120 crop (1.2 km x 1.2 km @ 10 m/px) used
for the zoom_per_band figure, hard 1-99 percentile stretched.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_CACHE = REPO_ROOT / "demos" / "aircraft_parallax" / "cache_l1c"
DOCS_DIR = REPO_ROOT / "docs" / "showcase" / "aircraft_parallax"

POINT_LAT = 57.71809
POINT_LON = 11.66456
CROP_HALF_PX = 60
UPSCALE = 8


def find_safe() -> Path:
    return sorted(SAFE_CACHE.glob("S2*_MSIL1C_20260111*.SAFE"))[0]


def world_to_pixel(src, lon, lat):
    xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
    return src.index(xs[0], ys[0])


def main():
    safe = find_safe()
    img_data = next((safe / "GRANULE").glob("*")) / "IMG_DATA"
    print(f"SAFE: {safe.name}")

    from PIL import Image
    crop_origin = None
    for band_id in ("B02", "B03", "B04", "B08"):
        jp2 = next(img_data.glob(f"*_{band_id}.jp2"))
        with rasterio.open(jp2) as src:
            row, col = world_to_pixel(src, POINT_LON, POINT_LAT)
            r0 = max(0, row - CROP_HALF_PX); r1 = r0 + 2 * CROP_HALF_PX
            c0 = max(0, col - CROP_HALF_PX); c1 = c0 + 2 * CROP_HALF_PX
            arr = src.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)
            if crop_origin is None:
                crop_origin = (r0, c0)

        valid = arr > 0
        lo, hi = np.percentile(arr[valid], [1, 99])
        norm = np.clip((arr - lo) / max(hi - lo, 1), 0, 1)
        gray = (norm * 255).astype(np.uint8)
        # 8x nearest-neighbour upscale so each native pixel = 8x8 screen pixels
        h, w = gray.shape
        rgb = np.stack([gray, gray, gray], axis=-1)
        img = Image.fromarray(rgb).resize((w * UPSCALE, h * UPSCALE), Image.NEAREST)
        out_path = DOCS_DIR / f"picker_{band_id}.png"
        img.save(out_path)
        print(f"  {band_id}: {out_path} ({img.width}x{img.height}), p1={lo:.0f} p99={hi:.0f}")

    print(f"\nCrop origin (full raster yx): {crop_origin}")
    print(f"Crop size (native px): 120 x 120")
    print(f"Upscale: {UPSCALE}x → display 960 x 960")


if __name__ == "__main__":
    main()
