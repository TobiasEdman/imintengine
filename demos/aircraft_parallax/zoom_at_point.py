"""Zoom in on the user-supplied aircraft point and hard-stretch each band.

Aircraft point (user-given): 57.71809 N, 11.66456 E
Scene: S2B 2026-01-11 10:43:19 UTC, tile T32VPK.

We do NOT try to find the aircraft anywhere else. We crop a small tile
around the exact point in each 10 m band, hard-stretch it (1-99 percentile
within the crop), then render:

  outputs/showcase/aircraft_parallax/zoom_rgb.png        (small RGB closeup)
  outputs/showcase/aircraft_parallax/zoom_per_band.png   (B02/B03/B04/B08 side-by-side, hard stretched)
  docs/showcase/aircraft_parallax/zoom_rgb.png
  docs/showcase/aircraft_parallax/zoom_per_band.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_REPO = Path("/Users/tobiasedman/Developer/ImintEngine")
sys.path.insert(0, str(MAIN_REPO))

# Aircraft point per user
AIRCRAFT_LAT = 57.71809
AIRCRAFT_LON = 11.66456

# Crop window — a few pixels around the point in each direction.
# 50 px = 500 m which easily contains the full B02→B08 parallax envelope
# (104 px = ~1 km would be max for Mach-1; 50 px is enough for Mach 0.5).
CROP_HALF_PX = 60

OUT_DIR = REPO_ROOT / "outputs" / "showcase" / "aircraft_parallax"
DOCS_DIR = REPO_ROOT / "docs" / "showcase" / "aircraft_parallax"
SAFE_CACHE = REPO_ROOT / "demos" / "aircraft_parallax" / "cache_l1c"


def find_safe() -> Path:
    candidates = sorted(SAFE_CACHE.glob("S2*_MSIL1C_20260111*.SAFE"))
    if not candidates:
        sys.exit("No SAFE found; run fetch_and_render.py first")
    return candidates[0]


def world_to_pixel(src, lon: float, lat: float) -> tuple[int, int]:
    """Project WGS84 (lon, lat) to (row, col) in the band's full raster."""
    xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
    x, y = xs[0], ys[0]
    row, col = src.index(x, y)
    return row, col


def hard_stretch(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Per-tile percentile stretch to 0..255 uint8."""
    valid = arr > 0
    if valid.sum() == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(arr[valid], [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (norm * 255).astype(np.uint8)


def main() -> int:
    safe = find_safe()
    print(f"SAFE: {safe.name}")
    print(f"Aircraft point: {AIRCRAFT_LAT} N, {AIRCRAFT_LON} E")

    img_data = next((safe / "GRANULE").glob("*")) / "IMG_DATA"

    # Get pixel coords from B02 (10m, all 10m bands share this grid)
    band_arrays = {}
    band_meta = {}
    for band_id in ("B02", "B03", "B04", "B08"):
        jp2 = next(img_data.glob(f"*_{band_id}.jp2"))
        with rasterio.open(jp2) as src:
            row, col = world_to_pixel(src, AIRCRAFT_LON, AIRCRAFT_LAT)
            print(f"  {band_id}: aircraft at row={row}, col={col}")
            r0 = max(0, row - CROP_HALF_PX)
            r1 = min(src.height, row + CROP_HALF_PX)
            c0 = max(0, col - CROP_HALF_PX)
            c1 = min(src.width, col + CROP_HALF_PX)
            arr = src.read(1, window=((r0, r1), (c0, c1)))
            band_arrays[band_id] = arr.astype(np.float32)
            band_meta[band_id] = {"row": row, "col": col, "crop_origin": (r0, c0)}
            print(f"    crop {arr.shape}, min={arr.min()}, max={arr.max()}, p99={np.percentile(arr[arr>0], 99):.0f}")

    # Hard stretch each band
    stretched = {bid: hard_stretch(arr, 1, 99) for bid, arr in band_arrays.items()}

    from PIL import Image, ImageDraw

    # 1) RGB closeup at native resolution
    h, w = next(iter(stretched.values())).shape
    rgb = np.stack([stretched["B04"], stretched["B03"], stretched["B02"]], axis=-1)
    rgb_img = Image.fromarray(rgb, mode="RGB")
    # 8x upscale with nearest neighbor for visibility
    upscale = 8
    rgb_big = rgb_img.resize((w * upscale, h * upscale), Image.NEAREST)
    # Mark center
    draw = ImageDraw.Draw(rgb_big)
    cx, cy = (w // 2) * upscale, (h // 2) * upscale
    draw.line([cx - 30, cy, cx - 10, cy], fill=(255, 255, 0), width=2)
    draw.line([cx + 10, cy, cx + 30, cy], fill=(255, 255, 0), width=2)
    draw.line([cx, cy - 30, cx, cy - 10], fill=(255, 255, 0), width=2)
    draw.line([cx, cy + 10, cx, cy + 30], fill=(255, 255, 0), width=2)
    rgb_path = OUT_DIR / "zoom_rgb.png"
    rgb_big.save(rgb_path)
    print(f"\n  RGB closeup: {rgb_path}  ({rgb_big.width}x{rgb_big.height})")
    rgb_big.save(DOCS_DIR / "zoom_rgb.png")

    # 2) Per-band 4-panel grid — each band hard-stretched, 8x upscaled
    pad = 16
    label_h = 30
    cell_w = w * upscale
    cell_h = h * upscale + label_h
    grid = Image.new("RGB", (cell_w * 2 + pad * 3, cell_h * 2 + pad * 3 + 40), (250, 250, 250))
    grid_draw = ImageDraw.Draw(grid)
    grid_draw.text((pad, 8), "Aircraft point 57.71809°N, 11.66456°E — hard 1-99% stretch, 8× nearest neighbor",
                   fill=(50, 50, 50))

    positions = [
        ("B02", 0, 0, (21, 101, 192), "B02 · 490 nm (blue) · t\u2080"),
        ("B03", 1, 0, (67, 160, 71), "B03 · 560 nm (green) · t\u2080+0.527s"),
        ("B04", 0, 1, (229, 57, 53), "B04 · 665 nm (red) · t\u2080+0.585s"),
        ("B08", 1, 1, (236, 64, 122), "B08 · 842 nm (NIR) · t\u2080+1.005s"),
    ]
    for band_id, gx, gy, color, label in positions:
        s = stretched[band_id]
        rgb_band = np.stack([s, s, s], axis=-1)
        bimg = Image.fromarray(rgb_band, mode="RGB").resize((cell_w, h * upscale), Image.NEAREST)
        x0 = pad + gx * (cell_w + pad)
        y0 = 40 + pad + gy * (cell_h + pad) + label_h
        grid.paste(bimg, (x0, y0))
        grid_draw.text((x0 + 4, y0 - label_h + 6), label, fill=color)
        # Center crosshair
        ccx = x0 + cell_w // 2
        ccy = y0 + (h * upscale) // 2
        grid_draw.line([ccx - 30, ccy, ccx - 10, ccy], fill=(255, 255, 0), width=2)
        grid_draw.line([ccx + 10, ccy, ccx + 30, ccy], fill=(255, 255, 0), width=2)
        grid_draw.line([ccx, ccy - 30, ccx, ccy - 10], fill=(255, 255, 0), width=2)
        grid_draw.line([ccx, ccy + 10, ccx, ccy + 30], fill=(255, 255, 0), width=2)

    grid_path = OUT_DIR / "zoom_per_band.png"
    grid.save(grid_path)
    print(f"  Per-band zoom: {grid_path}  ({grid.width}x{grid.height})")
    grid.save(DOCS_DIR / "zoom_per_band.png")

    # 3) Print intensity values at the exact center pixel and ±1 in each band
    print("\nCenter-pixel intensities at the aircraft point:")
    cy_local, cx_local = h // 2, w // 2
    for band_id, arr in band_arrays.items():
        p = arr[cy_local, cx_local]
        peak3 = arr[max(0,cy_local-1):cy_local+2, max(0,cx_local-1):cx_local+2].max()
        p99 = np.percentile(arr[arr>0], 99) if (arr>0).any() else 0
        print(f"  {band_id}: center={p:.0f}, max(3x3)={peak3:.0f}, p99(crop)={p99:.0f}, p99(full)={np.percentile(arr[arr>0],99):.0f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
