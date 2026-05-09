"""Track the aircraft HEAD over water (east of the user-given point).

The user observes that the streak's HEAD lies east of the original 1.2x1.2 km
crop, over water. Water gives much better SNR than winter ground because
water has near-zero NIR reflectance, so the contrail head pops out cleanly.

We expand the crop and search a 60x60 px window centred on the brightest
B08 pixel (the aircraft body is the strongest NIR reflector). Then we
locate the head in each band and measure the offsets.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
from scipy.ndimage import median_filter, label

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_CACHE = REPO_ROOT / "demos" / "aircraft_parallax" / "cache_l1c"
OUT_DIR = REPO_ROOT / "outputs" / "showcase" / "aircraft_parallax"
DOCS_DIR = REPO_ROOT / "docs" / "showcase" / "aircraft_parallax"

# User-supplied point (somewhere along the streak); the head is east of this.
POINT_LAT = 57.71809
POINT_LON = 11.66456

# Wider crop east of the point: shift centre east so the streak head is
# inside our window. East = +cos(lat) deg of lon. Half-window = 200 px = 2 km.
EAST_SHIFT_M = 800.0  # 800 m east — head is over water
DLON_SHIFT = EAST_SHIFT_M / (111320.0 * math.cos(math.radians(POINT_LAT)))
SEARCH_LAT = POINT_LAT
SEARCH_LON = POINT_LON + DLON_SHIFT
CROP_HALF_PX = 200

DT_B02_B08 = 1.005
BAND_OFFSET_S = {"B02": 0.0, "B03": 0.527, "B04": 0.585, "B08": 1.005}


def find_safe() -> Path:
    return sorted(SAFE_CACHE.glob("S2*_MSIL1C_20260111*.SAFE"))[0]


def world_to_pixel(src, lon, lat):
    xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
    return src.index(xs[0], ys[0])


def find_head_in_water(arr_b08: np.ndarray, water_arr_b04: np.ndarray) -> tuple[int, int, float]:
    """Find the brightest B08 pixel that's over water (B04 < some threshold).

    Water in winter at low sun is very dark in red. We mask B08 to only
    pixels where B04 is below the median of the whole crop, then return
    the location of the brightest masked B08 pixel.
    """
    water_thresh = np.percentile(water_arr_b04[water_arr_b04 > 0], 30)
    water_mask = water_arr_b04 < water_thresh
    if water_mask.sum() == 0:
        return -1, -1, 0.0
    masked = np.where(water_mask, arr_b08, -1)
    flat_idx = masked.argmax()
    y, x = np.unravel_index(flat_idx, masked.shape)
    return int(y), int(x), float(masked[y, x])


def find_head_near(arr: np.ndarray, seed_y: int, seed_x: int, radius: int = 25) -> tuple[float, float, float]:
    """Find the brightest pixel in a band within `radius` of (seed_y, seed_x).

    Returns sub-pixel centroid via 3x3 weighted mean around brightest pixel.
    """
    h, w = arr.shape
    y0 = max(0, seed_y - radius)
    y1 = min(h, seed_y + radius + 1)
    x0 = max(0, seed_x - radius)
    x1 = min(w, seed_x + radius + 1)
    sub = arr[y0:y1, x0:x1]
    if sub.size == 0:
        return float("nan"), float("nan"), 0.0

    # Use residual (subtract local water background) so the head pops
    bg = np.percentile(sub, 50)
    res = sub - bg
    res[res < 0] = 0
    if res.max() == 0:
        return float("nan"), float("nan"), 0.0

    # Brightest pixel in residual
    flat_idx = res.argmax()
    py, px = np.unravel_index(flat_idx, res.shape)
    peak_value = float(res[py, px])

    # Sub-pixel centroid via 3x3 weighted mean
    yc, xc, wsum = 0.0, 0.0, 0.0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            ny, nx_ = py + dy, px + dx
            if 0 <= ny < res.shape[0] and 0 <= nx_ < res.shape[1]:
                v = res[ny, nx_]
                if v > 0:
                    yc += (ny) * v
                    xc += (nx_) * v
                    wsum += v
    if wsum == 0:
        return float(py + y0), float(px + x0), peak_value
    yc = yc / wsum + y0
    xc = xc / wsum + x0
    return yc, xc, peak_value


def main():
    safe = find_safe()
    print(f"SAFE: {safe.name}")
    print(f"User point: {POINT_LAT} N, {POINT_LON} E")
    print(f"Search centre (shifted {EAST_SHIFT_M:.0f} m east): {SEARCH_LAT}, {SEARCH_LON}\n")

    img_data = next((safe / "GRANULE").glob("*")) / "IMG_DATA"

    bands = {}
    crops = {}
    crop_origin = None
    for band_id in ("B02", "B03", "B04", "B08"):
        jp2 = next(img_data.glob(f"*_{band_id}.jp2"))
        with rasterio.open(jp2) as src:
            row, col = world_to_pixel(src, SEARCH_LON, SEARCH_LAT)
            r0 = max(0, row - CROP_HALF_PX); r1 = r0 + 2 * CROP_HALF_PX
            c0 = max(0, col - CROP_HALF_PX); c1 = c0 + 2 * CROP_HALF_PX
            arr = src.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)
            bands[band_id] = arr
            crops[band_id] = (r0, c0)
            if crop_origin is None:
                crop_origin = (r0, c0)
                print(f"Crop origin (full raster): row={r0}, col={c0}, size={arr.shape}")

    # Find seed in B08 over water
    seed_y, seed_x, seed_v = find_head_in_water(bands["B08"], bands["B04"])
    print(f"\nB08 brightest over water: ({seed_y}, {seed_x}) value={seed_v:.0f}")

    # In each band, find brightest residual pixel near seed
    heads = {}
    for band_id in ("B02", "B03", "B04", "B08"):
        # Search radius scales with delta-t (Mach 1 max)
        dt = BAND_OFFSET_S[band_id]
        radius = max(20, int(dt * 343 / 10) + 10)  # px
        cy, cx, peak = find_head_near(bands[band_id], seed_y, seed_x, radius=radius)
        heads[band_id] = {
            "centroid_yx": [cy, cx],
            "peak_residual": peak,
            "search_radius_px": radius,
        }
        print(f"  {band_id}: head centroid = ({cy:.2f}, {cx:.2f}), peak residual = {peak:.0f}, radius = {radius}px")

    # Compute B02 -> B08 offset
    cy02, cx02 = heads["B02"]["centroid_yx"]
    cy08, cx08 = heads["B08"]["centroid_yx"]
    dy = cy08 - cy02
    dx = cx08 - cx02
    dist_px = math.hypot(dy, dx)
    dist_m = dist_px * 10.0
    speed = dist_m / DT_B02_B08
    print(f"\nB02 -> B08 head offset: dy={dy:+.2f}, dx={dx:+.2f}, |Δ|={dist_px:.2f} px = {dist_m:.0f} m")
    print(f"Aircraft ground speed (head-to-head): {speed:.0f} m/s = {speed*3.6:.0f} km/h = Mach {speed/343:.2f}")

    # Cross-check
    print("\nCross-check (offsets from B02 head):")
    for band_id in ("B03", "B04", "B08"):
        cy, cx = heads[band_id]["centroid_yx"]
        ddy = cy - cy02
        ddx = cx - cx02
        d = math.hypot(ddy, ddx)
        dt = BAND_OFFSET_S[band_id]
        v = d * 10.0 / dt if dt > 0 else float("nan")
        print(f"  {band_id}: Δt={dt:.3f}s, Δ=({ddy:+.2f}, {ddx:+.2f}) px, |Δ|={d:.2f} px, v={v:.0f} m/s = Mach {v/343:.2f}")

    # Render zoom panels around head
    from PIL import Image, ImageDraw
    HEAD_HALF = 30  # 30 px = 300 m around head
    head_y_int = int(round(cy02))
    head_x_int = int(round(cx02))
    py0 = max(0, head_y_int - HEAD_HALF)
    py1 = min(bands["B02"].shape[0], head_y_int + HEAD_HALF)
    px0 = max(0, head_x_int - HEAD_HALF)
    px1 = min(bands["B02"].shape[1], head_x_int + HEAD_HALF)
    crop_h = py1 - py0
    crop_w = px1 - px0
    upscale = 12

    pad = 16
    label_h = 30
    cell_w = crop_w * upscale
    cell_h = crop_h * upscale + label_h
    grid = Image.new("RGB", (cell_w * 2 + pad * 3, cell_h * 2 + pad * 3 + 50), (250, 250, 250))
    grid_draw = ImageDraw.Draw(grid)
    grid_draw.text((pad, 8), f"Head zoom: 12× pixel-zoom on {crop_w}×{crop_h} px window over water — band centroids ringed",
                   fill=(50, 50, 50))

    positions = [
        ("B02", 0, 0, (21, 101, 192), "B02 · 490 nm (blue) · t\u2080"),
        ("B03", 1, 0, (67, 160, 71), "B03 · 560 nm (green) · t\u2080+0.527s"),
        ("B04", 0, 1, (229, 57, 53), "B04 · 665 nm (red) · t\u2080+0.585s"),
        ("B08", 1, 1, (236, 64, 122), "B08 · 842 nm (NIR) · t\u2080+1.005s"),
    ]
    for band_id, gx, gy, color, lbl in positions:
        sub = bands[band_id][py0:py1, px0:px1]
        valid = sub > 0
        lo, hi = np.percentile(sub[valid], [1, 99]) if valid.sum() > 0 else (0, 1)
        norm = np.clip((sub - lo) / max(hi - lo, 1), 0, 1)
        gray = (norm * 255).astype(np.uint8)
        rgb_band = np.stack([gray, gray, gray], axis=-1)
        bimg = Image.fromarray(rgb_band).resize((cell_w, crop_h * upscale), Image.NEAREST)
        x0p = pad + gx * (cell_w + pad)
        y0p = 50 + pad + gy * (cell_h + pad) + label_h
        grid.paste(bimg, (x0p, y0p))
        grid_draw.text((x0p + 4, y0p - label_h + 6), lbl, fill=color)
        # Mark head centroid
        cy_b, cx_b = heads[band_id]["centroid_yx"]
        local_y = (cy_b - py0) * upscale
        local_x = (cx_b - px0) * upscale
        if 0 <= local_y < crop_h * upscale and 0 <= local_x < crop_w * upscale:
            r = 10
            grid_draw.ellipse([x0p + local_x - r, y0p + local_y - r,
                               x0p + local_x + r, y0p + local_y + r],
                              outline=color, width=2)

    head_path = OUT_DIR / "head_zoom.png"
    grid.save(head_path)
    print(f"\nHead zoom panel: {head_path} ({grid.width}x{grid.height})")
    grid.save(DOCS_DIR / "head_zoom.png")

    # RGB head closeup
    sub_b04 = bands["B04"][py0:py1, px0:px1]
    sub_b03 = bands["B03"][py0:py1, px0:px1]
    sub_b02 = bands["B02"][py0:py1, px0:px1]
    rgb_head = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    for ch_idx, ch_arr in enumerate([sub_b04, sub_b03, sub_b02]):
        valid = ch_arr > 0
        if valid.sum() == 0:
            continue
        lo, hi = np.percentile(ch_arr[valid], [1, 99])
        if hi <= lo:
            continue
        rgb_head[..., ch_idx] = (np.clip((ch_arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    rgb_img = Image.fromarray(rgb_head).resize((crop_w * upscale, crop_h * upscale), Image.NEAREST)
    rgb_path = OUT_DIR / "head_rgb.png"
    rgb_img.save(rgb_path)
    rgb_img.save(DOCS_DIR / "head_rgb.png")
    print(f"RGB head closeup: {rgb_path}")

    report = {
        "safe_archive": safe.name,
        "user_point": {"lat": POINT_LAT, "lon": POINT_LON},
        "search_centre": {"lat": SEARCH_LAT, "lon": SEARCH_LON, "east_shift_m": EAST_SHIFT_M},
        "delta_t_b02_b08_s": DT_B02_B08,
        "method": "find brightest B08 over water (B04 below 30th percentile), then find brightest residual within Mach-1 search radius in each band",
        "head_centroids": heads,
        "b02_b08_offset_px": dist_px,
        "b02_b08_offset_m": dist_m,
        "speed_m_s": speed,
        "speed_mach": speed / 343.0,
        "speed_km_h": speed * 3.6,
    }
    out = OUT_DIR / "head_over_water_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
