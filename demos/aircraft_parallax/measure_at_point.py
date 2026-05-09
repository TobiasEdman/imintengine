"""Measure parallax offset at the user-given aircraft point.

Aircraft point: 57.71809 N, 11.66456 E.

Approach: take the band crop (120x120 around the point), subtract a local
median background, threshold the residual, and find the streak centroid
in each band. The B02 -> B08 centroid offset converted by Δt = 1.005 s
gives the aircraft's ground speed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_CACHE = REPO_ROOT / "demos" / "aircraft_parallax" / "cache_l1c"
OUT_DIR = REPO_ROOT / "outputs" / "showcase" / "aircraft_parallax"

AIRCRAFT_LAT = 57.71809
AIRCRAFT_LON = 11.66456
CROP_HALF_PX = 60
DT_B02_B08 = 1.005  # seconds, from MTD_TL.xml

BAND_OFFSET_S = {"B02": 0.0, "B03": 0.527, "B04": 0.585, "B08": 1.005}


def find_safe() -> Path:
    return sorted(SAFE_CACHE.glob("S2*_MSIL1C_20260111*.SAFE"))[0]


def world_to_pixel(src, lon: float, lat: float) -> tuple[int, int]:
    xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
    return src.index(xs[0], ys[0])


def streak_residual(arr: np.ndarray) -> np.ndarray:
    """Local-background-subtracted residual emphasising linear bright streaks."""
    from scipy.ndimage import median_filter
    bg = median_filter(arr, size=15)
    res = arr - bg
    res[res < 0] = 0
    return res


def streak_centroid(arr: np.ndarray, top_n_pixels: int = 30) -> tuple[float, float, float]:
    """Take the brightest top-N residual pixels, return their intensity-weighted centroid + flag for confidence."""
    res = streak_residual(arr)
    if res.max() == 0:
        return float("nan"), float("nan"), 0.0
    flat = res.flatten()
    idx = np.argpartition(flat, -top_n_pixels)[-top_n_pixels:]
    rows, cols = np.unravel_index(idx, res.shape)
    weights = flat[idx]
    cy = float(np.average(rows, weights=weights))
    cx = float(np.average(cols, weights=weights))
    confidence = float(weights.mean())
    return cy, cx, confidence


def main():
    safe = find_safe()
    print(f"SAFE: {safe.name}")
    print(f"Aircraft point: {AIRCRAFT_LAT} N, {AIRCRAFT_LON} E\n")

    img_data = next((safe / "GRANULE").glob("*")) / "IMG_DATA"
    centroids = {}
    crops = {}
    for band_id in ("B02", "B03", "B04", "B08"):
        jp2 = next(img_data.glob(f"*_{band_id}.jp2"))
        with rasterio.open(jp2) as src:
            row, col = world_to_pixel(src, AIRCRAFT_LON, AIRCRAFT_LAT)
            r0 = max(0, row - CROP_HALF_PX)
            r1 = min(src.height, row + CROP_HALF_PX)
            c0 = max(0, col - CROP_HALF_PX)
            c1 = min(src.width, col + CROP_HALF_PX)
            arr = src.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)
            crops[band_id] = arr
            cy, cx, conf = streak_centroid(arr)
            centroids[band_id] = {
                "centroid_yx_local": [cy, cx],
                "confidence_residual": conf,
                "crop_origin_yx": [r0, c0],
            }
            print(f"  {band_id}: streak centroid (local) = ({cy:.2f}, {cx:.2f}), residual mean = {conf:.0f}")

    # Compute offsets from B02 to each band
    cy_b02, cx_b02 = centroids["B02"]["centroid_yx_local"]
    print(f"\nOffsets relative to B02:")
    offsets = {}
    for band_id in ("B03", "B04", "B08"):
        cy, cx = centroids[band_id]["centroid_yx_local"]
        dy = cy - cy_b02
        dx = cx - cx_b02
        dist_px = math.hypot(dy, dx)
        dist_m = dist_px * 10.0
        dt = BAND_OFFSET_S[band_id]
        speed = dist_m / dt if dt > 0 else float("nan")
        offsets[band_id] = {
            "delta_t_s": dt,
            "dy_px": dy,
            "dx_px": dx,
            "distance_px": dist_px,
            "distance_m": dist_m,
            "speed_m_s": speed,
            "speed_mach": speed / 343.0 if speed == speed else float("nan"),
        }
        print(f"  {band_id} (Δt={dt:.3f}s): dy={dy:+.2f}, dx={dx:+.2f}, "
              f"|Δ|={dist_px:.2f} px = {dist_m:.0f} m → "
              f"v = {speed:.0f} m/s = Mach {speed/343:.2f}")

    report = {
        "safe_archive": safe.name,
        "aircraft_point_wgs84": {"lat": AIRCRAFT_LAT, "lon": AIRCRAFT_LON},
        "band_offsets_seconds": BAND_OFFSET_S,
        "centroids": centroids,
        "offsets_from_B02": offsets,
        "interpretation": (
            "Streak centroid is the intensity-weighted mean of the top 30 residual "
            "pixels per band crop (residual = arr - 15px median filter). "
            "The B02->B08 offset divided by Δt = 1.005 s yields the aircraft "
            "ground speed."
        ),
    }
    out = OUT_DIR / "measurement_at_point.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
