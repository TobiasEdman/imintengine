"""Track the contrail head (= aircraft body) per band, NOT the streak centroid.

Approach:
1. Crop a wider 240x240 px window so the entire streak is visible.
2. For each band, compute residual = arr - 21px median(arr).
3. Threshold residual at 3·MAD.
4. Skeletonize the streak and compute its principal axis (PCA).
5. Project all streak points onto the principal axis.
6. The streak's HEAD is one of the two extremes. To pick which extreme is
   the head, we use a property of contrails: contrails dissipate with age.
   Younger (newer) parts are denser/brighter. The head (= aircraft) is the
   brightest extreme.
7. The B02 -> B08 head offset / Δt = aircraft ground speed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
from scipy.ndimage import median_filter, label, binary_dilation

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_CACHE = REPO_ROOT / "demos" / "aircraft_parallax" / "cache_l1c"
OUT_DIR = REPO_ROOT / "outputs" / "showcase" / "aircraft_parallax"
DOCS_DIR = REPO_ROOT / "docs" / "showcase" / "aircraft_parallax"

AIRCRAFT_LAT = 57.71809
AIRCRAFT_LON = 11.66456
CROP_HALF_PX = 120
DT_B02_B08 = 1.005
BAND_OFFSET_S = {"B02": 0.0, "B03": 0.527, "B04": 0.585, "B08": 1.005}


def find_safe() -> Path:
    return sorted(SAFE_CACHE.glob("S2*_MSIL1C_20260111*.SAFE"))[0]


def world_to_pixel(src, lon, lat):
    xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
    return src.index(xs[0], ys[0])


def extract_streak(arr: np.ndarray):
    """Return (head_yx, tail_yx, axis_unit_vec, residual_arr, mask)."""
    bg = median_filter(arr, 21)
    res = arr - bg
    res[res < 0] = 0

    if res.max() == 0:
        return None

    mad = np.median(np.abs(res - np.median(res))) + 1
    threshold = 4 * mad
    mask = res > threshold

    # Keep only the largest connected component
    labels, n = label(mask)
    if n == 0:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    biggest = sizes.argmax()
    streak_mask = labels == biggest

    if streak_mask.sum() < 10:
        return None

    ys, xs = np.where(streak_mask)
    coords = np.column_stack([ys, xs]).astype(np.float64)
    centroid = coords.mean(axis=0)
    coords_c = coords - centroid

    # Principal axis via SVD
    cov = np.cov(coords_c.T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, 1]  # axis of largest variance (streak direction)
    axis_norm = axis / np.linalg.norm(axis)

    # Project each point onto axis
    projections = coords_c @ axis_norm

    # Sort projection extremes
    idx_min = projections.argmin()
    idx_max = projections.argmax()

    end_a = (ys[idx_min], xs[idx_min])
    end_b = (ys[idx_max], xs[idx_max])

    # Pick the brighter end as the HEAD (aircraft).
    # Average residual within 5 px around each end:
    def end_brightness(end):
        ey, ex = end
        rsum = 0.0
        n = 0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx_ = ey + dy, ex + dx
                if 0 <= ny < res.shape[0] and 0 <= nx_ < res.shape[1]:
                    if streak_mask[ny, nx_]:
                        rsum += res[ny, nx_]
                        n += 1
        return rsum / max(n, 1)

    b_a = end_brightness(end_a)
    b_b = end_brightness(end_b)
    if b_a >= b_b:
        head, tail = end_a, end_b
        head_brightness, tail_brightness = b_a, b_b
    else:
        head, tail = end_b, end_a
        head_brightness, tail_brightness = b_b, b_a

    return {
        "head_yx": head,
        "tail_yx": tail,
        "axis_unit": axis_norm.tolist(),
        "centroid_yx": centroid.tolist(),
        "head_brightness": float(head_brightness),
        "tail_brightness": float(tail_brightness),
        "n_pixels": int(streak_mask.sum()),
        "streak_length_px": float(projections.max() - projections.min()),
        "mask": streak_mask,
        "residual": res,
    }


def main():
    safe = find_safe()
    print(f"SAFE: {safe.name}")
    print(f"Aircraft point: {AIRCRAFT_LAT} N, {AIRCRAFT_LON} E\n")

    img_data = next((safe / "GRANULE").glob("*")) / "IMG_DATA"
    band_results = {}

    for band_id in ("B02", "B03", "B04", "B08"):
        jp2 = next(img_data.glob(f"*_{band_id}.jp2"))
        with rasterio.open(jp2) as src:
            row, col = world_to_pixel(src, AIRCRAFT_LON, AIRCRAFT_LAT)
            r0 = max(0, row - CROP_HALF_PX); r1 = r0 + 2 * CROP_HALF_PX
            c0 = max(0, col - CROP_HALF_PX); c1 = c0 + 2 * CROP_HALF_PX
            arr = src.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)

        result = extract_streak(arr)
        if result is None:
            print(f"  {band_id}: no streak detected")
            continue
        band_results[band_id] = {
            "head_yx": result["head_yx"],
            "tail_yx": result["tail_yx"],
            "head_brightness": result["head_brightness"],
            "tail_brightness": result["tail_brightness"],
            "streak_length_px": result["streak_length_px"],
            "n_pixels": result["n_pixels"],
            "axis_unit": result["axis_unit"],
        }
        print(f"  {band_id}:")
        print(f"     head pixel: {result['head_yx']}, head brightness res = {result['head_brightness']:.0f}")
        print(f"     tail pixel: {result['tail_yx']}, tail brightness res = {result['tail_brightness']:.0f}")
        print(f"     streak length = {result['streak_length_px']:.1f} px = {result['streak_length_px']*10:.0f} m")
        print(f"     mask size = {result['n_pixels']} px")

    # Compute aircraft speed from B02 -> B08 head offset
    if "B02" in band_results and "B08" in band_results:
        h02 = band_results["B02"]["head_yx"]
        h08 = band_results["B08"]["head_yx"]
        dy = h08[0] - h02[0]
        dx = h08[1] - h02[1]
        dist_px = math.hypot(dy, dx)
        dist_m = dist_px * 10.0
        speed = dist_m / DT_B02_B08
        speed_mach = speed / 343.0
        speed_kmh = speed * 3.6
        print(f"\nB02 -> B08 head offset: dy={dy:+d}, dx={dx:+d}, |Δ|={dist_px:.2f} px = {dist_m:.0f} m")
        print(f"Aircraft ground speed (head-to-head): {speed:.0f} m/s = {speed_kmh:.0f} km/h = Mach {speed_mach:.2f}")

    # Also report B04->B08 and B03->B08 for cross-check
    print("\nCross-check (offsets between heads, all relative to B02):")
    if "B02" in band_results:
        h02 = band_results["B02"]["head_yx"]
        rows = []
        for band_id in ("B03", "B04", "B08"):
            if band_id not in band_results:
                continue
            h = band_results[band_id]["head_yx"]
            dy = h[0] - h02[0]
            dx = h[1] - h02[1]
            dist_px = math.hypot(dy, dx)
            dist_m = dist_px * 10.0
            dt = BAND_OFFSET_S[band_id]
            v = dist_m / dt if dt > 0 else float("nan")
            rows.append({
                "band": band_id, "delta_t_s": dt, "dy_px": dy, "dx_px": dx,
                "distance_px": dist_px, "distance_m": dist_m,
                "speed_m_s": v, "speed_mach": v / 343.0 if v == v else float("nan"),
            })
            print(f"  {band_id}: Δt={dt:.3f}s, Δ=({dy:+d},{dx:+d}) px, |Δ|={dist_px:.2f} px, v={v:.0f} m/s, Mach {v/343:.2f}")

        report = {
            "safe_archive": safe.name,
            "aircraft_point": {"lat": AIRCRAFT_LAT, "lon": AIRCRAFT_LON},
            "delta_t_b02_b08_s": DT_B02_B08,
            "method": "track contrail HEAD (brightest extreme of streak axis) per band",
            "head_offsets_from_B02": rows,
            "band_results": band_results,
        }
        out = OUT_DIR / "head_tracking_report.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
