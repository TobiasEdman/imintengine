"""Measure aircraft body size and characterise the streak vs head.

A jet aircraft at cruise typically has a 30-70 m fuselage. If the bright
"head" of the streak is the aircraft body it should be 1-7 pixels at 10
m/px. The thin extending tail is the condensation trail.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_CACHE = REPO_ROOT / "demos" / "aircraft_parallax" / "cache_l1c"
AIRCRAFT_LAT = 57.71809
AIRCRAFT_LON = 11.66456
CROP_HALF_PX = 60


def find_safe() -> Path:
    return sorted(SAFE_CACHE.glob("S2*_MSIL1C_20260111*.SAFE"))[0]


def world_to_pixel(src, lon, lat):
    xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
    return src.index(xs[0], ys[0])


def main():
    safe = find_safe()
    print(f"SAFE: {safe.name}\n")
    img_data = next((safe / "GRANULE").glob("*")) / "IMG_DATA"

    for band_id in ("B02", "B03", "B04", "B08"):
        jp2 = next(img_data.glob(f"*_{band_id}.jp2"))
        with rasterio.open(jp2) as src:
            row, col = world_to_pixel(src, AIRCRAFT_LON, AIRCRAFT_LAT)
            r0 = max(0, row - CROP_HALF_PX); r1 = r0 + 2 * CROP_HALF_PX
            c0 = max(0, col - CROP_HALF_PX); c1 = c0 + 2 * CROP_HALF_PX
            arr = src.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)

        from scipy.ndimage import median_filter, label
        bg = median_filter(arr, 15)
        res = arr - bg
        res[res < 0] = 0

        # Threshold for streak presence: 3*MAD above zero
        mad = np.median(np.abs(res - np.median(res))) + 1
        thresh = 3 * mad
        mask = res > thresh
        labeled, n = label(mask)
        # Find largest component (= the streak)
        if n == 0:
            print(f"  {band_id}: no significant components above threshold")
            continue

        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        biggest_id = sizes.argmax()
        biggest_mask = labeled == biggest_id
        biggest_size = biggest_mask.sum()

        # Find streak bounding box dims
        ys, xs = np.where(biggest_mask)
        bbox_h = ys.max() - ys.min() + 1
        bbox_w = xs.max() - xs.min() + 1
        # Length = max dimension of bounding box (streak diagonal)
        # More accurate: PCA on the masked points
        coords = np.column_stack([ys, xs]).astype(np.float64)
        coords -= coords.mean(axis=0)
        cov = np.cov(coords.T)
        evals = np.linalg.eigvalsh(cov)
        streak_length_px = 4 * np.sqrt(evals[1])  # 2 sigma each side
        streak_width_px = 4 * np.sqrt(evals[0])

        # Look for the brightest point in the streak — that's the aircraft head
        head_idx = res[biggest_mask].argmax()
        head_y, head_x = ys[head_idx], xs[head_idx]
        head_intensity = res[head_y, head_x]

        # Compactness around head: count pixels above threshold within 3 px of head
        ny = np.arange(max(0, head_y-3), min(arr.shape[0], head_y+4))
        nx = np.arange(max(0, head_x-3), min(arr.shape[1], head_x+4))
        Y, X = np.meshgrid(ny, nx, indexing="ij")
        head_neighbourhood = res[Y, X]
        head_compact_mask = head_neighbourhood > 0.5 * head_intensity
        head_size_px = head_compact_mask.sum()

        print(f"  {band_id}:")
        print(f"     largest component: {biggest_size} px above 3·MAD threshold")
        print(f"     bounding box: {bbox_h} × {bbox_w} px")
        print(f"     streak length (PCA, 4σ): {streak_length_px:.1f} px = {streak_length_px*10:.0f} m")
        print(f"     streak width  (PCA, 4σ): {streak_width_px:.1f} px = {streak_width_px*10:.0f} m")
        print(f"     brightest point at ({head_y}, {head_x}), residual = {head_intensity:.0f}")
        print(f"     compact 'head' size around brightest: {head_size_px} px above 50% peak")
        print()


if __name__ == "__main__":
    main()
