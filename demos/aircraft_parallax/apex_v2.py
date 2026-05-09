"""Find contrail apex per band — robust version constrained to water.

The streak goes from upper-right (apex, narrow + bright) to lower-left
(tail, wide + faint) across the user's original 120x120 px crop. We:

1. Use the original crop (centred on user point 57.71809N, 11.66456E).
2. Mask to water only via B04.
3. Find the streak's principal axis from the union of all four bands'
   bright residual pixels (more pixels → more stable axis).
4. Per band: project water-pixels above 3·MAD onto axis. The APEX is the
   end with HIGHER mean intensity. We find sub-pixel apex by fitting a
   linear taper to the streak's perpendicular extent vs. axial position
   and intersecting upper/lower edges.
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

POINT_LAT = 57.71809
POINT_LON = 11.66456
# Use the SAME 120x120 crop the user already saw the streak in.
CROP_HALF_PX = 60

DT_B02_B08 = 1.005
BAND_OFFSET_S = {"B02": 0.0, "B03": 0.527, "B04": 0.585, "B08": 1.005}


def find_safe() -> Path:
    return sorted(SAFE_CACHE.glob("S2*_MSIL1C_20260111*.SAFE"))[0]


def world_to_pixel(src, lon, lat):
    xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
    return src.index(xs[0], ys[0])


def main():
    safe = find_safe()
    print(f"SAFE: {safe.name}")
    print(f"Centre on user point: {POINT_LAT} N, {POINT_LON} E\n")

    img_data = next((safe / "GRANULE").glob("*")) / "IMG_DATA"
    bands = {}
    for band_id in ("B02", "B03", "B04", "B08"):
        jp2 = next(img_data.glob(f"*_{band_id}.jp2"))
        with rasterio.open(jp2) as src:
            row, col = world_to_pixel(src, POINT_LON, POINT_LAT)
            r0 = max(0, row - CROP_HALF_PX); r1 = r0 + 2 * CROP_HALF_PX
            c0 = max(0, col - CROP_HALF_PX); c1 = c0 + 2 * CROP_HALF_PX
            arr = src.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)
            bands[band_id] = arr

    # Water mask via B04 (winter water has very low red reflectance)
    b04 = bands["B04"]
    valid = b04 > 0
    p30 = np.percentile(b04[valid], 30)
    water = b04 < p30
    print(f"Water pixels: {water.sum()} / {water.size} ({100*water.sum()/water.size:.1f}%)")

    # Step 1: principal axis from union of B04+B08 bright residual pixels (best contrast)
    union_mask = np.zeros_like(b04, dtype=bool)
    for band_id in ("B04", "B08", "B02"):
        arr = bands[band_id]
        bg = median_filter(arr, 21)
        res = arr - bg
        res[res < 0] = 0
        if not water.any():
            continue
        mad = np.median(np.abs(res[water] - np.median(res[water]))) + 1
        thr = 3 * mad
        union_mask |= (res > thr) & water

    print(f"Union streak mask: {union_mask.sum()} pixels")
    if union_mask.sum() < 30:
        print("Too few pixels in union mask — abort")
        return

    ys, xs = np.where(union_mask)
    coords = np.column_stack([ys, xs]).astype(np.float64)
    centroid = coords.mean(axis=0)
    coords_c = coords - centroid
    cov = np.cov(coords_c.T)
    evals, evecs = np.linalg.eigh(cov)
    u = evecs[:, 1] / np.linalg.norm(evecs[:, 1])
    v = np.array([-u[1], u[0]])
    print(f"Principal axis u = ({u[0]:+.3f}, {u[1]:+.3f}) (yx)")
    print(f"Centroid (yx) = ({centroid[0]:.2f}, {centroid[1]:.2f})")

    # Step 2: per-band, project water pixels with significant residual onto axis
    apex_per_band = {}
    for band_id, arr in bands.items():
        bg = median_filter(arr, 21)
        res = arr - bg
        res[res < 0] = 0
        if not water.any():
            apex_per_band[band_id] = {"error": "no water"}
            continue
        mad = np.median(np.abs(res[water] - np.median(res[water]))) + 1
        thr = 3 * mad
        bm = (res > thr) & water
        labels, n = label(bm)
        if n == 0:
            apex_per_band[band_id] = {"error": "no streak"}
            continue
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        biggest = sizes.argmax()
        bm = labels == biggest
        if bm.sum() < 15:
            apex_per_band[band_id] = {"error": f"only {bm.sum()} px"}
            continue

        ys_b, xs_b = np.where(bm)
        coords_b = np.column_stack([ys_b, xs_b]).astype(np.float64) - centroid
        s_b = coords_b @ u
        w_b = coords_b @ v
        intensities = res[ys_b, xs_b]

        # Bin along s and find upper/lower w-extents (only for bins with >=2 pixels)
        s_min, s_max = s_b.min(), s_b.max()
        nbins = max(8, int((s_max - s_min) / 2.5))
        edges = np.linspace(s_min, s_max, nbins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        bin_idx = np.clip(np.digitize(s_b, edges) - 1, 0, nbins - 1)

        upper = np.full(nbins, np.nan)
        lower = np.full(nbins, np.nan)
        intensity_per_bin = np.zeros(nbins)
        for i in range(nbins):
            mask_i = bin_idx == i
            if mask_i.sum() < 2:
                continue
            upper[i] = w_b[mask_i].max()
            lower[i] = w_b[mask_i].min()
            intensity_per_bin[i] = intensities[mask_i].mean()

        valid_b = (~np.isnan(upper)) & (intensity_per_bin > 0)
        if valid_b.sum() < 4:
            apex_per_band[band_id] = {"error": f"only {valid_b.sum()} valid bins"}
            continue

        # Fit linear edges
        sc = centres[valid_b]
        uw = upper[valid_b]
        lw = lower[valid_b]
        pu = np.polyfit(sc, uw, 1)
        pl = np.polyfit(sc, lw, 1)

        # Apex = intersection
        if abs(pu[0] - pl[0]) < 1e-6:
            apex_per_band[band_id] = {"error": "edges parallel"}
            continue
        s_apex = (pl[1] - pu[1]) / (pu[0] - pl[0])
        # Sanity: apex should be at one of the s extremes (where edges meet) — check
        # which extreme is the bright (head) end:
        intensity_low_s = intensities[s_b < (s_min + 0.3 * (s_max - s_min))].mean() if (s_b < (s_min + 0.3 * (s_max - s_min))).any() else 0
        intensity_high_s = intensities[s_b > (s_max - 0.3 * (s_max - s_min))].mean() if (s_b > (s_max - 0.3 * (s_max - s_min))).any() else 0
        head_end = "high_s" if intensity_high_s > intensity_low_s else "low_s"

        # The apex direction matters: if the streak narrows towards higher s, then s_apex should be > s_max
        w_apex = pu[0] * s_apex + pu[1]
        apex_coords_local = s_apex * u + w_apex * v + centroid
        apex_per_band[band_id] = {
            "apex_yx": apex_coords_local.tolist(),
            "s_apex": float(s_apex),
            "w_apex": float(w_apex),
            "s_range": [float(s_min), float(s_max)],
            "head_end": head_end,
            "intensity_low_s": float(intensity_low_s),
            "intensity_high_s": float(intensity_high_s),
            "n_streak_pixels": int(bm.sum()),
            "n_valid_bins": int(valid_b.sum()),
        }
        print(f"  {band_id}: apex (yx) = ({apex_coords_local[0]:.2f}, {apex_coords_local[1]:.2f}), "
              f"s_apex={s_apex:.1f} (s_range={s_min:.1f}..{s_max:.1f}), "
              f"head_end={head_end}, streak={bm.sum()} px, bins={valid_b.sum()}")

    # Compute offsets
    if "B02" in apex_per_band and "B08" in apex_per_band and "apex_yx" in apex_per_band["B02"] and "apex_yx" in apex_per_band["B08"]:
        a02 = apex_per_band["B02"]["apex_yx"]
        a08 = apex_per_band["B08"]["apex_yx"]
        dy = a08[0] - a02[0]
        dx = a08[1] - a02[1]
        d = math.hypot(dy, dx)
        v = d * 10.0 / DT_B02_B08
        print(f"\nB02→B08 apex offset: dy={dy:+.2f}, dx={dx:+.2f}, |Δ|={d:.2f} px = {d*10:.0f} m")
        print(f"Aircraft ground speed: {v:.0f} m/s = {v*3.6:.0f} km/h = Mach {v/343:.2f}")

    # Render apex visualisation on the existing zoom (8x like before)
    from PIL import Image, ImageDraw
    upscale = 8
    H, W = bands["B02"].shape
    pad = 16
    label_h = 30
    cell_w = W * upscale
    cell_h = H * upscale + label_h
    grid = Image.new("RGB", (cell_w * 2 + pad * 3, cell_h * 2 + pad * 3 + 50), (250, 250, 250))
    grid_draw = ImageDraw.Draw(grid)
    grid_draw.text((pad, 8),
                   "Apex tracking — narrow/bright end of contrail wedge per band. Coloured rings = each band's apex; offset between rings = aircraft displacement.",
                   fill=(50, 50, 50))

    band_colors = {
        "B02": (76, 156, 252),     # blue
        "B03": (76, 220, 100),     # green
        "B04": (252, 76, 76),      # red
        "B08": (236, 64, 122),     # NIR-magenta
    }
    band_layout = [
        ("B02", 0, 0, "B02 · 490 nm · t\u2080"),
        ("B03", 1, 0, "B03 · 560 nm · t\u2080+0.527s"),
        ("B04", 0, 1, "B04 · 665 nm · t\u2080+0.585s"),
        ("B08", 1, 1, "B08 · 842 nm · t\u2080+1.005s"),
    ]

    for band_id, gx, gy, lbl in band_layout:
        arr = bands[band_id]
        valid_b = arr > 0
        if valid_b.sum() == 0:
            continue
        lo, hi = np.percentile(arr[valid_b], [1, 99])
        norm = np.clip((arr - lo) / max(hi - lo, 1), 0, 1)
        gray = (norm * 255).astype(np.uint8)
        rgb_band = np.stack([gray, gray, gray], axis=-1)
        bimg = Image.fromarray(rgb_band).resize((cell_w, H * upscale), Image.NEAREST)
        x0p = pad + gx * (cell_w + pad)
        y0p = 50 + pad + gy * (cell_h + pad) + label_h
        grid.paste(bimg, (x0p, y0p))
        grid_draw.text((x0p + 4, y0p - label_h + 6), lbl, fill=band_colors[band_id])
        # Mark all bands' apexes in this panel for direct comparison
        for bid_overlay in ("B02", "B03", "B04", "B08"):
            d = apex_per_band.get(bid_overlay, {})
            if "apex_yx" not in d:
                continue
            ay, ax = d["apex_yx"]
            cy_p = ay * upscale + y0p
            cx_p = ax * upscale + x0p
            if 0 <= cy_p - y0p < H * upscale and 0 <= cx_p - x0p < cell_w:
                r = 12 if bid_overlay == band_id else 8
                w_line = 3 if bid_overlay == band_id else 2
                grid_draw.ellipse([cx_p - r, cy_p - r, cx_p + r, cy_p + r],
                                  outline=band_colors[bid_overlay], width=w_line)

    apex_path = OUT_DIR / "apex_v2.png"
    grid.save(apex_path)
    grid.save(DOCS_DIR / "apex_v2.png")
    print(f"\nApex panel: {apex_path}")

    report = {
        "safe_archive": safe.name,
        "user_point": {"lat": POINT_LAT, "lon": POINT_LON},
        "method": "Apex via wedge edge intersection, water-masked, principal axis from union of B02/B04/B08 bright residual",
        "apexes": apex_per_band,
    }
    if "B02" in apex_per_band and "B08" in apex_per_band and "apex_yx" in apex_per_band["B02"]:
        a02 = apex_per_band["B02"]["apex_yx"]
        a08 = apex_per_band["B08"]["apex_yx"]
        dy = a08[0] - a02[0]
        dx = a08[1] - a02[1]
        d = math.hypot(dy, dx)
        v = d * 10.0 / DT_B02_B08
        report["b02_b08_offset_px"] = d
        report["b02_b08_offset_m"] = d * 10
        report["speed_m_s"] = v
        report["speed_km_h"] = v * 3.6
        report["speed_mach"] = v / 343.0

    out = OUT_DIR / "apex_v2_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
