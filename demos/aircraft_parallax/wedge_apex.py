"""Find the contrail wedge apex per band: aircraft position via edge intersection.

A contrail is a wedge — narrow at the aircraft (newly formed) and wider
behind it as the trail diffuses. Tracking the wedge APEX (= the point
where the upper and lower edges intersect) gives the aircraft's actual
position at each band exposure, free of:

  * intensity-weighted centroid bias (which depends on contrail age profile)
  * contrail wind drift (the entire trail drifts together; apex tracks
    the aircraft)

Method per band:
  1. Background-subtract (median filter, large kernel).
  2. Threshold residual at 3·MAD → binary streak mask.
  3. Keep largest connected component.
  4. PCA on mask points → principal axis u, perpendicular axis v.
  5. Project all streak pixels onto (s, w) where s = along-axis, w = perp.
  6. Bin s; for each bin compute upper edge (max w) and lower edge (min w).
  7. Linear-fit each edge.
  8. Apex = intersection of the two edge lines.

Then B02→B08 apex offset / Δt = aircraft ground speed.
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

# Wide window centred east of the user point so we capture the wedge
# (head over water + tail towards original point).
EAST_SHIFT_M = 600.0
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


def water_only_mask(arr_b04: np.ndarray) -> np.ndarray:
    """True where B04 is in lowest 40% (= dark = water in winter)."""
    valid = arr_b04 > 0
    if valid.sum() == 0:
        return np.zeros_like(arr_b04, dtype=bool)
    threshold = np.percentile(arr_b04[valid], 40)
    return arr_b04 < threshold


def fit_wedge_apex(arr: np.ndarray, water_mask: np.ndarray, debug_label: str = "") -> dict:
    """Find the wedge apex in `arr` restricted to water_mask region."""
    # Residual: subtract water background only inside water region
    bg = float(np.median(arr[water_mask])) if water_mask.any() else 0.0
    res = arr - bg
    res = np.where(water_mask, res, 0.0)
    res[res < 0] = 0

    if res.max() == 0:
        return {"error": "no signal over water", "label": debug_label}

    mad = np.median(np.abs(res[res > 0] - np.median(res[res > 0]))) + 1
    threshold = 3 * mad
    mask = res > threshold
    labels, n = label(mask)
    if n == 0:
        return {"error": "no streak components", "label": debug_label}
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    biggest = sizes.argmax()
    streak_mask = labels == biggest
    if streak_mask.sum() < 30:
        return {"error": f"streak too small ({streak_mask.sum()} px)", "label": debug_label}

    ys, xs = np.where(streak_mask)
    coords = np.column_stack([ys, xs]).astype(np.float64)
    centroid = coords.mean(axis=0)
    coords_c = coords - centroid

    cov = np.cov(coords_c.T)
    evals, evecs = np.linalg.eigh(cov)
    u = evecs[:, 1] / np.linalg.norm(evecs[:, 1])  # along-streak
    v = np.array([-u[1], u[0]])                     # perpendicular

    s = coords_c @ u
    w = coords_c @ v

    # Bin s into fixed-width bins and find upper/lower edges per bin
    s_min, s_max = s.min(), s.max()
    nbins = int(s_max - s_min) // 3 + 1  # ~3 px per bin
    if nbins < 5:
        return {"error": "too few bins", "label": debug_label}
    bin_edges = np.linspace(s_min, s_max, nbins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(s, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    upper_w = np.full(nbins, np.nan)
    lower_w = np.full(nbins, np.nan)
    counts = np.zeros(nbins)
    for i in range(nbins):
        mask_i = bin_idx == i
        if mask_i.sum() == 0:
            continue
        upper_w[i] = w[mask_i].max()
        lower_w[i] = w[mask_i].min()
        counts[i] = mask_i.sum()

    valid_bins = ~np.isnan(upper_w) & (counts >= 1)
    if valid_bins.sum() < 5:
        return {"error": "too few valid bins", "label": debug_label}

    sc = bin_centres[valid_bins]
    uw = upper_w[valid_bins]
    lw = lower_w[valid_bins]

    # Linear fit upper and lower edges
    pu = np.polyfit(sc, uw, 1)  # uw = pu[0]*s + pu[1]
    pl = np.polyfit(sc, lw, 1)

    # Intersection: pu[0]*s + pu[1] = pl[0]*s + pl[1]
    # s_apex = (pl[1] - pu[1]) / (pu[0] - pl[0])
    if abs(pu[0] - pl[0]) < 1e-9:
        return {"error": "edges parallel", "label": debug_label}
    s_apex = (pl[1] - pu[1]) / (pu[0] - pl[0])
    w_apex = pu[0] * s_apex + pu[1]  # = pl[0]*s_apex + pl[1] (should match)

    apex_coords_c = s_apex * u + w_apex * v
    apex_yx = apex_coords_c + centroid
    return {
        "apex_yx": apex_yx.tolist(),
        "axis_u": u.tolist(),
        "axis_v": v.tolist(),
        "centroid_yx": centroid.tolist(),
        "s_range": [float(s_min), float(s_max)],
        "n_streak_pixels": int(streak_mask.sum()),
        "n_valid_bins": int(valid_bins.sum()),
        "upper_edge_slope": float(pu[0]),
        "upper_edge_intercept": float(pu[1]),
        "lower_edge_slope": float(pl[0]),
        "lower_edge_intercept": float(pl[1]),
        "s_apex": float(s_apex),
        "w_apex": float(w_apex),
        "label": debug_label,
        "_streak_mask": streak_mask,
        "_residual": res,
        "_sc": sc, "_uw": uw, "_lw": lw,
    }


def main():
    safe = find_safe()
    print(f"SAFE: {safe.name}")
    print(f"User point: {POINT_LAT} N, {POINT_LON} E")
    print(f"Search centre (shifted {EAST_SHIFT_M:.0f} m east): {SEARCH_LAT}, {SEARCH_LON}\n")

    img_data = next((safe / "GRANULE").glob("*")) / "IMG_DATA"
    bands = {}
    for band_id in ("B02", "B03", "B04", "B08"):
        jp2 = next(img_data.glob(f"*_{band_id}.jp2"))
        with rasterio.open(jp2) as src:
            row, col = world_to_pixel(src, SEARCH_LON, SEARCH_LAT)
            r0 = max(0, row - CROP_HALF_PX); r1 = r0 + 2 * CROP_HALF_PX
            c0 = max(0, col - CROP_HALF_PX); c1 = c0 + 2 * CROP_HALF_PX
            arr = src.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)
            bands[band_id] = arr
            print(f"  {band_id}: shape={arr.shape}")

    water = water_only_mask(bands["B04"])
    print(f"\nWater pixels in crop: {water.sum()} / {water.size} ({100*water.sum()/water.size:.1f}%)\n")

    apexes = {}
    for band_id, arr in bands.items():
        result = fit_wedge_apex(arr, water, debug_label=band_id)
        if "error" in result:
            print(f"  {band_id}: ERROR {result['error']}")
            continue
        apexes[band_id] = result
        print(f"  {band_id}: apex pixel = ({result['apex_yx'][0]:.2f}, {result['apex_yx'][1]:.2f}), "
              f"streak {result['n_streak_pixels']} px, {result['n_valid_bins']} bins")

    # Compute B02 -> B08 apex offset
    if "B02" in apexes and "B08" in apexes:
        a02 = apexes["B02"]["apex_yx"]
        a08 = apexes["B08"]["apex_yx"]
        dy = a08[0] - a02[0]
        dx = a08[1] - a02[1]
        dist_px = math.hypot(dy, dx)
        dist_m = dist_px * 10.0
        speed = dist_m / DT_B02_B08
        print(f"\nB02 -> B08 apex offset: dy={dy:+.2f}, dx={dx:+.2f}, |Δ|={dist_px:.2f} px = {dist_m:.0f} m")
        print(f"Aircraft ground speed: {speed:.0f} m/s = {speed*3.6:.0f} km/h = Mach {speed/343:.2f}")

        # Cross-check
        print("\nCross-check (apex offsets relative to B02):")
        rows = []
        for band_id in ("B03", "B04", "B08"):
            if band_id not in apexes:
                continue
            a = apexes[band_id]["apex_yx"]
            ddy = a[0] - a02[0]
            ddx = a[1] - a02[1]
            d = math.hypot(ddy, ddx)
            dt = BAND_OFFSET_S[band_id]
            v = d * 10.0 / dt if dt > 0 else float("nan")
            rows.append({
                "band": band_id, "delta_t_s": dt,
                "dy_px": float(ddy), "dx_px": float(ddx),
                "distance_px": float(d), "distance_m": float(d * 10),
                "speed_m_s": float(v), "speed_mach": float(v / 343.0),
            })
            print(f"  {band_id}: Δt={dt:.3f}s, Δ=({ddy:+.2f},{ddx:+.2f}) px, |Δ|={d:.2f} px, "
                  f"v={v:.0f} m/s = {v*3.6:.0f} km/h = Mach {v/343:.2f}")

    # Render apex visualisation
    from PIL import Image, ImageDraw
    upscale = 4
    H, W = bands["B02"].shape
    pad = 16
    label_h = 30
    cell_w = W * upscale
    cell_h = H * upscale + label_h
    grid = Image.new("RGB", (cell_w * 2 + pad * 3, cell_h * 2 + pad * 3 + 50), (250, 250, 250))
    grid_draw = ImageDraw.Draw(grid)
    grid_draw.text((pad, 8),
                   f"Wedge-apex tracking: edges fitted, apex = intersection. B02 yellow → B08 magenta = aircraft displacement.",
                   fill=(50, 50, 50))

    positions = [
        ("B02", 0, 0, (255, 235, 59), "B02 · 490 nm · t\u2080"),
        ("B03", 1, 0, (76, 175, 80), "B03 · 560 nm · t\u2080+0.527s"),
        ("B04", 0, 1, (244, 67, 54), "B04 · 665 nm · t\u2080+0.585s"),
        ("B08", 1, 1, (236, 64, 122), "B08 · 842 nm · t\u2080+1.005s"),
    ]

    for band_id, gx, gy, color, lbl in positions:
        arr = bands[band_id]
        valid = arr > 0
        if valid.sum() == 0:
            continue
        lo, hi = np.percentile(arr[valid], [2, 98])
        norm = np.clip((arr - lo) / max(hi - lo, 1), 0, 1)
        gray = (norm * 255).astype(np.uint8)
        rgb_band = np.stack([gray, gray, gray], axis=-1)
        bimg = Image.fromarray(rgb_band).resize((cell_w, H * upscale), Image.NEAREST)
        x0p = pad + gx * (cell_w + pad)
        y0p = 50 + pad + gy * (cell_h + pad) + label_h
        grid.paste(bimg, (x0p, y0p))
        grid_draw.text((x0p + 4, y0p - label_h + 6), lbl, fill=color)
        if band_id in apexes:
            ay, ax = apexes[band_id]["apex_yx"]
            # Draw all four band apex points overlaid as a reference (B02 yellow + this band)
            for ref_band, _, _, ref_color, _ in positions:
                if ref_band in apexes:
                    ry, rx = apexes[ref_band]["apex_yx"]
                    cy_p = ry * upscale + y0p
                    cx_p = rx * upscale + x0p
                    if 0 <= cy_p - y0p < H * upscale and 0 <= cx_p - x0p < cell_w:
                        r = 6
                        grid_draw.ellipse([cx_p - r, cy_p - r, cx_p + r, cy_p + r],
                                          outline=ref_color, width=2)

    apex_path = OUT_DIR / "wedge_apex.png"
    grid.save(apex_path)
    grid.save(DOCS_DIR / "wedge_apex.png")
    print(f"\nApex visualisation: {apex_path}")

    # Save report
    report = {
        "safe_archive": safe.name,
        "user_point_wgs84": {"lat": POINT_LAT, "lon": POINT_LON},
        "search_centre_wgs84": {"lat": SEARCH_LAT, "lon": SEARCH_LON, "east_shift_m": EAST_SHIFT_M},
        "delta_t_b02_b08_s": DT_B02_B08,
        "method": (
            "Per band: water-mask via B04 < 40th-percentile; residual = arr - water-bg; "
            "threshold at 3·MAD; largest connected component; PCA principal axis; "
            "project pixels to (s along, w perp); bin s; fit linear upper/lower edges; "
            "apex = intersection. Aircraft displacement B02→B08 / Δt = ground speed."
        ),
        "apexes": {bid: {k: v for k, v in d.items() if not k.startswith("_")} for bid, d in apexes.items()},
        "offsets_from_B02": rows if "B02" in apexes else [],
    }
    out = OUT_DIR / "wedge_apex_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
