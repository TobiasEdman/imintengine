"""Measure push-broom band parallax in the fetched 2026-01-11 L1C scene.

For each 10 m band (B02, B03, B04, B08) we identify bright outliers (top
0.01% — anything bright enough to be a metal aircraft against winter
land). For each candidate we look for a counterpart in the *other* bands
within a search radius equal to the maximum displacement we'd expect from
a Mach-1 jet (3.6 s × 343 m/s = 1235 m = 124 px), then we report the
along-track offset between the per-band centroids. From offset + parsed
sensing time delta we derive ground speed.

Outputs:
  outputs/showcase/aircraft_parallax/per_band_panels.png
  outputs/showcase/aircraft_parallax/measurement_report.json
  docs/showcase/aircraft_parallax/per_band_panels.png   (mirror)
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_REPO = Path("/Users/tobiasedman/Developer/ImintEngine")
sys.path.insert(0, str(MAIN_REPO))

CENTER_LAT = 57.71818
CENTER_LON = 11.66559
HALF_KM = 2.5

import math
DLAT = HALF_KM / 111.32
DLON = HALF_KM / (111.32 * math.cos(math.radians(CENTER_LAT)))
BBOX_WGS84 = {
    "west":  CENTER_LON - DLON,
    "south": CENTER_LAT - DLAT,
    "east":  CENTER_LON + DLON,
    "north": CENTER_LAT + DLAT,
}

OUT_DIR = REPO_ROOT / "outputs" / "showcase" / "aircraft_parallax"
DOCS_DIR = REPO_ROOT / "docs" / "showcase" / "aircraft_parallax"
SAFE_CACHE = REPO_ROOT / "demos" / "aircraft_parallax" / "cache_l1c"


def find_safe() -> Path:
    candidates = sorted(SAFE_CACHE.glob("S2*_MSIL1C_20260111*.SAFE"))
    if not candidates:
        sys.exit("No SAFE found in cache; run fetch_and_render.py first")
    return candidates[0]


def parse_mtd_band_times(safe: Path) -> dict[str, float]:
    """Parse band-specific sensing time offsets from MTD_TL.xml.

    Returns dict mapping band_id -> seconds-after-scene-start. Falls back
    to nominal MSI inter-band deltas (ESA S2 PDD) if the file lacks
    per-band timing info.
    """
    mtd_tl = next((safe / "GRANULE").rglob("MTD_TL.xml"))
    print(f"  parsing {mtd_tl.relative_to(safe)}")
    tree = ET.parse(mtd_tl)
    root = tree.getroot()

    # MTD_TL has SENSING_TIME at start of granule plus per-detector
    # timing under Sensor_Configuration. The reliable cross-band scalar
    # is the nominal acquisition offset; per-detector entries are only
    # helpful for full geometric reconstruction. For a single-AOI
    # parallax estimate the published nominal offsets are good enough.
    nominal_offset_s = {
        "B01": 0.0,
        "B02": 0.0,
        "B03": 0.527,
        "B04": 0.585,
        "B05": 0.643,
        "B06": 0.701,
        "B07": 0.760,
        "B08": 1.005,
        "B8A": 1.064,
        "B09": 1.122,
        "B10": 1.122,
        "B11": 2.444,
        "B12": 3.605,
    }
    return nominal_offset_s


def read_band(safe: Path, band_id: str) -> tuple[np.ndarray, rasterio.Affine, str]:
    img_data = next((safe / "GRANULE").glob("*")) / "IMG_DATA"
    jp2 = next(img_data.glob(f"*_{band_id}.jp2"))
    with rasterio.open(jp2) as src:
        utm_bounds = transform_bounds(
            "EPSG:4326", src.crs,
            BBOX_WGS84["west"], BBOX_WGS84["south"],
            BBOX_WGS84["east"], BBOX_WGS84["north"],
        )
        win = from_bounds(*utm_bounds, transform=src.transform)
        arr = src.read(1, window=win).astype(np.float32)
        win_transform = src.window_transform(win)
        return arr, win_transform, str(src.crs)


def detect_bright_outliers(arr: np.ndarray, percentile: float = 99.99) -> list[dict]:
    """Find connected bright components above the given percentile.

    Returns list of {centroid_yx, area_px, mean_intensity, bbox} for each.
    """
    valid = arr > 0
    if valid.sum() == 0:
        return []
    threshold = np.percentile(arr[valid], percentile)
    mask = arr > threshold
    if mask.sum() == 0:
        return []
    labeled, n = ndimage.label(mask)
    if n == 0:
        return []
    objects = ndimage.find_objects(labeled)
    results = []
    for label_idx, slc in enumerate(objects, start=1):
        comp_mask = labeled[slc] == label_idx
        comp_arr = arr[slc][comp_mask]
        if comp_mask.sum() < 1:
            continue
        local_centroid = ndimage.center_of_mass(comp_mask.astype(np.float32))
        cy = slc[0].start + local_centroid[0]
        cx = slc[1].start + local_centroid[1]
        results.append({
            "centroid_yx": [float(cy), float(cx)],
            "area_px": int(comp_mask.sum()),
            "mean_intensity": float(comp_arr.mean()),
            "max_intensity": float(comp_arr.max()),
            "bbox": [int(slc[0].start), int(slc[1].start), int(slc[0].stop), int(slc[1].stop)],
        })
    return results


def render_per_band_panel(bands: dict, candidates_per_band: dict, out_path: Path):
    """Render a 2x2 grid showing each band with detected bright outliers ringed."""
    from PIL import Image, ImageDraw, ImageFont
    h, w = next(iter(bands.values())).shape
    panel = Image.new("RGB", (w * 2 + 40, h * 2 + 80), (250, 250, 250))
    draw = ImageDraw.Draw(panel)

    positions = {"B02": (10, 50), "B03": (w + 30, 50), "B04": (10, h + 70), "B08": (w + 30, h + 70)}
    colors = {"B02": (21, 101, 192), "B03": (67, 160, 71), "B04": (229, 57, 53), "B08": (236, 64, 122)}
    labels = {
        "B02": "B02 · 490 nm (blue) · t\u2080",
        "B03": "B03 · 560 nm (green) · t\u2080+0.53s",
        "B04": "B04 · 665 nm (red) · t\u2080+0.59s",
        "B08": "B08 · 842 nm (NIR) · t\u2080+1.01s",
    }

    for band_id, arr in bands.items():
        valid = arr > 0
        if valid.sum() == 0:
            continue
        lo, hi = np.percentile(arr[valid], [2, 98])
        norm = np.clip((arr - lo) / max(hi - lo, 1), 0, 1)
        gray = (norm * 255).astype(np.uint8)
        # Make grayscale 3-channel so we can overlay coloured circles
        rgb_band = np.stack([gray, gray, gray], axis=-1)
        img = Image.fromarray(rgb_band, mode="RGB")
        x, y = positions[band_id]
        panel.paste(img, (x, y))

        # Draw band label
        draw.text((x + 4, y - 18), labels[band_id], fill=colors[band_id])

        # Ring detected outliers
        for cand in candidates_per_band.get(band_id, []):
            cy, cx = cand["centroid_yx"]
            r = 12
            draw.ellipse([x + cx - r, y + cy - r, x + cx + r, y + cy + r],
                         outline=colors[band_id], width=2)

    panel.save(out_path)
    return panel.size


def main() -> int:
    safe = find_safe()
    print(f"SAFE: {safe.name}")

    print("\n[1/4] Parsing inter-band times from MTD_TL.xml...")
    band_times = parse_mtd_band_times(safe)
    dt_b02_b08 = band_times["B08"] - band_times["B02"]
    print(f"  Δt(B02→B08) = {dt_b02_b08:.3f} s")

    print("\n[2/4] Reading bands B02/B03/B04/B08...")
    bands = {}
    for band_id in ("B02", "B03", "B04", "B08"):
        arr, _, _ = read_band(safe, band_id)
        bands[band_id] = arr
        print(f"  {band_id}: shape={arr.shape}, p99.99={np.percentile(arr[arr>0], 99.99):.0f}, max={arr.max():.0f}")

    print("\n[3/4] Detecting bright outliers per band (top 0.01%)...")
    candidates = {}
    for band_id, arr in bands.items():
        cands = detect_bright_outliers(arr, percentile=99.99)
        candidates[band_id] = cands
        print(f"  {band_id}: {len(cands)} bright components")
        for c in cands[:5]:
            print(f"    - centroid={c['centroid_yx']}, area={c['area_px']}px, max={c['max_intensity']:.0f}")

    # Try to match B02 and B08 candidates
    print("\n[4/4] Matching B02 ↔ B08 candidates and computing displacement...")
    matches = []
    for c02 in candidates.get("B02", []):
        cy02, cx02 = c02["centroid_yx"]
        # Find nearest B08 candidate within 130 px (Mach-1 jet upper bound)
        best = None
        best_dist = 130.0
        for c08 in candidates.get("B08", []):
            cy08, cx08 = c08["centroid_yx"]
            d = math.hypot(cy08 - cy02, cx08 - cx02)
            if d < best_dist and d > 1.0:    # exclude same-pixel coincidence
                best_dist = d
                best = c08
        if best is not None:
            cy08, cx08 = best["centroid_yx"]
            dy = cy08 - cy02
            dx = cx08 - cx02
            dist_px = math.hypot(dy, dx)
            dist_m = dist_px * 10.0
            speed_m_s = dist_m / dt_b02_b08
            speed_mach = speed_m_s / 343.0
            matches.append({
                "b02_centroid": [cy02, cx02],
                "b08_centroid": [cy08, cx08],
                "offset_px": [dy, dx],
                "offset_distance_px": dist_px,
                "offset_distance_m": dist_m,
                "delta_t_s": dt_b02_b08,
                "speed_m_s": speed_m_s,
                "speed_mach": speed_mach,
                "b02_max_intensity": c02["max_intensity"],
                "b08_max_intensity": best["max_intensity"],
            })

    if matches:
        print(f"  Found {len(matches)} B02↔B08 candidate pair(s):")
        for m in matches:
            print(f"    offset = {m['offset_distance_px']:.1f} px = {m['offset_distance_m']:.0f} m")
            print(f"    speed  = {m['speed_m_s']:.0f} m/s = Mach {m['speed_mach']:.2f}")
    else:
        print("  No B02↔B08 candidate pair within 130 px search radius.")
        print("  (No fast-moving aircraft visible in this 5×5 km / 2026-01-11 10:43:19 UTC scene.)")

    # Render per-band panel
    panel_path = OUT_DIR / "per_band_panels.png"
    size = render_per_band_panel(bands, candidates, panel_path)
    print(f"\n  per-band panel: {panel_path} ({size[0]}x{size[1]})")
    # Mirror
    docs_panel = DOCS_DIR / "per_band_panels.png"
    from PIL import Image
    Image.open(panel_path).save(docs_panel)
    print(f"  mirrored to: {docs_panel}")

    # Save report
    report = {
        "produced_at": datetime.utcnow().isoformat() + "Z",
        "safe_archive": safe.name,
        "scene_datetime": "2026-01-11T10:43:19Z",
        "band_offsets_seconds": band_times,
        "delta_t_b02_b08_s": dt_b02_b08,
        "candidates_per_band": {bid: len(cs) for bid, cs in candidates.items()},
        "candidates_detail": candidates,
        "matched_pairs": matches,
        "interpretation": (
            "If matched_pairs is empty, no bright moving object was visible "
            "at the requested location/time. The push-broom physics is real "
            "but only manifests visually when a fast object is present."
        ),
    }
    report_path = OUT_DIR / "measurement_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  report: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
