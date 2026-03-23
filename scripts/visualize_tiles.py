#!/usr/bin/env python3
"""Generate an HTML page visualizing tiles with 4 seasonal images + NMD labels.

Picks representative tiles for each class and renders them as an interactive
HTML page with base64-embedded images.

Usage:
    python scripts/visualize_tiles.py --data-dir data/lulc_seasonal_vpp --output tile_viewer.html
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.class_schema import get_class_names

# NMD class colors (RGB) for label overlay
LABEL_COLORS = {
    0: (40, 40, 40),        # background - dark gray
    1: (0, 100, 0),         # forest_pine - dark green
    2: (0, 60, 0),          # forest_spruce - very dark green
    3: (144, 238, 144),     # forest_deciduous - light green
    4: (34, 139, 34),       # forest_mixed - forest green
    5: (210, 180, 60),      # forest_temp_non_forest - olive/yellow
    6: (85, 107, 47),       # forest_wetland - dark olive
    7: (139, 90, 43),       # open_wetland - brown
    8: (255, 215, 0),       # cropland - gold
    9: (210, 180, 140),     # open_land - tan
    10: (255, 69, 0),       # developed - red-orange
    11: (255, 0, 0),        # buildings - red
    12: (0, 100, 255),      # water - blue
}


def array_to_png_b64(arr: np.ndarray) -> str:
    """Convert a (H,W,3) uint8 array to a base64 PNG string."""
    from PIL import Image
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_rgb(image: np.ndarray, frame_idx: int, n_bands: int = 6) -> np.ndarray:
    """Extract RGB from a multitemporal image for a given frame.

    Image shape: (T*n_bands, H, W) interleaved.
    Bands are B02,B03,B04,B08,B11,B12 — use B04(R),B03(G),B02(B).
    """
    offset = frame_idx * n_bands
    if offset + 3 > image.shape[0]:
        return np.zeros((image.shape[1], image.shape[2], 3), dtype=np.uint8)

    # B04=index 2 (red), B03=index 1 (green), B02=index 0 (blue)
    r = image[offset + 2]  # B04
    g = image[offset + 1]  # B03
    b = image[offset + 0]  # B02

    # Stack and scale to 0-255 (reflectance is typically 0-1 float32)
    rgb = np.stack([r, g, b], axis=-1)
    if rgb.dtype == np.float32 or rgb.dtype == np.float64:
        # Clip and scale
        rgb = np.clip(rgb * 3.5 * 255, 0, 255).astype(np.uint8)
    elif rgb.max() > 255:
        # uint16 sentinel data
        rgb = np.clip(rgb / 30, 0, 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)

    return rgb


def render_labels(label: np.ndarray) -> np.ndarray:
    """Convert label array to RGB using class colors."""
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in LABEL_COLORS.items():
        mask = label == cls_id
        rgb[mask] = color
    return rgb


def find_representative_tiles(tiles_dir: Path, num_classes: int, per_class: int = 3):
    """Find tiles that are good representatives for each class.

    Picks tiles where the class has the highest fraction.
    """
    class_names = get_class_names(num_classes)
    tiles = sorted(tiles_dir.glob("tile_*.npz"))

    # For each class, find tiles where it's most dominant
    class_tiles = defaultdict(list)  # class_id -> [(fraction, tile_path), ...]

    for t in tiles:
        d = np.load(t)
        label = d["label"]
        total = label.size
        uniq, counts = np.unique(label, return_counts=True)
        for u, c in zip(uniq, counts):
            u = int(u)
            if u == 0:
                continue
            frac = float(c) / total
            if frac > 0.05:  # at least 5% of tile
                class_tiles[u].append((frac, t))

    # Pick top per_class tiles for each class
    result = {}
    for cls_id in range(1, num_classes + 1):
        candidates = class_tiles.get(cls_id, [])
        candidates.sort(key=lambda x: x[0], reverse=True)
        # Pick diverse tiles (not all from same area)
        picked = []
        seen_areas = set()
        for frac, path in candidates:
            # Use grid area (first 3 digits of easting) as diversity key
            area_key = path.stem[:15]
            if area_key not in seen_areas or len(picked) < per_class:
                picked.append((frac, path))
                seen_areas.add(area_key)
            if len(picked) >= per_class:
                break
        result[cls_id] = picked

    return result


def generate_html(tiles_dir: Path, num_classes: int, output_path: Path, per_class: int = 3):
    """Generate the HTML visualization page."""
    class_names = get_class_names(num_classes)
    reps = find_representative_tiles(tiles_dir, num_classes, per_class)

    # Build HTML
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LULC Tile Viewer — %d-class</title>
<style>
body { font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }
h1 { text-align: center; color: #e94560; }
h2 { color: #0f3460; background: #e94560; padding: 8px 16px; border-radius: 6px; margin-top: 40px; }
.class-section { margin-bottom: 30px; }
.tile-row { display: flex; gap: 6px; align-items: flex-start; margin-bottom: 16px; background: #16213e; padding: 10px; border-radius: 8px; }
.tile-row img { width: 180px; height: 180px; image-rendering: pixelated; border-radius: 4px; }
.tile-info { min-width: 200px; font-size: 13px; padding: 4px 10px; }
.tile-info .name { font-weight: bold; color: #e94560; font-size: 14px; }
.tile-info .detail { color: #999; margin-top: 2px; }
.frame-label { text-align: center; font-size: 11px; color: #888; }
.legend { display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0; padding: 14px; background: #16213e; border-radius: 8px; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 13px; }
.legend-swatch { width: 18px; height: 18px; border-radius: 3px; border: 1px solid #444; }
.summary { background: #16213e; padding: 16px; border-radius: 8px; margin-bottom: 20px; }
.summary table { width: 100%%; border-collapse: collapse; }
.summary td, .summary th { padding: 6px 12px; text-align: left; border-bottom: 1px solid #333; }
.summary th { color: #e94560; }
.bar { background: #0f3460; height: 14px; border-radius: 3px; }
</style>
</head>
<body>
<h1>LULC Tile Viewer — %d classes</h1>
""" % (num_classes, num_classes))

    # Legend
    html_parts.append('<div class="legend">')
    for cls_id in range(1, num_classes + 1):
        name = class_names.get(cls_id, "?")
        r, g, b = LABEL_COLORS.get(cls_id, (128, 128, 128))
        html_parts.append(
            '<div class="legend-item">'
            '<div class="legend-swatch" style="background:rgb(%d,%d,%d)"></div>'
            '%d: %s</div>' % (r, g, b, cls_id, name)
        )
    html_parts.append('</div>')

    # Class distribution summary
    html_parts.append('<div class="summary"><h3>Class Distribution</h3><table>')
    html_parts.append('<tr><th>Class</th><th>Tiles</th><th>Distribution</th></tr>')

    # Quick scan for distribution
    tiles = sorted(tiles_dir.glob("tile_*.npz"))
    class_px = defaultdict(int)
    for t in tiles:
        d = np.load(t)
        uniq, counts = np.unique(d["label"], return_counts=True)
        for u, c in zip(uniq, counts):
            class_px[int(u)] += int(c)
    total_px = sum(class_px.values())

    for cls_id in range(1, num_classes + 1):
        name = class_names.get(cls_id, "?")
        px = class_px.get(cls_id, 0)
        pct = 100.0 * px / total_px if total_px > 0 else 0
        n_tiles = len(reps.get(cls_id, []))
        bar_width = max(1, int(pct * 5))
        r, g, b = LABEL_COLORS.get(cls_id, (128, 128, 128))
        html_parts.append(
            '<tr><td>%d. %s</td><td>%d reps</td>'
            '<td><div class="bar" style="width:%dpx;background:rgb(%d,%d,%d)"></div> %.1f%%</td></tr>'
            % (cls_id, name, n_tiles, bar_width, r, g, b, pct)
        )
    html_parts.append('</table></div>')

    # Per-class tile visualizations
    for cls_id in range(1, num_classes + 1):
        name = class_names.get(cls_id, "?")
        tiles_for_class = reps.get(cls_id, [])

        html_parts.append('<div class="class-section">')
        html_parts.append('<h2>%d. %s (%d tiles)</h2>' % (cls_id, name, len(tiles_for_class)))

        if not tiles_for_class:
            html_parts.append('<p style="color:#888">No representative tiles found.</p>')
            html_parts.append('</div>')
            continue

        for frac, tile_path in tiles_for_class:
            d = np.load(tile_path)
            image = d["image"]
            label = d["label"]
            temporal_mask = d.get("temporal_mask", np.ones(4))
            dates = d.get("dates", np.array(["?", "?", "?", "?"]))
            num_frames = int(d.get("num_frames", 4))
            n_bands = int(d.get("num_bands", 6))

            # Class breakdown for this tile
            uniq, counts = np.unique(label, return_counts=True)
            breakdown = ", ".join(
                "%s: %.0f%%" % (class_names.get(int(u), "?"), 100.0 * c / label.size)
                for u, c in zip(uniq, counts) if int(u) > 0
            )

            html_parts.append('<div class="tile-row">')

            # Tile info
            html_parts.append('<div class="tile-info">')
            html_parts.append('<div class="name">%s</div>' % tile_path.stem)
            html_parts.append('<div class="detail">%.0f%% %s</div>' % (frac * 100, name))
            html_parts.append('<div class="detail">%s</div>' % breakdown)
            html_parts.append('</div>')

            # 4 seasonal frames
            for fi in range(num_frames):
                if temporal_mask[fi] > 0:
                    rgb = render_rgb(image, fi, n_bands)
                    b64 = array_to_png_b64(rgb)
                    date_str = str(dates[fi]) if fi < len(dates) else "?"
                else:
                    # Missing frame — dark placeholder
                    rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
                    b64 = array_to_png_b64(rgb)
                    date_str = "missing"

                html_parts.append('<div>')
                html_parts.append('<img src="data:image/png;base64,%s" title="Frame %d: %s">' % (b64, fi, date_str))
                html_parts.append('<div class="frame-label">F%d: %s</div>' % (fi, date_str))
                html_parts.append('</div>')

            # NMD label map
            label_rgb = render_labels(label)
            b64_label = array_to_png_b64(label_rgb)
            html_parts.append('<div>')
            html_parts.append('<img src="data:image/png;base64,%s" title="NMD Labels">' % b64_label)
            html_parts.append('<div class="frame-label">NMD Labels</div>')
            html_parts.append('</div>')

            html_parts.append('</div>')  # tile-row

        html_parts.append('</div>')  # class-section

    html_parts.append('</body></html>')

    output_path.write_text("\n".join(html_parts))
    print("Generated: %s (%d tiles visualized)" % (output_path, sum(len(v) for v in reps.values())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--output", default="tile_viewer.html")
    parser.add_argument("--per-class", type=int, default=3)
    args = parser.parse_args()

    tiles_dir = Path(args.data_dir) / "tiles"
    output_path = Path(args.output)

    generate_html(tiles_dir, args.num_classes, output_path, args.per_class)


if __name__ == "__main__":
    main()
