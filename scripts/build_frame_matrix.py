#!/usr/bin/env python3
"""
Generate a 5-tile x 6-band frame matrix HTML visualization.

For each of 5 sample tiles, renders all 4 temporal frames as rows,
each showing the 6 Sentinel-2 bands (B02, B03, B04, B8A, B11, B12)
plus a true-colour and NIR false-colour composite.

Layout per tile:
  Rows: Frame 0 (autumn yr-1), Frame 1 (spring), Frame 2 (summer), Frame 3 (late summer)
  Cols: B02, B03, B04, B8A, B11, B12, RGB, NIR-CIR

Output: self-contained HTML with base64-encoded images.

Usage:
  python scripts/build_frame_matrix.py --data-dir /data/unified_v2 --out frame_matrix.html
  python scripts/build_frame_matrix.py --data-dir /data/unified_v2 --tiles tile_501280_6621280 tile_601280_7071280
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("pip install Pillow"); sys.exit(1)


BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
FRAME_LABELS = ["Frame 0: Autumn (yr-1)", "Frame 1: Spring", "Frame 2: Summer", "Frame 3: Late summer"]
N_BANDS = 6


def percentile_stretch(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Stretch to 0-255 using percentile clipping."""
    p_lo, p_hi = np.percentile(arr[arr > 0], [lo, hi]) if (arr > 0).any() else (0, 1)
    if p_hi <= p_lo:
        p_hi = p_lo + 1
    stretched = np.clip((arr - p_lo) / (p_hi - p_lo), 0, 1)
    return (stretched * 255).astype(np.uint8)


def band_to_png_b64(band: np.ndarray) -> str:
    """Single-band array -> base64 PNG (grayscale)."""
    img = Image.fromarray(percentile_stretch(band), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def composite_to_png_b64(bands: np.ndarray, indices: list[int]) -> str:
    """3-band composite -> base64 PNG (RGB)."""
    rgb = np.stack([percentile_stretch(bands[i]) for i in indices], axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def select_diverse_tiles(data_dir: Path, n: int = 5) -> list[Path]:
    """Pick n tiles spread across different sources and locations."""
    tiles = sorted(data_dir.glob("*.npz"))
    if len(tiles) <= n:
        return tiles[:n]

    # Sample from different geographic regions (by northing)
    northings = []
    for t in tiles:
        try:
            d = np.load(t, allow_pickle=True)
            northings.append((int(d.get("northing", 0)), t))
        except Exception:
            continue

    northings.sort(key=lambda x: x[0])
    step = max(1, len(northings) // n)
    selected = [northings[i * step][1] for i in range(n)]
    return selected[:n]


def build_tile_data(npz_path: Path) -> dict | None:
    """Extract frame matrix data from a single .npz tile."""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"  Skip {npz_path.name}: {e}")
        return None

    spectral = data.get("spectral", data.get("image"))
    if spectral is None:
        return None
    spectral = spectral.astype(np.float32)

    n_frames = spectral.shape[0] // N_BANDS
    if n_frames < 2:
        return None

    temporal_mask = data.get("temporal_mask", np.ones(n_frames, dtype=np.uint8))
    doy = data.get("doy", np.zeros(n_frames, dtype=np.int32))
    dates = list(data.get("dates", []))
    year = int(data.get("year", data.get("lpis_year", 0)))
    source = str(data.get("source", "unknown"))

    frames = []
    for t in range(min(n_frames, 4)):
        start = t * N_BANDS
        bands = spectral[start:start + N_BANDS]  # (6, H, W)

        band_images = [band_to_png_b64(bands[b]) for b in range(N_BANDS)]
        rgb = composite_to_png_b64(bands, [2, 1, 0])       # B04, B03, B02
        nir_cir = composite_to_png_b64(bands, [3, 2, 1])   # B8A, B04, B03

        valid = bool(temporal_mask[t]) if t < len(temporal_mask) else True
        frame_doy = int(doy[t]) if t < len(doy) else 0
        frame_date = str(dates[t]) if t < len(dates) else ""

        frames.append({
            "bands": band_images,
            "rgb": rgb,
            "nir_cir": nir_cir,
            "valid": valid,
            "doy": frame_doy,
            "date": frame_date,
        })

    return {
        "name": npz_path.stem,
        "year": year,
        "source": source,
        "n_frames": n_frames,
        "frames": frames,
    }


def generate_html(tiles_data: list[dict]) -> str:
    """Build self-contained HTML with the frame matrix."""
    col_headers = BANDS + ["True Colour", "NIR CIR"]
    n_cols = len(col_headers)

    # Build image grid HTML
    tile_blocks = []
    for td in tiles_data:
        rows_html = []
        for fi, frame in enumerate(td["frames"]):
            label = FRAME_LABELS[fi] if fi < len(FRAME_LABELS) else f"Frame {fi}"
            status = "" if frame["valid"] else " (padded)"
            date_info = f" | {frame['date']}" if frame["date"] else ""
            doy_info = f" | DOY {frame['doy']}" if frame["doy"] else ""

            cells = []
            for b64 in frame["bands"]:
                cells.append(f'<td><img src="data:image/png;base64,{b64}"></td>')
            cells.append(f'<td><img src="data:image/png;base64,{frame["rgb"]}"></td>')
            cells.append(f'<td><img src="data:image/png;base64,{frame["nir_cir"]}"></td>')

            rows_html.append(
                f'<tr><th class="row-label">{label}{status}{doy_info}{date_info}</th>'
                + "".join(cells)
                + "</tr>"
            )

        tile_blocks.append({
            "name": td["name"],
            "year": td["year"],
            "source": td["source"],
            "rows": "\n".join(rows_html),
        })

    tiles_html = ""
    for tb in tile_blocks:
        tiles_html += f"""
        <div class="tile-block">
          <h2>{tb['name']} <span class="meta">year={tb['year']} source={tb['source']}</span></h2>
          <table>
            <thead><tr><th></th>{''.join(f'<th>{h}</th>' for h in col_headers)}</tr></thead>
            <tbody>{tb['rows']}</tbody>
          </table>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Frame Matrix — {len(tiles_data)} tiles x 4 frames x 8 columns</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0b0e17; color: #e0e0e0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 13px; padding: 20px;
  }}
  h1 {{ font-size: 18px; margin-bottom: 16px; color: #ffa726; }}
  h2 {{
    font-size: 14px; margin: 12px 0 6px;
    border-bottom: 1px solid #333; padding-bottom: 4px;
  }}
  h2 .meta {{ color: #888; font-weight: normal; font-size: 12px; }}
  .tile-block {{ margin-bottom: 32px; }}
  table {{
    border-collapse: collapse;
    margin: 0 auto;
  }}
  th, td {{
    padding: 2px; text-align: center; border: 1px solid #222;
  }}
  thead th {{
    background: #1a1d2e; color: #90caf9;
    font-size: 11px; padding: 6px 4px;
    position: sticky; top: 0; z-index: 10;
  }}
  .row-label {{
    text-align: right; padding-right: 8px;
    font-size: 11px; color: #aaa; white-space: nowrap;
    min-width: 220px; background: #0f1220;
  }}
  td img {{
    width: 96px; height: 96px;
    image-rendering: pixelated;
    display: block;
  }}
  td img:hover {{
    transform: scale(2.5);
    position: relative; z-index: 100;
    box-shadow: 0 0 20px rgba(255,167,38,0.5);
    transition: transform 0.15s;
  }}
  /* Size slider */
  .controls {{
    margin: 12px 0; display: flex; align-items: center; gap: 12px;
  }}
  .controls input[type=range] {{ width: 200px; }}
  .controls label {{ color: #888; font-size: 12px; }}
</style>
</head>
<body>
<h1>Multitemporal Frame Matrix</h1>
<div class="controls">
  <label>Cell size: <span id="sz-val">96</span>px</label>
  <input type="range" id="sz" min="48" max="256" value="96" step="8">
</div>
{tiles_html}
<script>
const slider = document.getElementById('sz');
const szVal = document.getElementById('sz-val');
slider.addEventListener('input', () => {{
  const v = slider.value;
  szVal.textContent = v;
  document.querySelectorAll('td img').forEach(img => {{
    img.style.width = v + 'px';
    img.style.height = v + 'px';
  }});
}});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Build frame matrix HTML")
    parser.add_argument("--data-dir", required=True, help="Path to unified_v2/")
    parser.add_argument("--out", default="frame_matrix.html")
    parser.add_argument("--tiles", nargs="*", help="Specific tile stems (without .npz)")
    parser.add_argument("-n", type=int, default=5, help="Number of tiles if auto-selecting")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found"); sys.exit(1)

    if args.tiles:
        paths = [data_dir / f"{t}.npz" for t in args.tiles]
        paths = [p for p in paths if p.exists()]
    else:
        paths = select_diverse_tiles(data_dir, args.n)

    print(f"Processing {len(paths)} tiles...")
    tiles_data = []
    for p in paths:
        print(f"  {p.name}...")
        td = build_tile_data(p)
        if td:
            tiles_data.append(td)

    if not tiles_data:
        print("ERROR: No valid tiles found"); sys.exit(1)

    html = generate_html(tiles_data)
    out = Path(args.out)
    out.write_text(html)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Wrote {out} ({size_mb:.1f} MB, {len(tiles_data)} tiles)")


if __name__ == "__main__":
    main()
