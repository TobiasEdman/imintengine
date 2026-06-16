#!/usr/bin/env python3
"""Self-contained tile inspection dashboard.

For each tile in ``--data-dir`` renders, as one static HTML page with
base64-embedded PNGs (no server runtime, no external assets):

  * the N temporal frames as true-colour RGB (B04/B03/B02, 2-98% stretch)
  * the unified 23-class label overlay, using the schema palette
    (``UNIFIED_COLORS``) and the same palette alpha-blended on a frame
  * one panel per present auxiliary channel (dem, skg tree-data, markfukt,
    vpp phenology, harvest mask, the 2016 background frame, Sentinel-1)

Absent channels are skipped, so the page doubles as a visual "what aux is
actually injected" report. Band order and stretch match the repo
convention (``scripts/generate_tile_previews.py`` / ``inspect_tile.py``).

Usage:
    python scripts/render_tile_inspection_dashboard.py \
        --data-dir /data/_national_dryrun --out /tmp/dashboard.html
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imint.training.unified_schema import (  # noqa: E402
    NUM_UNIFIED_CLASSES,
    UNIFIED_CLASS_NAMES,
    UNIFIED_COLORS,
)

# Canonical RGB helpers live in scripts/tile_rgb.py (shared with the campaign
# dashboard). Alias to the legacy private names so the call sites stay unchanged.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tile_rgb import (  # noqa: E402
    N_BANDS,
    png_b64 as _png_b64,
    stretch_rgb as _stretch_rgb,
    frame_rgb as _frame_rgb,
)

_COLOR_LUT = np.zeros((NUM_UNIFIED_CLASSES, 3), dtype=np.uint8)
for _i in range(NUM_UNIFIED_CLASSES):
    _COLOR_LUT[_i] = UNIFIED_COLORS[_i]

# Continuous aux channels → a perceptually-ordered colormap each.
AUX_PANELS: list[tuple[str, str]] = [
    ("dem", "terrain"),
    ("height", "viridis"),
    ("volume", "viridis"),
    ("basal_area", "viridis"),
    ("diameter", "viridis"),
    ("markfukt", "Blues"),
    ("vpp_sosd", "RdYlGn_r"),
    ("vpp_eosd", "RdYlGn"),
    ("vpp_length", "magma"),
    ("vpp_maxv", "YlGn"),
    ("vpp_minv", "YlGn"),
]


def _label_rgb(label: np.ndarray) -> np.ndarray:
    return _COLOR_LUT[np.clip(label.astype(np.int64), 0, NUM_UNIFIED_CLASSES - 1)]


def _overlay_rgb(frame: np.ndarray, label: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    lab = _label_rgb(label).astype(np.float32)
    out = frame.astype(np.float32)
    mask = (label > 0)[..., None]
    blended = (1.0 - alpha) * out + alpha * lab
    return np.where(mask, blended, out).astype(np.uint8)


def _aux_rgb(arr: np.ndarray, cmap_name: str) -> tuple[np.ndarray, str]:
    """Colormap a single-channel aux array (NaN-aware) → (H,W,3) uint8, caption."""
    a = np.asarray(arr, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros((*a.shape, 3), np.uint8), "all-nan"
    lo, hi = np.percentile(finite, (2, 98))
    span = max(float(hi - lo), 1e-6)
    normed = np.clip((a - lo) / span, 0.0, 1.0)
    normed = np.nan_to_num(normed, nan=0.0)
    rgba = colormaps[cmap_name](normed)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    rgb[~np.isfinite(a)] = (30, 30, 30)  # nodata = dark grey
    return rgb, f"min={float(finite.min()):.2f} max={float(finite.max()):.2f}"


def _harvest_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.full((h, w, 3), 20, np.uint8)
    out[mask > 0] = (255, 40, 40)
    return out


def _panel(title: str, rgb_u8: np.ndarray, caption: str = "") -> str:
    img = _png_b64(rgb_u8)
    cap = f"<div class='cap'>{caption}</div>" if caption else ""
    return (
        f"<figure class='panel'><figcaption>{title}</figcaption>"
        f"<img src='data:image/png;base64,{img}'/>{cap}</figure>"
    )


def _legend_html() -> str:
    chips = []
    for i in range(NUM_UNIFIED_CLASSES):
        r, g, b = UNIFIED_COLORS[i]
        chips.append(
            f"<span class='chip'><i style='background:rgb({r},{g},{b})'></i>"
            f"{i} {UNIFIED_CLASS_NAMES[i]}</span>"
        )
    return "<div class='legend'>" + "".join(chips) + "</div>"


def _render_tile(fp: str) -> str:
    d = dict(np.load(fp, allow_pickle=True))
    name = os.path.basename(fp)
    sp = d["spectral"]
    n_frames = int(d.get("num_frames", sp.shape[0] // N_BANDS))
    dates = [str(x) for x in d.get("dates", [])]

    panels: list[str] = []
    for fi in range(n_frames):
        date = dates[fi] if fi < len(dates) else "?"
        panels.append(_panel(f"frame {fi} — {date}", _frame_rgb(sp, fi)))

    have = set(d.keys())

    def _is_hw(arr) -> bool:
        a = np.asarray(arr)
        return a.ndim == 2 and a.shape == sp.shape[1:]

    if "label" in have and _is_hw(d["label"]):
        label = np.asarray(d["label"])
        panels.append(_panel("label (23-class)", _label_rgb(label)))
        best = min(n_frames - 1, max(0, n_frames - 2))
        panels.append(_panel("label ∘ frame", _overlay_rgb(_frame_rgb(sp, best), label)))
    if "harvest_mask" in have and _is_hw(d["harvest_mask"]):
        panels.append(_panel("harvest_mask", _harvest_rgb(np.asarray(d["harvest_mask"]))))

    for key, cmap_name in AUX_PANELS:
        if key in have and _is_hw(d[key]):
            rgb, cap = _aux_rgb(d[key], cmap_name)
            panels.append(_panel(key, rgb, cap))

    f16 = np.asarray(d["frame_2016"]) if "frame_2016" in have else None
    if f16 is not None and f16.ndim == 3 and f16.shape[0] >= 3 and int(d.get("has_frame_2016", 0)) == 1:
        rgb = _stretch_rgb(f16[2], f16[1], f16[0])
        panels.append(_panel(f"frame_2016 — {str(d.get('frame_2016_date',''))[2:12]}", rgb))
    s1 = np.asarray(d["s1_vv_vh"], dtype=np.float32) if "s1_vv_vh" in have else None
    if s1 is not None and s1.ndim == 3 and int(d.get("has_s1", 0)) == 1:
        rgb, cap = _aux_rgb(s1[0], "gray")  # first VV frame
        panels.append(_panel("s1 VV (frame 0)", rgb, cap))

    core = ["label", "harvest_mask", "dem", "height", "volume", "basal_area",
            "diameter", "markfukt", "vpp_sosd", "vpp_eosd"]
    present = [k for k in core if k in have]
    extras = [k for k in ("s1_vv_vh", "frame_2016", "tessera") if k in have]
    missing = [k for k in core if k not in have]
    summary = (
        f"<div class='tmeta'>{sp.shape[1]}×{sp.shape[2]} px · {n_frames} frames · "
        f"core aux {len(present)}/{len(core)} · extras {extras or '—'}"
        + (f" · <b style='color:#c0392b'>missing {missing}</b>" if missing else "")
        + "</div>"
    )
    return (
        f"<section class='tile'><h2>{name}</h2>{summary}"
        f"<div class='grid'>{''.join(panels)}</div></section>"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="Glob <dir>/*.npz")
    ap.add_argument("--out", required=True, help="Output HTML path")
    ap.add_argument("--tiles", nargs="*", help="Optional explicit tile basenames")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.tiles:
        wanted = set(args.tiles)
        files = [f for f in files if os.path.basename(f) in wanted]
    if not files:
        print(f"no tiles in {args.data_dir}", file=sys.stderr)
        return 1

    sections = "".join(_render_tile(f) for f in files)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Tile inspection — {os.path.basename(args.data_dir)}</title>
<style>
  body {{ font-family: ui-monospace, monospace; margin: 18px; background:#fafafa; color:#171717; }}
  h1 {{ font-size: 18px; }} h2 {{ font-size: 15px; margin: 22px 0 4px; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 18px; }}
  .chip {{ font-size:11px; display:flex; align-items:center; gap:4px; }}
  .chip i {{ width:12px; height:12px; display:inline-block; border:1px solid #999; }}
  .tmeta {{ font-size:12px; color:#555; margin-bottom:6px; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(190px,1fr)); gap:10px; }}
  .panel {{ margin:0; background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:6px; }}
  .panel figcaption {{ font-size:11px; color:#374151; margin-bottom:4px; }}
  .panel img {{ width:100%; image-rendering:pixelated; border-radius:4px; }}
  .cap {{ font-size:10px; color:#6b7280; margin-top:2px; }}
  .tile {{ border-top:2px solid #1A4338; padding-top:6px; margin-top:18px; }}
</style></head><body>
<h1>Tile inspection — {os.path.basename(args.data_dir)} ({len(files)} tiles)</h1>
{_legend_html()}
{sections}
</body></html>"""

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out}  ({len(files)} tiles, {len(html)//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
