#!/usr/bin/env python3
"""Render a data-source showcase for one unified-dataset tile.

Produces PNGs (true-colour RGB, NIR-CIR pseudocolour, the 23-class label,
NMD base, and the auxiliary sources) plus a self-contained showcase.html.
RGB / NIR-CIR follow the repo's standard viz parameters (B04/B03/B02 and
B8A/B03/B04, 2-98 percentile stretch — see imint.utils.bands_to_rgb).

Usage:
    python scripts/render_dataset_showcase.py --tile <tile.npz> --out <dir> [--frame 2]
"""
from __future__ import annotations

import argparse
import base64
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAND_ORDER = ["B02", "B03", "B04", "B8A", "B11", "B12"]

# Frozen copy of imint/training/unified_schema.py UNIFIED_COLORS /
# UNIFIED_CLASSES so this showcase script runs standalone (no imint
# import chain) inside a minimal numpy+matplotlib job. Keep in sync with
# the schema module — it is the source of truth.
UNIFIED_CLASSES = {
    0: "bakgrund", 1: "tallskog", 2: "granskog", 3: "lövskog", 4: "blandskog",
    5: "sumpskog", 6: "tillfälligt ej skog", 7: "våtmark", 8: "öppen mark",
    9: "bebyggelse", 10: "vatten", 11: "vete", 12: "korn", 13: "havre",
    14: "oljeväxter", 15: "slåttervall", 16: "bete", 17: "potatis",
    18: "sockerbetor", 19: "trindsäd", 20: "råg", 21: "majs", 22: "hygge",
}
UNIFIED_COLORS = {
    0: (0, 0, 0), 1: (0, 100, 0), 2: (34, 139, 34), 3: (50, 205, 50),
    4: (60, 179, 113), 5: (46, 79, 46), 6: (160, 200, 120), 7: (139, 90, 43),
    8: (210, 180, 140), 9: (255, 0, 0), 10: (0, 0, 255), 11: (230, 180, 34),
    12: (212, 130, 23), 13: (255, 255, 100), 14: (45, 180, 90),
    15: (100, 200, 100), 16: (80, 160, 60), 17: (180, 80, 40),
    18: (200, 100, 200), 19: (140, 180, 50), 20: (190, 150, 80),
    21: (220, 200, 0), 22: (0, 206, 209),
}


def _pct_stretch(stack: np.ndarray) -> np.ndarray:
    """2-98 percentile stretch to [0,1] — the repo's display standard."""
    p2, p98 = np.percentile(stack, [2, 98])
    return np.clip((stack - p2) / (p98 - p2 + 1e-6), 0, 1)


def _frame_bands(spectral: np.ndarray, frame: int) -> dict:
    base = frame * len(BAND_ORDER)
    return {name: spectral[base + i] for i, name in enumerate(BAND_ORDER)}


def rgb_image(spectral: np.ndarray, frame: int) -> np.ndarray:
    """True-colour RGB (B04,B03,B02) with standard percentile stretch.

    Mirrors imint.utils.bands_to_rgb: R=B04, G=B03, B=B02, 2-98% stretch.
    """
    b = _frame_bands(spectral, frame)
    stack = np.stack([b["B04"], b["B03"], b["B02"]], axis=-1).astype(np.float32)
    return _pct_stretch(stack)


def nir_cir_image(spectral: np.ndarray, frame: int) -> np.ndarray:
    """False-colour NIR-CIR: R=B8A, G=B03, B=B04 (vegetation = red)."""
    b = _frame_bands(spectral, frame)
    stack = np.stack([b["B8A"], b["B03"], b["B04"]], axis=-1).astype(np.float32)
    return _pct_stretch(stack)


def colorize_label(label: np.ndarray) -> np.ndarray:
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cls, rgb in UNIFIED_COLORS.items():
        out[label == cls] = rgb
    return out


def _save_rgb(arr: np.ndarray, path: pathlib.Path, upscale: int = 384) -> None:
    fig = plt.figure(figsize=(upscale / 100, upscale / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(arr, interpolation="nearest")
    fig.savefig(path, dpi=100); plt.close(fig)


def _save_heat(arr: np.ndarray, path: pathlib.Path, cmap: str,
               vmin=None, vmax=None, robust=True, upscale: int = 384) -> None:
    a = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if robust and vmin is None and finite.any():
        vmin, vmax = np.percentile(a[finite], [2, 98])
    fig = plt.figure(figsize=(upscale / 100, upscale / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.savefig(path, dpi=100); plt.close(fig)


def _save_frames_strip(spectral: np.ndarray, n_frames: int,
                       dates, path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, n_frames, figsize=(2.4 * n_frames, 2.7), dpi=100)
    if n_frames == 1:
        axes = [axes]
    titles = ["autumn (yr-1)", "spring", "summer", "late summer"]
    for f in range(n_frames):
        axes[f].imshow(rgb_image(spectral, f), interpolation="nearest")
        axes[f].axis("off")
        cap = titles[f] if f < len(titles) else f"frame {f}"
        if dates is not None and f < len(dates):
            cap += f"\n{dates[f]}"
        axes[f].set_title(cap, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def _save_legend(label: np.ndarray, path: pathlib.Path) -> None:
    present = sorted(int(c) for c in np.unique(label))
    fig, ax = plt.subplots(figsize=(3.0, 0.32 * max(len(present), 1) + 0.2), dpi=100)
    ax.axis("off")
    for i, cls in enumerate(present):
        rgb = np.array(UNIFIED_COLORS.get(cls, (0, 0, 0))) / 255.0
        ax.add_patch(plt.Rectangle((0, i), 0.6, 0.8, color=rgb))
        ax.text(0.8, i + 0.4, f"{cls}  {UNIFIED_CLASSES.get(cls, '?')}",
                va="center", fontsize=8)
    ax.set_xlim(0, 4); ax.set_ylim(0, len(present))
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


# (key, png-name, title, source, how-rendered)
PANELS = [
    ("rgb",       "True-colour RGB",        "Sentinel-2 (B04/B03/B02), 2-98% stretch"),
    ("nir_cir",   "NIR-CIR pseudocolour",   "Sentinel-2 (B8A/B03/B04) — vegetation in red"),
    ("label",     "Unified label (23 cls)", "NMD + LPIS + SKS"),
    ("nmd",       "NMD base land cover",    "Naturvårdsverket NMD"),
    ("dem",       "Elevation (DEM)",        "Copernicus DEM, cmap=terrain"),
    ("height",    "Forest canopy height",   "SLU Skogliga grunddata, cmap=viridis"),
    ("vpp",       "VPP start-of-season",    "Copernicus HR-VPP, cmap=viridis"),
    ("harvest",   "Harvest-readiness prob", "Skogsstyrelsen SKS, cmap=magma"),
    ("frames",    "Multitemporal RGB",      "4 frames: autumn(yr-1) + 3 VPP-guided"),
]


def render(tile_path: str, out_dir: str, frame: int) -> dict:
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    z = np.load(tile_path, allow_pickle=True)
    spectral = z["spectral"] if "spectral" in z.files else z["image"]
    n_frames = int(z["num_frames"]) if "num_frames" in z.files else spectral.shape[0] // 6
    frame = min(frame, n_frames - 1)
    dates = [str(d) for d in z["dates"]] if "dates" in z.files else None
    label = z["label"]

    _save_rgb(rgb_image(spectral, frame), out / "rgb.png")
    _save_rgb(nir_cir_image(spectral, frame), out / "nir_cir.png")
    _save_rgb(colorize_label(label), out / "label.png")
    if "nmd_label" in z.files:
        _save_rgb(colorize_label(z["nmd_label"]), out / "nmd.png")
    if "dem" in z.files:
        _save_heat(z["dem"], out / "dem.png", "terrain")
    if "height" in z.files:
        _save_heat(z["height"], out / "height.png", "viridis")
    if "vpp_sosd" in z.files:
        _save_heat(z["vpp_sosd"], out / "vpp.png", "viridis")
    if "harvest_probability" in z.files:
        # Nodata is stored as a ~3.3e38 sentinel; mask anything outside the
        # valid probability range [0,1] before plotting.
        hp = np.asarray(z["harvest_probability"], dtype=np.float32)
        hp = np.where((hp >= 0) & (hp <= 1), hp, np.nan)
        _save_heat(hp, out / "harvest.png", "magma", vmin=0, vmax=1, robust=False)
    _save_frames_strip(spectral, n_frames, dates, out / "frames.png")
    _save_legend(label, out / "legend.png")

    meta = {
        "tile": pathlib.Path(tile_path).name,
        "frame": frame,
        "dates": dates,
        "easting": int(z["easting"]) if "easting" in z.files else None,
        "northing": int(z["northing"]) if "northing" in z.files else None,
    }
    _write_html(out, meta)
    print(f"rendered showcase for {meta['tile']} -> {out}")
    return meta


def _write_html(out: pathlib.Path, meta: dict) -> None:
    cards = []
    for key, title, source in PANELS:
        png = out / f"{key}.png"
        if not png.exists():
            continue
        cards.append(
            f'<figure class="card"><img src="{key}.png" alt="{title}" loading="lazy">'
            f'<figcaption><b>{title}</b><span>{source}</span></figcaption></figure>'
        )
    legend_html = ""
    if (out / "legend.png").exists():
        legend_html = '<div class="legend"><img src="legend.png" alt="label legend"></div>'
    sub = f"tile {meta['tile']}"
    if meta.get("easting"):
        sub += f" · center {meta['easting']},{meta['northing']} (EPSG:3006)"
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unified v5 256px — data-source showcase</title>
<style>
  body{{margin:0;background:#0b0e17;color:#d8dae5;
    font-family:-apple-system,Segoe UI,Arial,sans-serif}}
  header{{padding:20px 24px;border-bottom:1px solid #1e293b;
    background:linear-gradient(135deg,#111827,#1e293b)}}
  header h1{{margin:0;font-size:17px;color:#f1f5f9}}
  header p{{margin:4px 0 0;font-size:12px;color:#94a3b8}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
    gap:16px;padding:20px 24px}}
  .card{{margin:0;background:#111827;border:1px solid #1e293b;border-radius:8px;
    overflow:hidden}}
  .card img{{display:block;width:100%;height:auto;background:#000}}
  figcaption{{padding:8px 10px;font-size:12px;display:flex;flex-direction:column}}
  figcaption span{{color:#94a3b8;font-size:11px;margin-top:2px}}
  .legend{{padding:0 24px 28px}}
  .legend img{{background:#111827;border:1px solid #1e293b;border-radius:8px;
    padding:8px;max-width:340px}}
</style></head><body>
<header>
  <h1>ImintEngine — Unified v5 (256 px) · data-source showcase</h1>
  <p>{sub}. RGB &amp; NIR-CIR use standard viz parameters (B04/B03/B02 and
     B8A/B03/B04, 2–98% stretch).</p>
</header>
<div class="grid">
{chr(10).join(cards)}
</div>
{legend_html}
</body></html>"""
    (out / "showcase.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=2)
    a = ap.parse_args()
    render(a.tile, a.out, a.frame)
