"""
demos/era5_metafilter/render_set_means.py

Computes per-pixel mean RGB and per-pixel mean Cloud Optical Thickness for
each of the three sets (A_baseline, B_metafilter, C_fetch_s2) and renders
them as panel-style images for the GitHub Pages showcase.

Output: one composite PNG per set (RGB-mean | COT-mean) plus a small
metadata JSON, written to docs/showcase/era5_metafilter/.

Why mean-of-means: the question the showcase asks is "if you train on this
set, what's the typical pixel quality?" — that's a per-pixel mean across
the whole set, not a per-scene aggregate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from imint.fetch import S2L2A_SPECTRAL_BANDS  # noqa: E402
from imint.analyzers.cot import (  # noqa: E402
    DEFAULT_MODEL_PATHS, _load_ensemble, cot_inference,
)

HERE = Path(__file__).parent
CACHE_ROOT = HERE / "cache_11band"
DOCS_OUT = REPO_ROOT / "docs" / "showcase" / "era5_metafilter"

RGB_LO, RGB_HI = 0.0, 0.30
COT_VMAX = 0.5
COT_CMAP = LinearSegmentedColormap.from_list(
    "cot", [
        (0.00, "#0d4d2a"),
        (0.05, "#27ae60"),
        (0.10, "#f1c40f"),
        (0.30, "#e67e22"),
        (1.00, "#ecf0f1"),
    ],
)
SET_COLORS = {
    "A_baseline":   "#7f8c8d",
    "B_metafilter": "#27ae60",
    "C_fetch_s2":   "#2980b9",
}
SET_TITLES = {
    "A_baseline":   "A — Baseline (alla scener, ingen filter)",
    "B_metafilter": "B — Atmosfär-filter (ERA5-passande dagar)",
    "C_fetch_s2":   "C — STAC-cc (granul cloud_cover ≤ 30 %)",
}


def stream_means(set_name: str, models) -> dict:
    """Iterate npz files in a set, accumulating per-pixel sums.

    Returns a dict with mean_rgb (H, W, 3) uint8, mean_cot (H, W) float32,
    grand_mean_cot scalar, and n.
    """
    cache_dir = CACHE_ROOT / set_name
    npz_files = sorted(cache_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No npz files in {cache_dir}")

    rgb_sum: np.ndarray | None = None
    cot_sum: np.ndarray | None = None
    n = 0
    print(f"\n--- {set_name} ({len(npz_files)} scener) ---")

    for i, npz in enumerate(npz_files, 1):
        data = np.load(npz)
        arr = data["arr"]  # (11, H, W)
        bands = {b: arr[j] for j, b in enumerate(S2L2A_SPECTRAL_BANDS)}

        rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1)
        cot = cot_inference(bands, models)  # (H, W) float32

        if rgb_sum is None:
            rgb_sum = np.zeros_like(rgb, dtype=np.float64)
            cot_sum = np.zeros_like(cot, dtype=np.float64)
        rgb_sum += rgb
        cot_sum += cot
        n += 1
        print(f"  [{i:>3}/{len(npz_files)}] {npz.stem}", end="\r")

    rgb_mean = rgb_sum / n
    cot_mean = cot_sum / n

    rgb_8bit = np.clip(
        (rgb_mean - RGB_LO) / (RGB_HI - RGB_LO), 0.0, 1.0,
    )
    rgb_8bit = (rgb_8bit * 255).astype(np.uint8)

    grand_mean = float(np.nanmean(cot_mean))
    print(f"  done. grand-mean COT = {grand_mean:.4f}")

    return {
        "name": set_name,
        "n": n,
        "rgb_mean": rgb_8bit,
        "cot_mean": cot_mean.astype(np.float32),
        "grand_mean_cot": grand_mean,
        "thick_cloud_frac": float((cot_mean >= 0.025).mean()),
        "thin_cloud_frac": float(((cot_mean >= 0.015) & (cot_mean < 0.025)).mean()),
        "clear_frac": float((cot_mean < 0.015).mean()),
    }


def render_panel(stats: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6), facecolor="#0f0f0f")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.04)

    axes[0].imshow(stats["rgb_mean"], interpolation="nearest")
    axes[0].set_title("Mean RGB (B04/B03/B02)", color="white",
                      fontsize=11, pad=8)
    axes[0].axis("off")

    im = axes[1].imshow(stats["cot_mean"], cmap=COT_CMAP,
                         vmin=0, vmax=COT_VMAX, interpolation="nearest")
    axes[1].set_title(
        f"Mean COT — AOI-grand-mean = {stats['grand_mean_cot']:.4f}",
        color="white", fontsize=11, pad=8, fontweight="bold",
    )
    axes[1].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.02)
    cbar.set_label("Mean COT", color="white", fontsize=9)
    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.outline.set_edgecolor("#444")

    set_color = SET_COLORS[stats["name"]]
    fig.suptitle(
        f"{SET_TITLES[stats['name']]}  ·  n = {stats['n']}  ·  "
        f"klar-pixlar {stats['clear_frac']*100:.0f} %  ·  "
        f"tjocka-moln-pixlar {stats['thick_cloud_frac']*100:.0f} %",
        color=set_color, fontsize=12, fontweight="bold", y=0.96,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor="#0f0f0f", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if not DEFAULT_MODEL_PATHS:
        sys.exit("No COT ensemble weights in imint/fm/cot_models/")
    print(f"Loading {len(DEFAULT_MODEL_PATHS)} MLP5 ensemble members…")
    models = _load_ensemble(DEFAULT_MODEL_PATHS, device="cpu")

    set_names = ["A_baseline", "B_metafilter", "C_fetch_s2"]
    summary: dict[str, dict] = {}

    for name in set_names:
        s = stream_means(name, models)
        out_path = DOCS_OUT / f"set_mean_{name}.png"
        render_panel(s, out_path)
        summary[name] = {
            "n": s["n"],
            "grand_mean_cot": round(s["grand_mean_cot"], 5),
            "clear_frac": round(s["clear_frac"], 4),
            "thin_cloud_frac": round(s["thin_cloud_frac"], 4),
            "thick_cloud_frac": round(s["thick_cloud_frac"], 4),
            "panel_path": str(out_path.relative_to(REPO_ROOT / "docs")),
        }
        print(f"  → {out_path.relative_to(REPO_ROOT)}")

    with open(DOCS_OUT / "set_means.json", "w") as f:
        json.dump({
            "rgb_stretch": {"lo": RGB_LO, "hi": RGB_HI},
            "cot_vmax": COT_VMAX,
            "thick_cloud_threshold": 0.025,
            "thin_cloud_threshold": 0.015,
            "sets": summary,
        }, f, indent=2, ensure_ascii=False)
    print("\nKlart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
