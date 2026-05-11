"""
demos/era5_metafilter/render_frames.py

Renders per-scene "frame cards" — RGB thumbnail + COT heatmap overlay with
pixel-wise mean COT printed on top — for every cached tile across all three
sets. Output is consumed by the GitHub Pages showcase (docs/index.html).

For each scene we produce:
    docs/showcase/era5_metafilter/frames/{set}/{date}.jpg
    (single composite image: RGB | COT-heatmap, with mean COT in caption)

Plus a manifest:
    docs/showcase/era5_metafilter/frames/manifest.json
    — set → list of {date, mean_cot, thick_frac, rgb_path, cot_path, frame_path}
      so the HTML can render galleries without re-reading npz files.

Run:
    python demos/era5_metafilter/render_frames.py
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
from imint.analyzers.cot import _load_ensemble, cot_inference, DEFAULT_MODEL_PATHS  # noqa: E402

HERE = Path(__file__).parent
CACHE_ROOT = HERE / "cache_11band"
DOCS_OUT = REPO_ROOT / "docs" / "showcase" / "era5_metafilter" / "frames"

# Reflectance stretch for RGB (BOA values typically 0-0.3 for land)
RGB_LO, RGB_HI = 0.0, 0.30

# COT heatmap — DES brand palette
# (#1a4338 dark green / #cff8e4 mint / #fdd5c2 peach / #ff826c red / white)
COT_VMAX = 0.5
COT_CMAP = LinearSegmentedColormap.from_list(
    "cot", [
        (0.00, "#1a4338"),  # very clear (DES dark green)
        (0.05, "#cff8e4"),  # clear (DES mint)
        (0.10, "#fdd5c2"),  # thin haze (DES peach)
        (0.30, "#ff826c"),  # cloud edge (DES red)
        (1.00, "#ffffff"),  # thick cloud (white)
    ],
)


def to_rgb(arr: np.ndarray) -> np.ndarray:
    """Build an 8-bit RGB from S2L2A_SPECTRAL_BANDS-ordered array.

    arr: (11, H, W). Bands are S2L2A_SPECTRAL_BANDS = [B02, B03, B04, ...].
    Map B04→R, B03→G, B02→B (Prithvi/standard true color).
    """
    band = {b: arr[i] for i, b in enumerate(S2L2A_SPECTRAL_BANDS)}
    rgb = np.stack([band["B04"], band["B03"], band["B02"]], axis=-1)
    rgb = np.clip((rgb - RGB_LO) / (RGB_HI - RGB_LO), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def render_frame(
    arr: np.ndarray,
    cot: np.ndarray,
    date: str,
    set_name: str,
    out_path: Path,
) -> dict:
    rgb = to_rgb(arr)
    valid = np.isfinite(cot)
    mean_cot = float(cot[valid].mean()) if valid.any() else float("nan")
    thick_frac = float((cot[valid] >= 0.025).mean()) if valid.any() else float("nan")
    median_cot = float(np.median(cot[valid])) if valid.any() else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0), facecolor="#0f0f0f")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05, wspace=0.04)

    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("RGB (B04/B03/B02)", color="white", fontsize=10, pad=6)
    axes[0].axis("off")

    im = axes[1].imshow(cot, cmap=COT_CMAP, vmin=0, vmax=COT_VMAX, interpolation="nearest")
    axes[1].set_title(f"COT — mean = {mean_cot:.4f}",
                      color="white", fontsize=10, pad=6,
                      fontweight="bold")
    axes[1].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.02)
    cbar.set_label("COT", color="white", fontsize=8)
    cbar.ax.tick_params(colors="white", labelsize=7)
    cbar.outline.set_edgecolor("#444")

    set_color = {"A_baseline": "#7f8c8d"}.get(set_name, "#888")
    fig.suptitle(
        f"{set_name}  ·  {date}  ·  thick-cloud-pixels {thick_frac*100:.0f}%",
        color=set_color, fontsize=11, fontweight="bold", y=0.97,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor="#0f0f0f", bbox_inches="tight")
    plt.close(fig)

    return {
        "date": date,
        "mean_cot": round(mean_cot, 5),
        "median_cot": round(median_cot, 5),
        "thick_cloud_frac": round(thick_frac, 4),
        "frame_path": str(out_path.relative_to(REPO_ROOT / "docs")),
    }


STRATEGIES_METRICS = HERE / "strategies_metrics.json"


def render_baseline_pool(models) -> list[dict]:
    """Render every cached scene once into frames/A_baseline/. Returns
    list of manifest records sorted by mean COT ascending."""
    cache_dir = CACHE_ROOT / "A_baseline"
    npz_files = sorted(cache_dir.glob("*.npz"))
    print(f"\n--- A_baseline ({len(npz_files)} scener) ---")
    records: list[dict] = []
    for i, npz in enumerate(npz_files, 1):
        arr = np.load(npz)["arr"]
        bands = {b: arr[j] for j, b in enumerate(S2L2A_SPECTRAL_BANDS)}
        cot = cot_inference(bands, models)
        out_path = DOCS_OUT / "A_baseline" / f"{npz.stem}.jpg"
        rec = render_frame(arr, cot, npz.stem, "A_baseline", out_path)
        records.append(rec)
        print(f"  [{i:>3}/{len(npz_files)}] {npz.stem}  mean_cot={rec['mean_cot']:.4f}")
    records.sort(key=lambda r: r["mean_cot"])
    return records


def m_sets_from_metrics(baseline_records: list[dict]) -> dict[str, list[dict]]:
    """Build M0–M5 manifest entries by selecting the subset of baseline
    records whose date is in each strategy's date list. Frames are reused
    from A_baseline/ — no re-render needed."""
    if not STRATEGIES_METRICS.exists():
        print(f"\n  (skipping M-sets: {STRATEGIES_METRICS.name} not found — "
              f"run benchmark_strategies.py first)")
        return {}
    with open(STRATEGIES_METRICS) as f:
        metrics = json.load(f)
    by_date = {r["date"]: r for r in baseline_records}
    out: dict[str, list[dict]] = {}
    for key, strat in metrics["strategies"].items():
        picks = [by_date[d] for d in strat["dates"] if d in by_date]
        picks.sort(key=lambda r: r["mean_cot"])
        out[key] = picks
        missing = set(strat["dates"]) - set(by_date)
        tag = f" (missing {len(missing)})" if missing else ""
        print(f"  {key:<22} n={len(picks)}{tag}")
    return out


def main() -> int:
    if not DEFAULT_MODEL_PATHS:
        sys.exit("No COT ensemble weights in imint/fm/cot_models/")
    print(f"Loading {len(DEFAULT_MODEL_PATHS)} MLP5 ensemble members…")
    models = _load_ensemble(DEFAULT_MODEL_PATHS, device="cpu")

    baseline = render_baseline_pool(models)
    print("\n--- M-strategier (återanvänder A_baseline-jpgs) ---")
    m_sets = m_sets_from_metrics(baseline)

    manifest = {"A_baseline": baseline, **m_sets}
    manifest_path = DOCS_OUT / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({
            "rgb_stretch": {"lo": RGB_LO, "hi": RGB_HI},
            "cot_vmax": COT_VMAX,
            "thick_cloud_threshold": 0.025,
            "thin_cloud_threshold": 0.015,
            "sets": manifest,
        }, f, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in manifest.values())
    print(f"\nKlart. {total} entries i {manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
