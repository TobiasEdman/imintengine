"""
demos/era5_metafilter/compute_cot.py

Runs the real DES COT (Cloud Optical Thickness) MLP5 ensemble model
(imint/analyzers/cot.py — Pirinen et al. 2024) on every cached 11-band
Sentinel-2 tile and aggregates mean COT per set.

Reads:
    demos/era5_metafilter/cache_11band/{A_baseline,B_metafilter,C_fetch_s2}/*.npz
Writes:
    demos/era5_metafilter/cot_metrics.json
    demos/era5_metafilter/figures/07_mean_cot_per_set.png
    demos/era5_metafilter/figures/08_cot_per_scene.png
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from imint.fetch import S2L2A_SPECTRAL_BANDS  # noqa: E402
from imint.analyzers.cot import COT_BAND_ORDER, MLP5, _load_ensemble, cot_inference  # noqa: E402

HERE = Path(__file__).parent
CACHE_ROOT = HERE / "cache_11band"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

SET_COLORS = {
    "A_baseline": "#7f8c8d",
    "B_metafilter": "#27ae60",
    "C_fetch_s2": "#2980b9",
}


def array_to_band_dict(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Map (11, H, W) array to {"B02": ..., ..., "B12": ...}.

    The fetch returns bands in S2L2A_SPECTRAL_BANDS order; COT_BAND_ORDER is
    a permutation of the same set, so we map by band name explicitly rather
    than relying on positional alignment.
    """
    if arr.shape[0] != len(S2L2A_SPECTRAL_BANDS):
        raise ValueError(
            f"Expected {len(S2L2A_SPECTRAL_BANDS)} bands, got {arr.shape[0]} "
            f"(re-fetch with the 11-band default)."
        )
    return {band: arr[i] for i, band in enumerate(S2L2A_SPECTRAL_BANDS)}


def process_set(name: str, models) -> dict:
    cache_dir = CACHE_ROOT / name
    npz_files = sorted(cache_dir.glob("*.npz"))
    per_scene: list[dict] = []
    for i, npz in enumerate(npz_files, 1):
        data = np.load(npz)
        arr = data["arr"]
        bands = array_to_band_dict(arr)
        cot_map = cot_inference(bands, models)
        valid = np.isfinite(cot_map)
        cot_valid = cot_map[valid]
        per_scene.append({
            "date": npz.stem,
            "mean_cot": round(float(cot_valid.mean()), 5),
            "median_cot": round(float(np.median(cot_valid)), 5),
            "p90_cot": round(float(np.percentile(cot_valid, 90)), 5),
            "max_cot": round(float(cot_valid.max()), 5),
            "thick_cloud_frac": round(float((cot_valid >= 0.025).mean()), 4),
            "thin_cloud_frac": round(float(((cot_valid >= 0.015) & (cot_valid < 0.025)).mean()), 4),
            "clear_frac": round(float((cot_valid < 0.015).mean()), 4),
        })
        print(f"  [{i:>3}/{len(npz_files)}] {npz.stem}  mean_cot={per_scene[-1]['mean_cot']:.4f}  "
              f"thick={per_scene[-1]['thick_cloud_frac']:.2f}")
    cots = [p["mean_cot"] for p in per_scene]
    return {
        "name": name,
        "n_scenes": len(per_scene),
        "mean_cot": round(float(np.mean(cots)), 5) if cots else None,
        "median_cot": round(float(np.median(cots)), 5) if cots else None,
        "std_cot": round(float(np.std(cots)), 5) if cots else None,
        "min_cot": round(float(np.min(cots)), 5) if cots else None,
        "max_cot": round(float(np.max(cots)), 5) if cots else None,
        "mean_clear_frac": round(float(np.mean([p["clear_frac"] for p in per_scene])), 4) if per_scene else None,
        "mean_thick_cloud_frac": round(float(np.mean([p["thick_cloud_frac"] for p in per_scene])), 4) if per_scene else None,
        "per_scene": per_scene,
    }


def plot_mean_cot(stats: dict[str, dict], path: Path) -> None:
    names = list(stats.keys())
    means = [stats[n]["mean_cot"] for n in names]
    medians = [stats[n]["median_cot"] for n in names]
    stds = [stats[n]["std_cot"] for n in names]
    colors = [SET_COLORS[n] for n in names]
    labels = [f"{n}\nn={stats[n]['n_scenes']}" for n in names]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, width=0.55,
                  capsize=8, error_kw={"alpha": 0.5, "lw": 1.4})
    ax.scatter(x, medians, marker="D", color="#2c3e50", s=70,
               zorder=3, label="Median")

    ymax = max(means) if means else 1
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + ymax * 0.04,
                f"{m:.4f}", ha="center", va="bottom",
                fontweight="bold", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Cloud Optical Thickness")
    ax.set_title(
        "Mean COT per set — DES MLP5-ensemble (Pirinen 2024)\n"
        "Skåne (Lund-omgivning) · 2022-06-01 → 2022-08-31 · 11-band L2A · "
        "DES openEO"
    )
    ax.axhline(0.025, color="#c0392b", ls=":", lw=1.0, alpha=0.6,
               label="Thick-cloud thresh (0.025)")
    ax.axhline(0.015, color="#e67e22", ls=":", lw=1.0, alpha=0.6,
               label="Thin-cloud thresh (0.015)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_per_scene_cot(stats: dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.0))

    for name in stats:
        per = stats[name]["per_scene"]
        if not per:
            continue
        xs = [date.fromisoformat(p["date"]) for p in per]
        ys = [p["mean_cot"] for p in per]
        ax.scatter(xs, ys, color=SET_COLORS[name], s=60, alpha=0.85,
                   edgecolor="white", linewidth=0.5,
                   label=f"{name} (mean={stats[name]['mean_cot']:.4f})")

    ax.axhline(0.025, color="#c0392b", ls=":", lw=1.0, alpha=0.6,
               label="Thick-cloud thresh")
    ax.axhline(0.015, color="#e67e22", ls=":", lw=1.0, alpha=0.6,
               label="Thin-cloud thresh")
    ax.set_ylabel("Per-scen mean COT (AOI)")
    ax.set_xlabel("Datum (2022)")
    ax.set_title(
        "Per-scen mean COT — Jun–Aug 2022\n"
        "B_metafilter och C_fetch_s2 är delmängder av A_baseline; "
        "lägre = klarare"
    )
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    set_names = ["A_baseline", "B_metafilter", "C_fetch_s2"]

    # Load model ensemble once
    from imint.analyzers.cot import DEFAULT_MODEL_PATHS
    if not DEFAULT_MODEL_PATHS:
        sys.exit("No COT ensemble weights found in imint/fm/cot_models/")
    print(f"Loading {len(DEFAULT_MODEL_PATHS)} MLP5 ensemble members…")
    models = _load_ensemble(DEFAULT_MODEL_PATHS, device="cpu")

    stats = {}
    for n in set_names:
        print(f"\n--- {n} ---")
        stats[n] = process_set(n, models)

    output = {
        "model": {
            "type": "DES MLP5 ensemble",
            "reference": "Pirinen et al. 2024, Remote Sensing — github.com/DigitalEarthSweden/ml-cloud-opt-thick",
            "input_bands": COT_BAND_ORDER,
            "n_ensemble_members": len(DEFAULT_MODEL_PATHS),
            "thick_cloud_threshold": 0.025,
            "thin_cloud_threshold": 0.015,
        },
        "sets": {
            n: {k: v for k, v in s.items() if k != "per_scene"}
            for n, s in stats.items()
        },
        "per_scene": {n: s["per_scene"] for n, s in stats.items()},
    }
    with open(HERE / "cot_metrics.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    plot_mean_cot(stats, FIGS / "07_mean_cot_per_set.png")
    plot_per_scene_cot(stats, FIGS / "08_cot_per_scene.png")

    print("\n=== Mean COT per set ===")
    print(f"  {'Set':<14} {'n':>4} {'mean':>9} {'median':>9} {'std':>9} "
          f"{'clear%':>8} {'thick%':>8}")
    for n in set_names:
        s = stats[n]
        print(
            f"  {n:<14} {s['n_scenes']:>4} {s['mean_cot']:>9.4f} "
            f"{s['median_cot']:>9.4f} {s['std_cot']:>9.4f} "
            f"{s['mean_clear_frac']*100:>7.1f}% "
            f"{s['mean_thick_cloud_frac']*100:>7.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
