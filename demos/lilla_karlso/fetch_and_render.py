"""
demos/lilla_karlso/fetch_and_render.py

End-to-end demonstration of the Atmosfär-pipelinen on a real AOI:

    Lilla Karlsö (utanför Gotland, fågelreservat)
    Vår + sommar 2025 (april-september)

Uses imint.training.optimal_fetch.optimal_fetch_dates(mode="era5_then_scl")
to pick clear days, fetches all 11 L2A spectral bands via DES openEO,
runs the DES MLP5 COT ensemble, and writes per-scene frames + a manifest.

Outputs:
    docs/showcase/lilla_karlso/
        plan.json            — selection trace
        manifest.json        — per-scene record (date, mean COT, frame path)
        frames/<date>.jpg    — RGB | COT-heatmap composite per scene

Run:
    python demos/lilla_karlso/fetch_and_render.py
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from imint.config.env import load_env
load_env()

from imint.fetch import fetch_seasonal_image, S2L2A_SPECTRAL_BANDS  # noqa: E402
from imint.training.optimal_fetch import optimal_fetch_dates  # noqa: E402
from imint.analyzers.cot import (  # noqa: E402
    DEFAULT_MODEL_PATHS, _load_ensemble, cot_inference,
)

# ── AOI / period ───────────────────────────────────────────────────────────
AOI_NAME = "Lilla Karlsö"
BBOX = {
    # Lilla Karlsö ligger på 57.311°N 18.061°E (Wikipedia, kontroll mot
    # Lantmäteriet). ~7 km × 5 km AOI som täcker hela ön plus en strimma
    # hav i alla riktningar — bra för att se kalkstensplatån + sealife.
    "west":  18.02, "south": 57.28,
    "east":  18.10, "north": 57.34,
}
PERIOD_START = "2025-04-01"
PERIOD_END   = "2025-09-30"

# ── Workers / thresholds ───────────────────────────────────────────────────
N_WORKERS      = 6
MAX_AOI_CLOUD  = 0.10
SCENE_CC_MAX   = 30.0

# ── Output paths ───────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
DOCS_OUT    = REPO_ROOT / "docs" / "showcase" / "lilla_karlso"
FRAMES_DIR  = DOCS_OUT / "frames"
SPEC_CACHE  = HERE / "cache_11band"
DOCS_OUT.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
SPEC_CACHE.mkdir(parents=True, exist_ok=True)

# ── Frame styling ───────────────────────────────────────────────────────────
# Per-scene percentile stretch — Östersjö-vatten är mörkt (BOA blue ~0.02),
# kalkstensplatån är ljus (~0.20), så ett fast 0–0.30 stretch ger antingen
# helt svart hav eller utbränd platå. Använd p2/p98 per scen istället.
COT_VMAX = 0.5
COT_CMAP = LinearSegmentedColormap.from_list(
    "cot", [
        (0.00, "#0d4d2a"), (0.05, "#27ae60"),
        (0.10, "#f1c40f"), (0.30, "#e67e22"),
        (1.00, "#ecf0f1"),
    ],
)


def to_rgb(arr: np.ndarray) -> np.ndarray:
    """Per-scene p2/p98 stretch; robust against pure-water or pure-land AOIs."""
    band = {b: arr[i] for i, b in enumerate(S2L2A_SPECTRAL_BANDS)}
    rgb = np.stack([band["B04"], band["B03"], band["B02"]], axis=-1)
    valid = np.isfinite(rgb).all(axis=-1)
    if not valid.any():
        return np.zeros(rgb.shape, dtype=np.uint8)
    lo = np.percentile(rgb[valid], 2)
    hi = np.percentile(rgb[valid], 98)
    if hi <= lo:
        hi = lo + 1e-3
    rgb = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def fetch_one(date_str: str) -> dict:
    """Fetch spectral, cache to .npz, return metadata."""
    cache = SPEC_CACHE / f"{date_str}.npz"
    if cache.exists():
        return {"date": date_str, "elapsed_s": 0.0, "from_cache": True,
                "success": True}
    t0 = time.time()
    try:
        result = fetch_seasonal_image(date=date_str, coords=BBOX, source="des")
        if result is None:
            return {"date": date_str, "elapsed_s": round(time.time()-t0, 2),
                    "success": False, "from_cache": False, "error": "None"}
        arr, _ = result
        np.savez_compressed(cache, arr=arr.astype(np.float32))
        return {"date": date_str, "elapsed_s": round(time.time()-t0, 2),
                "success": True, "from_cache": False, "shape": list(arr.shape)}
    except Exception as e:
        return {"date": date_str, "elapsed_s": round(time.time()-t0, 2),
                "success": False, "from_cache": False, "error": str(e)[:200]}


def render_frame(arr: np.ndarray, cot: np.ndarray, date: str) -> dict:
    rgb = to_rgb(arr)
    valid = np.isfinite(cot)
    mean_cot   = float(cot[valid].mean()) if valid.any() else float("nan")
    median_cot = float(np.median(cot[valid])) if valid.any() else float("nan")
    thick_frac = float((cot[valid] >= 0.025).mean()) if valid.any() else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0), facecolor="#0f0f0f")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05, wspace=0.04)
    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("RGB (B04/B03/B02)", color="white", fontsize=10, pad=6)
    axes[0].axis("off")

    im = axes[1].imshow(cot, cmap=COT_CMAP, vmin=0, vmax=COT_VMAX,
                        interpolation="nearest")
    axes[1].set_title(f"COT — mean = {mean_cot:.4f}",
                     color="white", fontsize=10, pad=6, fontweight="bold")
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.02)
    cbar.set_label("COT", color="white", fontsize=8)
    cbar.ax.tick_params(colors="white", labelsize=7)
    cbar.outline.set_edgecolor("#444")

    fig.suptitle(
        f"{AOI_NAME}  ·  {date}  ·  thick-cloud-pixels {thick_frac*100:.0f}%",
        color="#27ae60", fontsize=11, fontweight="bold", y=0.97,
    )
    out = FRAMES_DIR / f"{date}.jpg"
    fig.savefig(out, dpi=110, facecolor="#0f0f0f", bbox_inches="tight")
    plt.close(fig)

    return {
        "date": date,
        "mean_cot":         round(mean_cot, 5),
        "median_cot":       round(median_cot, 5),
        "thick_cloud_frac": round(thick_frac, 4),
        "frame_path":       str(out.relative_to(REPO_ROOT / "docs")),
    }


def main() -> int:
    print(f"AOI: {AOI_NAME}  bbox={BBOX}")
    print(f"Period: {PERIOD_START} → {PERIOD_END}")

    # ── Stage 1+2: optimal selection ────────────────────────────────────
    print("\n[1/4] Kör ERA5 → SCL urvalspipeline …")
    plan = optimal_fetch_dates(
        BBOX, PERIOD_START, PERIOD_END,
        mode="era5_then_scl", max_aoi_cloud=MAX_AOI_CLOUD,
        scene_cloud_max=SCENE_CC_MAX,
    )
    plan_data = {
        "aoi":                      AOI_NAME,
        "bbox_wgs84":               BBOX,
        "period":                   [PERIOD_START, PERIOD_END],
        "mode":                     plan.mode,
        "max_aoi_cloud":            MAX_AOI_CLOUD,
        "scene_cloud_max":          SCENE_CC_MAX,
        "selection_elapsed_s":      plan.elapsed_s,
        "candidates_after_stage":   plan.n_candidates_after,
        "selected_dates":           plan.dates,
    }
    with open(DOCS_OUT / "plan.json", "w") as f:
        json.dump(plan_data, f, indent=2, ensure_ascii=False)

    print(f"  ERA5: {plan.elapsed_s.get('era5', 0):.1f}s, "
          f"SCL-stack: {plan.elapsed_s.get('scl_stack', 0):.1f}s")
    print(f"  Kandidater per stage: {plan.n_candidates_after}")
    print(f"  Valda datum ({len(plan.dates)}): {plan.dates}")
    if not plan.dates:
        print("  Inga datum passerar filtren — avbryter.")
        return 0

    # ── Stage 3: spectral fetch ─────────────────────────────────────────
    print(f"\n[2/4] Hämtar spektral för {len(plan.dates)} datum (DES openEO, "
          f"{N_WORKERS} workers) …")
    t0 = time.time()
    spec_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(fetch_one, d): d for d in plan.dates}
        for fut in as_completed(futs):
            r = fut.result()
            spec_records.append(r)
            tag = "cache" if r["from_cache"] else ("OK" if r["success"] else "FAIL")
            print(f"  {r['date']}  {tag:<5}  ({r['elapsed_s']}s)")
    spec_wall = time.time() - t0
    print(f"  Total spektral wall-clock: {spec_wall:.1f}s")

    # ── Stage 4: COT inference + frames ─────────────────────────────────
    print(f"\n[3/4] Laddar MLP5 COT-ensemble + renderar frames …")
    models = _load_ensemble(DEFAULT_MODEL_PATHS, device="cpu")
    frames: list[dict] = []
    for d in sorted(plan.dates):
        npz = SPEC_CACHE / f"{d}.npz"
        if not npz.exists():
            continue
        arr = np.load(npz)["arr"]
        bands = {b: arr[i] for i, b in enumerate(S2L2A_SPECTRAL_BANDS)}
        cot = cot_inference(bands, models)
        frames.append(render_frame(arr, cot, d))
        print(f"  {d}  mean COT {frames[-1]['mean_cot']:.4f}")

    # Sort by mean COT (clearest first, like the Atmosfär tab)
    frames.sort(key=lambda r: r["mean_cot"])

    cots = [f["mean_cot"] for f in frames]
    summary = {
        "n_scenes":         len(frames),
        "mean_cot":         round(float(np.mean(cots)), 5) if cots else None,
        "median_cot":       round(float(np.median(cots)), 5) if cots else None,
        "min_cot":          round(float(np.min(cots)), 5) if cots else None,
        "max_cot":          round(float(np.max(cots)), 5) if cots else None,
        "spec_wall_s":      round(spec_wall, 2),
        "selection_total_s": round(sum(plan.elapsed_s.values()), 2),
        "total_wall_s":     round(spec_wall + sum(plan.elapsed_s.values()), 2),
    }

    print(f"\n[4/4] Skriver manifest + plan-data …")
    with open(DOCS_OUT / "manifest.json", "w") as f:
        json.dump({
            "aoi":      AOI_NAME,
            "period":   [PERIOD_START, PERIOD_END],
            "summary":  summary,
            "plan":     plan_data,
            "frames":   frames,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n=== Sammanfattning ===")
    print(f"  AOI:           {AOI_NAME}")
    print(f"  Period:        {PERIOD_START} → {PERIOD_END}")
    print(f"  Valda datum:   {len(frames)} (av {plan.n_candidates_after.get('era5', '?')} ERA5-kandidater)")
    print(f"  Selection:     {summary['selection_total_s']}s")
    print(f"  Spektral:      {summary['spec_wall_s']}s")
    print(f"  Total:         {summary['total_wall_s']}s")
    if summary["mean_cot"] is not None:
        print(f"  Mean COT:      {summary['mean_cot']:.4f}")
        print(f"  Median COT:    {summary['median_cot']:.4f}")
    print(f"\nArtefakter: {DOCS_OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
