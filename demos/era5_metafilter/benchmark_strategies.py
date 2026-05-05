"""
demos/era5_metafilter/benchmark_strategies.py

Real, end-to-end measurement of six fetch-selection strategies against
DES openEO. For each strategy we measure:

    * Selection wall-clock — ERA5 + STAC + SCL-stack, summed honestly.
    * Spectral fetch wall-clock — actually fetches the selected dates with
      6 parallel workers.
    * Total wall-clock — sum of the above (caller's experience).
    * API-call counts — STAC search, openEO SCL, openEO spectral.
    * Mean COT — re-uses cached 11-band tiles + the imint.analyzers.cot
      MLP5 ensemble.

Strategies (mode names mirror imint.training.optimal_fetch):

    M0  stac_only       — naive baseline
    M1  atmosphere      — ERA5 prefilter only
    M2  stac_then_scl   — Imint's current production pipeline
    M3  scl_only        — SCL-stack alone
    M4  era5_then_scl   — RECOMMENDED chain
    M5  era5_then_stac  — light variant (no openEO SCL call)

Run:
    python demos/era5_metafilter/benchmark_strategies.py        # use spectral cache
    python demos/era5_metafilter/benchmark_strategies.py --live # ignore cache, refetch
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from imint.config.env import load_env
load_env()

from imint.fetch import fetch_seasonal_image, S2L2A_SPECTRAL_BANDS  # noqa: E402
from imint.training.optimal_fetch import optimal_fetch_dates  # noqa: E402
from imint.analyzers.cot import (  # noqa: E402
    DEFAULT_MODEL_PATHS, _load_ensemble, cot_inference,
)

HERE = Path(__file__).parent
SPEC_CACHE = HERE / "cache_11band" / "A_baseline"
OUT_PATH = HERE / "strategies_metrics.json"

# ── AOI / period — same as the rest of the showcase ────────────────────────
AOI_NAME = "Skåne (Lund-omgivning)"
BBOX = {
    "west":  13.05, "south": 55.65,
    "east":  13.35, "north": 55.80,
}
PERIOD_START = "2022-06-01"
PERIOD_END   = "2022-08-31"
N_WORKERS    = 6
MAX_AOI_CLOUD = 0.10
STAC_CLOUD_MAX = 30.0

STRATEGIES = [
    ("M0_stac_only",      "stac_only",      "Naiv STAC"),
    ("M1_atmosphere",     "atmosphere",     "Atmosfär ensam"),
    ("M2_stac_then_scl",  "stac_then_scl",  "Nuvarande pipeline"),
    ("M3_scl_only",       "scl_only",       "SCL-stack ensam"),
    ("M4_era5_then_scl",  "era5_then_scl",  "ERA5 → SCL (rekommenderat)"),
    ("M5_era5_then_stac", "era5_then_stac", "ERA5 → STAC (lättviktig)"),
]


# ── Spectral fetch with cache ─────────────────────────────────────────────

def fetch_one_spectral(date_str: str, use_cache: bool) -> dict:
    cache_path = SPEC_CACHE / f"{date_str}.npz"
    cache_meta = SPEC_CACHE / f"{date_str}.json"

    if use_cache and cache_path.exists() and cache_meta.exists():
        with open(cache_meta) as f:
            meta = json.load(f)
        return {
            "date": date_str,
            "elapsed_s": meta.get("elapsed_s", 0.0),
            "from_cache": True,
            "success": True,
        }

    t0 = time.time()
    try:
        result = fetch_seasonal_image(date=date_str, coords=BBOX, source="des")
        success = result is not None
        if success:
            arr, _ = result
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, arr=arr.astype(np.float32))
            meta = {"date": date_str, "elapsed_s": round(time.time() - t0, 2),
                    "success": True}
            with open(cache_meta, "w") as f:
                json.dump(meta, f)
    except Exception as e:
        success = False
        meta = {"date": date_str, "elapsed_s": round(time.time() - t0, 2),
                "success": False, "error": str(e)[:200]}
    return {
        "date": date_str,
        "elapsed_s": round(time.time() - t0, 2),
        "from_cache": False,
        "success": success,
    }


def fetch_spectral_parallel(dates: list[str], use_cache: bool) -> tuple[float, list[dict]]:
    """Returns (wall_clock_s, per_date_records)."""
    if not dates:
        return 0.0, []
    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(fetch_one_spectral, d, use_cache): d for d in dates}
        for fut in as_completed(futs):
            results.append(fut.result())
    wall = time.time() - t0
    # If cache hit, the wall-clock is artificially short — estimate the
    # parallel real-fetch wall from per-date elapsed times.
    if use_cache and any(r["from_cache"] for r in results):
        per = sorted((r["elapsed_s"] for r in results), reverse=True)
        # Round-robin assignment to N_WORKERS gives an estimate of the
        # parallel completion time:
        slots = [0.0] * N_WORKERS
        for t in per:
            i = slots.index(min(slots))
            slots[i] += t
        wall_est = max(slots)
        return wall_est, results
    return wall, results


# ── Mean COT for a date list (uses cached 11-band tiles) ──────────────────

def mean_cot_for_dates(dates: list[str], models) -> dict:
    cots: list[float] = []
    per_date: list[dict] = []
    for d in dates:
        npz = SPEC_CACHE / f"{d}.npz"
        if not npz.exists():
            continue
        arr = np.load(npz)["arr"]
        bands = {b: arr[i] for i, b in enumerate(S2L2A_SPECTRAL_BANDS)}
        cot = cot_inference(bands, models)
        valid = np.isfinite(cot)
        cv = cot[valid]
        m = float(cv.mean()) if cv.size else float("nan")
        cots.append(m)
        per_date.append({
            "date": d,
            "mean_cot":         round(m, 5),
            "thick_cloud_frac": round(float((cv >= 0.025).mean()), 4),
            "clear_frac":       round(float((cv < 0.015).mean()), 4),
        })
    if not cots:
        return {"n": 0, "mean_cot": None, "median_cot": None, "per_scene": []}
    return {
        "n":           len(cots),
        "mean_cot":    round(float(np.mean(cots)), 5),
        "median_cot":  round(float(np.median(cots)), 5),
        "std_cot":     round(float(np.std(cots)), 5),
        "min_cot":     round(float(np.min(cots)), 5),
        "max_cot":     round(float(np.max(cots)), 5),
        "mean_clear_frac":       round(float(np.mean([r["clear_frac"] for r in per_date])), 4),
        "mean_thick_cloud_frac": round(float(np.mean([r["thick_cloud_frac"] for r in per_date])), 4),
        "per_scene":   per_date,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="Force live spectral re-fetch (ignore cache).")
    args = ap.parse_args()
    use_cache = not args.live

    print(f"AOI:    {AOI_NAME}  bbox={BBOX}")
    print(f"Period: {PERIOD_START} → {PERIOD_END}")
    print(f"Workers: {N_WORKERS}, max_aoi_cloud={MAX_AOI_CLOUD}, "
          f"scene_cloud_max={STAC_CLOUD_MAX}")
    print(f"Spectral cache: {'used' if use_cache else 'IGNORED (live refetch)'}")

    print("\nLoading COT MLP5 ensemble…")
    models = _load_ensemble(DEFAULT_MODEL_PATHS, device="cpu")

    summary: dict[str, dict] = {}
    for key, mode, label in STRATEGIES:
        print(f"\n=== {key}: {label} ({mode}) ===")
        plan = optimal_fetch_dates(
            BBOX, PERIOD_START, PERIOD_END,
            mode=mode, max_aoi_cloud=MAX_AOI_CLOUD, scene_cloud_max=STAC_CLOUD_MAX,
        )
        print(f"  selection elapsed: {plan.elapsed_s}")
        print(f"  candidates: {plan.n_candidates_after}")
        print(f"  selected dates ({len(plan.dates)}): {plan.dates}")

        spec_wall, spec_rec = fetch_spectral_parallel(plan.dates, use_cache)
        print(f"  spectral wall-clock (n={len(plan.dates)}, "
              f"workers={N_WORKERS}): {spec_wall:.1f}s "
              f"{'(estimated from cache)' if use_cache else '(live)'}")

        cot = mean_cot_for_dates(plan.dates, models)
        print(f"  mean COT: {cot['mean_cot']} (n={cot['n']})")

        select_total = sum(plan.elapsed_s.values())
        total = select_total + spec_wall

        api_calls = {
            "stac":           1 if "stac" in plan.elapsed_s else 0,
            "era5":           1 if "era5" in plan.elapsed_s else 0,
            "scl_stack":      1 if "scl_stack" in plan.elapsed_s else 0,
            "spectral_fetch": len(plan.dates),
        }
        api_calls["total_openEO"] = api_calls["scl_stack"] + api_calls["spectral_fetch"]

        summary[key] = {
            "label": label,
            "mode": mode,
            "selection_elapsed_s": plan.elapsed_s,
            "selection_total_s": round(select_total, 2),
            "spectral_wall_s": round(spec_wall, 2),
            "total_wall_s": round(total, 2),
            "n_dates": len(plan.dates),
            "candidates_after_stage": plan.n_candidates_after,
            "api_calls": api_calls,
            "cot": cot,
            "dates": plan.dates,
        }

    out = {
        "aoi": AOI_NAME,
        "bbox_wgs84": BBOX,
        "period": [PERIOD_START, PERIOD_END],
        "n_workers": N_WORKERS,
        "max_aoi_cloud": MAX_AOI_CLOUD,
        "scene_cloud_max": STAC_CLOUD_MAX,
        "cache_used": use_cache,
        "strategies": summary,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print(f"  {'Strategi':<32} {'n':>3} {'sel(s)':>7} {'spec(s)':>7} "
          f"{'tot(s)':>7} {'mean COT':>9}")
    print("-" * 90)
    for key, _, _ in STRATEGIES:
        s = summary[key]
        cot = s["cot"]["mean_cot"]
        cot_s = f"{cot:.4f}" if cot is not None else "  ―  "
        print(f"  {s['label']:<32} {s['n_dates']:>3} "
              f"{s['selection_total_s']:>7.1f} "
              f"{s['spectral_wall_s']:>7.1f} "
              f"{s['total_wall_s']:>7.1f} "
              f"{cot_s:>9}")
    print(f"\nWritten: {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
