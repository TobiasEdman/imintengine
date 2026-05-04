"""
demos/era5_metafilter/benchmark_three_sets.py

Three-set benchmark: actually fetches Sentinel-2 L2A scenes via DES openEO and
compares wall-clock time + data preservation between selection strategies.

Sets:
    A. baseline   — every S2-L2A scene over AOI for Jun-Jul-Aug 2022 (STAC,
                    no filter). Ground truth for what cloud-free AOI tiles
                    actually exist.
    B. metafilter — only dates passing the ERA5 weather pre-filter
                    (mirrors github.com/erikkallman/metafilter).
    C. fetch_s2   — dates surviving the existing pipeline's STAC cloud filter
                    (scene_cloud_max=30 %, mirrors _stac_available_dates in
                    imint/training/tile_fetch.py).

Each set is fetched LIVE through DES openEO with 6 parallel workers and its
own cache directory, so wall-clock times are independent and fair.

For each fetched tile we measure "cloud-free over AOI" via a B02 (blue-band)
brightness heuristic — pixels with B02 > 0.18 reflectance are likely cloud-
contaminated; if <10 % of AOI pixels are bright, the tile is considered
usable.

Outputs:
    benchmark_metrics.json
    figures/
        04_benchmark_time.png
        05_benchmark_data_loss.png
        06_benchmark_per_date.png

Run:
    python demos/era5_metafilter/benchmark_three_sets.py        # use cache if available
    python demos/era5_metafilter/benchmark_three_sets.py --live # ignore cache, refetch
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from imint.config.env import load_env
load_env()

from imint.fetch import fetch_seasonal_image  # noqa: E402

# ---------- Setup ----------
HERE = Path(__file__).parent
DATA = HERE / "data"
FIGS = HERE / "figures"
CACHE_ROOT = HERE / "cache_11band"   # per-set fetch caches (11-band, COT-ready)
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)
CACHE_ROOT.mkdir(exist_ok=True)

# AOI + period — same as replicate_metafilter.py, but Jun-Aug only
AOI_NAME = "Skåne (Lund-omgivning)"
BBOX_LL = (13.05, 55.65, 13.35, 55.80)
COORDS = {
    "west": BBOX_LL[0], "south": BBOX_LL[1],
    "east": BBOX_LL[2], "north": BBOX_LL[3],
}
PERIOD_START = date(2022, 6, 1)
PERIOD_END = date(2022, 8, 31)

N_WORKERS = 6
CLOUD_BLUE_THRESHOLD = 0.18      # B02 reflectance above which a pixel is likely cloud
AOI_BRIGHT_FRAC_LIMIT = 0.10     # AOI usable if <10 % bright pixels
SCENE_CLOUD_MAX_PCT = 30.0       # mirrors existing _stac_available_dates default


# ---------- Selection (re-uses cached STAC + ERA5 from replicate_metafilter.py) ----------

def load_stac() -> list[dict]:
    """Load cached STAC features and filter to Jun-Aug."""
    stac_cache = DATA / "stac_skane_2022.json"
    if not stac_cache.exists():
        sys.exit(
            f"Missing STAC cache: {stac_cache}\n"
            f"Run replicate_metafilter.py first to populate it."
        )
    with open(stac_cache) as f:
        payload = json.load(f)
    items = []
    for feat in payload["features"]:
        props = feat.get("properties", {})
        dt = props.get("datetime", "")[:10]
        cc = props.get("eo:cloud_cover")
        if not dt or cc is None:
            continue
        d = date.fromisoformat(dt)
        if PERIOD_START <= d <= PERIOD_END:
            items.append({"date": dt, "cloud_cover": float(cc), "id": feat.get("id", "")})
    # If multiple scenes per date (overlapping orbits), keep the one with lowest cc
    by_date: dict[str, dict] = {}
    for it in items:
        prev = by_date.get(it["date"])
        if prev is None or it["cloud_cover"] < prev["cloud_cover"]:
            by_date[it["date"]] = it
    return sorted(by_date.values(), key=lambda x: x["date"])


def load_era5_good_days() -> set[str]:
    """Re-derive metafilter pass dates using the same rule as replicate_metafilter.py."""
    era5_cache = DATA / "era5_skane_2022.json"
    if not era5_cache.exists():
        sys.exit(f"Missing ERA5 cache: {era5_cache} — run replicate_metafilter.py first")
    with open(era5_cache) as f:
        payload = json.load(f)
    daily = payload["daily"]
    by_date = {
        d: {"t2m": t, "tp": p}
        for d, t, p in zip(
            daily["time"], daily["temperature_2m_mean"], daily["precipitation_sum"]
        )
        if t is not None and p is not None
    }
    keep: set[str] = set()
    for d_str, w in by_date.items():
        d = date.fromisoformat(d_str)
        prev_sum = 0.0
        from datetime import timedelta
        for off in (1, 2):
            prev = by_date.get((d - timedelta(days=off)).isoformat())
            if prev is not None:
                prev_sum += prev["tp"]
        if w["tp"] <= 0.5 and prev_sum <= 3.0 and w["t2m"] >= 10.0:
            keep.add(d_str)
    return keep


def build_sets() -> dict[str, list[dict]]:
    stac = load_stac()
    good = load_era5_good_days()

    set_a = stac
    set_b = [s for s in stac if s["date"] in good]
    set_c = [s for s in stac if s["cloud_cover"] <= SCENE_CLOUD_MAX_PCT]

    return {"A_baseline": set_a, "B_metafilter": set_b, "C_fetch_s2": set_c}


# ---------- Cloud-free-over-AOI check ----------

def aoi_bright_fraction(arr: np.ndarray) -> float:
    """Fraction of AOI pixels whose B02 reflectance exceeds the cloud threshold.

    arr is (N, H, W); band 0 = B02 (blue). Saturated/bright pixels are clouds.
    """
    blue = arr[0]
    valid = np.isfinite(blue)
    if not valid.any():
        return 1.0  # treat all-nodata as fully clouded
    bright = (blue > CLOUD_BLUE_THRESHOLD) & valid
    return float(bright.sum() / valid.sum())


# ---------- Fetcher with per-set cache ----------

def fetch_one(date_str: str, cache_dir: Path) -> dict:
    """Fetch one date via DES openEO; cache the array as .npz."""
    cache_path = cache_dir / f"{date_str}.npz"
    cache_meta = cache_dir / f"{date_str}.json"

    if cache_path.exists() and cache_meta.exists():
        with open(cache_meta) as f:
            meta = json.load(f)
        meta["from_cache"] = True
        return meta

    t0 = time.time()
    error = None
    bright_frac: float | None = None
    success = False
    try:
        result = fetch_seasonal_image(date=date_str, coords=COORDS, source="des")
        if result is not None:
            arr, _ = result
            bright_frac = aoi_bright_fraction(arr)
            np.savez_compressed(cache_path, arr=arr.astype(np.float32))
            success = True
        else:
            error = "fetch_returned_none"
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)[:200]}"

    elapsed = time.time() - t0
    meta = {
        "date": date_str,
        "elapsed_s": round(elapsed, 2),
        "success": success,
        "bright_frac": bright_frac,
        "cloud_free_aoi": (
            bright_frac is not None and bright_frac < AOI_BRIGHT_FRAC_LIMIT
        ),
        "error": error,
        "from_cache": False,
    }
    with open(cache_meta, "w") as f:
        json.dump(meta, f)
    return meta


def run_set(name: str, items: list[dict], reset_cache: bool) -> dict:
    cache_dir = CACHE_ROOT / name
    if reset_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Set {name}: {len(items)} fetches, {N_WORKERS} workers ---")
    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(fetch_one, it["date"], cache_dir): it for it in items}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            tag = (
                "cache" if r["from_cache"]
                else ("OK" if r["success"] else "FAIL")
            )
            cf = "cf" if r["cloud_free_aoi"] else "  "
            print(
                f"  [{i:>3}/{len(items)}] {r['date']} "
                f"{tag:<5} {cf} bright={r.get('bright_frac') or 0:.2f} "
                f"({r['elapsed_s']}s)"
            )
    wall = time.time() - t0

    n_ok = sum(1 for r in results if r["success"] or r["from_cache"])
    n_cf = sum(1 for r in results if r["cloud_free_aoi"])
    n_cache = sum(1 for r in results if r["from_cache"])
    n_live = len(results) - n_cache
    return {
        "name": name,
        "n_dates": len(items),
        "n_success": n_ok,
        "n_cloud_free_aoi": n_cf,
        "n_from_cache": n_cache,
        "n_live": n_live,
        "wall_s": round(wall, 1),
        "wall_per_live_fetch_s": round(wall / max(n_live, 1), 1),
        "results": results,
    }


# ---------- Figures ----------

def plot_time(stats: dict[str, dict], path: Path) -> None:
    names = list(stats.keys())
    times = [stats[n]["wall_s"] for n in names]
    n_live = [stats[n]["n_live"] for n in names]
    labels = [
        f"{n}\n{stats[n]['n_dates']} dates ({stats[n]['n_live']} live)"
        for n in names
    ]
    colors = ["#7f8c8d", "#27ae60", "#2980b9"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, times, color=colors, width=0.6)
    for b, t, nl in zip(bars, times, n_live):
        ax.text(b.get_x() + b.get_width() / 2, t + max(times) * 0.01,
                f"{t:.0f}s", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Wall-clock fetch time (s)")
    ax.set_title(
        f"Fetch-tid per set — DES openEO, {N_WORKERS} workers\n"
        f"{AOI_NAME}, {PERIOD_START} → {PERIOD_END}"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_data_loss(stats: dict[str, dict], path: Path) -> None:
    """How many cloud-free AOI tiles does each set capture?"""
    truth = stats["A_baseline"]["n_cloud_free_aoi"]

    names = list(stats.keys())
    n_total = [stats[n]["n_dates"] for n in names]
    n_cf = [stats[n]["n_cloud_free_aoi"] for n in names]
    cf_capture_pct = [round(100 * c / max(truth, 1), 1) for c in n_cf]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, n_total, w, label="Total fetches", color="#bdc3c7")
    bars2 = ax.bar(x + w/2, n_cf, w, label="Cloud-free over AOI", color="#27ae60")

    for b, v in zip(bars1, n_total):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v),
                ha="center", va="bottom", fontsize=9)
    for b, v, pct in zip(bars2, n_cf, cf_capture_pct):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v}\n({pct}%)",
                ha="center", va="bottom", fontsize=9, color="#27ae60",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Antal scener")
    ax.set_title(
        f"Cloud-free-AOI träffsäkerhet "
        f"(A=truth, B/C visar % av A:s cloud-free fångade)\n"
        f"{AOI_NAME}, {PERIOD_START} → {PERIOD_END}"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_per_date(stats: dict[str, dict], path: Path) -> None:
    """Per-date strip showing which set picked each date and cloud-free verdict."""
    set_names = ["A_baseline", "B_metafilter", "C_fetch_s2"]
    by_set_dates: dict[str, set[str]] = {n: set() for n in set_names}
    by_date_cf: dict[str, bool] = {}
    for n in set_names:
        for r in stats[n]["results"]:
            by_set_dates[n].add(r["date"])
            if r["date"] not in by_date_cf:
                by_date_cf[r["date"]] = r["cloud_free_aoi"]

    all_dates = sorted({d for s in by_set_dates.values() for d in s})
    if not all_dates:
        return
    xs = [date.fromisoformat(d) for d in all_dates]

    fig, axes = plt.subplots(3, 1, figsize=(11, 4.8), sharex=True)
    colors = {"A_baseline": "#7f8c8d", "B_metafilter": "#27ae60", "C_fetch_s2": "#2980b9"}
    for ax, name in zip(axes, set_names):
        for d, x in zip(all_dates, xs):
            in_set = d in by_set_dates[name]
            cf = by_date_cf.get(d, False)
            if in_set and cf:
                ax.scatter([x], [1], color=colors[name], s=80,
                           edgecolor="white", linewidth=0.5, zorder=3)
            elif in_set:
                ax.scatter([x], [1], color=colors[name], s=40, alpha=0.45)
            else:
                ax.scatter([x], [1], color="#ecf0f1", s=20)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
        ax.set_xlim(xs[0], xs[-1])
    axes[-1].set_xlabel("Datum 2022 (Jun–Aug)")
    fig.suptitle(
        "Per-datum: vilka scener fångas av varje set?  "
        "(stora gröna prickar = cloud-free över AOI)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------- Main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--live", action="store_true",
        help="Clear caches and refetch every set live (for honest timing).",
    )
    args = ap.parse_args()

    sets = build_sets()
    print(f"AOI:    {AOI_NAME}  bbox={BBOX_LL}")
    print(f"Period: {PERIOD_START} → {PERIOD_END}")
    print(f"Sets:")
    for k, v in sets.items():
        print(f"  {k}: {len(v)} dates")

    stats = {}
    for name, items in sets.items():
        stats[name] = run_set(name, items, reset_cache=args.live)

    metrics = {
        "aoi": AOI_NAME,
        "bbox_ll": list(BBOX_LL),
        "period": [PERIOD_START.isoformat(), PERIOD_END.isoformat()],
        "n_workers": N_WORKERS,
        "cloud_blue_threshold": CLOUD_BLUE_THRESHOLD,
        "aoi_bright_frac_limit": AOI_BRIGHT_FRAC_LIMIT,
        "scene_cloud_max_pct": SCENE_CLOUD_MAX_PCT,
        "sets": {
            name: {
                k: v for k, v in s.items() if k != "results"
            } for name, s in stats.items()
        },
    }
    truth = stats["A_baseline"]["n_cloud_free_aoi"]
    for n, s in stats.items():
        metrics["sets"][n]["cloud_free_capture_pct_vs_A"] = round(
            100 * s["n_cloud_free_aoi"] / max(truth, 1), 1
        )

    with open(HERE / "benchmark_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_time(stats, FIGS / "04_benchmark_time.png")
    plot_data_loss(stats, FIGS / "05_benchmark_data_loss.png")
    plot_per_date(stats, FIGS / "06_benchmark_per_date.png")

    print("\n=== Sammanfattning ===")
    print(
        f"  {'Set':<14} {'dates':>6} {'live':>6} {'wall(s)':>8} "
        f"{'CF/AOI':>8} {'capture vs A':>14}"
    )
    for n, s in stats.items():
        cap = metrics["sets"][n]["cloud_free_capture_pct_vs_A"]
        print(
            f"  {n:<14} {s['n_dates']:>6} {s['n_live']:>6} "
            f"{s['wall_s']:>8.1f} {s['n_cloud_free_aoi']:>8} {cap:>13.1f}%"
        )
    print(f"\nArtefakter: {HERE.relative_to(Path.cwd())}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
