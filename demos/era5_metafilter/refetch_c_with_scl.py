"""
demos/era5_metafilter/refetch_c_with_scl.py

Re-defines C_fetch_s2 to mirror the *full* production pipeline:
    STAC eo:cloud_cover ≤ 30 %   (filter 1, used in original benchmark)
    AND
    AOI-SCL cloud fraction ≤ 0.10 (filter 2, missing in original benchmark)

The first benchmark only applied filter 1 — meaning the published C set
contained scenes that the actual pipeline would reject after fetching the
SCL band and finding the AOI clouded over (despite the granule average
looking clean).

This script:
  1. Loads the 36 A_baseline candidate dates from cot_metrics.json.
  2. Re-fetches each via the production fetch_des_data path with
     cloud_threshold=0.10 and include_scl=True (same as
     imint/training/cdse_s2.py defaults).
  3. Records pass/reject per date plus AOI SCL cloud fraction.
  4. The dates that ALSO have STAC granule cc ≤ 30 % AND survive the
     SCL filter form the new C_fetch_s2_full set.
  5. Computes mean COT for that new set using the already-cached 11-band
     tiles (no need to re-run COT on the SCL-fetched arrays).

Result is written to:
    demos/era5_metafilter/c_full_scl_results.json
    demos/era5_metafilter/c_full_metrics.json (mean COT per new set)

The set names referenced everywhere else (manifest.json, frames/, summary
cards) are left untouched — a follow-up wires the new metrics into the
showcase tab.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from imint.config.env import load_env
load_env()

from imint.fetch import fetch_des_data, FetchError, S2L2A_SPECTRAL_BANDS  # noqa: E402
from imint.analyzers.cot import (  # noqa: E402
    DEFAULT_MODEL_PATHS, _load_ensemble, cot_inference,
)

HERE = Path(__file__).parent
CACHE_ROOT = HERE / "cache_11band"
N_WORKERS = 6

COORDS = {
    "west": 13.05, "south": 55.65,
    "east": 13.35, "north": 55.80,
}
SCL_CLOUD_THRESHOLD = 0.10        # production default
STAC_CC_MAX = 30.0                # original C set definition


def load_a_baseline() -> list[dict]:
    with open(HERE / "cot_metrics.json") as f:
        m = json.load(f)
    return m["per_scene"]["A_baseline"]


def load_stac_cc() -> dict[str, float]:
    """Map date → STAC granule eo:cloud_cover from cached payload."""
    with open(HERE / "data" / "stac_skane_2022.json") as f:
        payload = json.load(f)
    out: dict[str, float] = {}
    for feat in payload["features"]:
        props = feat.get("properties", {})
        d = props.get("datetime", "")[:10]
        cc = props.get("eo:cloud_cover")
        if not d or cc is None:
            continue
        prev = out.get(d)
        if prev is None or cc < prev:
            out[d] = float(cc)
    return out


def probe_one(date_str: str) -> dict:
    """Try the production fetch path; record SCL cloud fraction."""
    t0 = time.time()
    rec = {
        "date": date_str,
        "scl_pass": False,
        "scl_cloud_fraction": None,
        "elapsed_s": None,
        "error": None,
    }
    try:
        result = fetch_des_data(
            date=date_str,
            coords=COORDS,
            cloud_threshold=SCL_CLOUD_THRESHOLD,
            include_scl=True,
        )
        rec["scl_cloud_fraction"] = round(float(result.cloud_fraction), 4)
        rec["scl_pass"] = True
    except FetchError as e:
        msg = str(e)
        if "Scene too cloudy" in msg:
            # Parse the percentage from the error message
            try:
                pct_str = msg.split(":")[1].split("%")[0].strip()
                rec["scl_cloud_fraction"] = round(float(pct_str) / 100, 4)
            except Exception:
                pass
            rec["error"] = "scl_rejected"
        else:
            rec["error"] = msg[:200]
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["elapsed_s"] = round(time.time() - t0, 2)
    return rec


def cot_for_date(date_str: str, models, set_for_cache: str = "A_baseline") -> float | None:
    """Pull mean COT for a date from cached 11-band tile (A has all 36)."""
    npz = CACHE_ROOT / set_for_cache / f"{date_str}.npz"
    if not npz.exists():
        return None
    arr = np.load(npz)["arr"]
    bands = {b: arr[i] for i, b in enumerate(S2L2A_SPECTRAL_BANDS)}
    cot = cot_inference(bands, models)
    valid = np.isfinite(cot)
    return float(cot[valid].mean()) if valid.any() else None


def main() -> int:
    a_baseline = load_a_baseline()
    stac_cc = load_stac_cc()
    dates = [r["date"] for r in a_baseline]

    print(f"Probing {len(dates)} dates with cloud_threshold={SCL_CLOUD_THRESHOLD} "
          f"(prod default) and include_scl=True…")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(probe_one, d): d for d in dates}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            r["stac_cc"] = stac_cc.get(r["date"])
            results.append(r)
            tag = "PASS" if r["scl_pass"] else "REJ "
            cc = f"stac_cc={r['stac_cc']:.0f}%" if r["stac_cc"] is not None else "stac_cc=?"
            scl = (
                f"scl={r['scl_cloud_fraction']:.0%}"
                if r["scl_cloud_fraction"] is not None else "scl=?"
            )
            print(f"  [{i:>3}/{len(dates)}] {r['date']} {tag} {cc} {scl} ({r['elapsed_s']}s)")

    results.sort(key=lambda r: r["date"])
    with open(HERE / "c_full_scl_results.json", "w") as f:
        json.dump({
            "scl_cloud_threshold": SCL_CLOUD_THRESHOLD,
            "stac_cc_max": STAC_CC_MAX,
            "n_workers": N_WORKERS,
            "per_scene": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nLoading COT model ensemble for new-set aggregation…")
    models = _load_ensemble(DEFAULT_MODEL_PATHS, device="cpu")

    # Define the three sets we'll publish
    pass_dates = {r["date"] for r in results if r["scl_pass"]}
    stac_pass = {d for d, cc in stac_cc.items() if cc <= STAC_CC_MAX}

    # Original C — STAC only — kept for sanity
    c_stac_only = sorted(d for d in stac_pass if d in {r["date"] for r in a_baseline})
    # New C — STAC AND SCL (the actual pipeline)
    c_full = sorted(d for d in (stac_pass & pass_dates))

    out_metrics: dict[str, dict] = {}
    for label, dates_in in [
        ("C_stac_only", c_stac_only),
        ("C_full_scl", c_full),
    ]:
        cots = []
        for d in dates_in:
            v = cot_for_date(d, models)
            if v is not None:
                cots.append(v)
        out_metrics[label] = {
            "n_scenes": len(dates_in),
            "dates": dates_in,
            "mean_cot": round(float(np.mean(cots)), 5) if cots else None,
            "median_cot": round(float(np.median(cots)), 5) if cots else None,
            "std_cot": round(float(np.std(cots)), 5) if cots else None,
        }

    with open(HERE / "c_full_metrics.json", "w") as f:
        json.dump(out_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n=== C set comparison ===")
    print(f"  C_stac_only : n={out_metrics['C_stac_only']['n_scenes']:>2}  "
          f"mean_cot={out_metrics['C_stac_only']['mean_cot']}")
    print(f"  C_full_scl  : n={out_metrics['C_full_scl']['n_scenes']:>2}  "
          f"mean_cot={out_metrics['C_full_scl']['mean_cot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
