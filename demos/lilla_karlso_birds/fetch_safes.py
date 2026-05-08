"""Steg 1+2 av Lilla Karlsö-pipelinen: optimal_fetch + L1C SAFE-fetch.

Stage 1: ERA5 → SCL atmosfärsfilter via optimal_fetch_dates
Stage 2: L1C SAFE-arkiv från Google Cloud public bucket per utvalt datum

Output: SAFE-arkiv i config.SAFE_CACHE + plan.json med urvalsspår.

Användning:
    python -m demos.lilla_karlso_birds.fetch_safes
    # eller med override för att inte fetcha:
    python -m demos.lilla_karlso_birds.fetch_safes --plan-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from imint.config.env import load_env
load_env()

from imint.fetch import fetch_l1c_safe_from_gcp  # noqa: E402
from imint.training.optimal_fetch import optimal_fetch_dates  # noqa: E402

from demos.lilla_karlso_birds import config  # noqa: E402


def run_optimal_fetch() -> dict:
    """Kör ERA5→SCL pipeline över hela häckningssäsongen.

    Returns: serialiserbar plan-dict.
    """
    print(f"=== optimal_fetch_dates ===")
    print(f"AOI: {config.BBOX_WGS84}")
    print(f"period: {config.PERIOD_START} .. {config.PERIOD_END}")
    print(f"max_aoi_cloud: {config.MAX_AOI_CLOUD}")

    plan = optimal_fetch_dates(
        bbox_wgs84=config.BBOX_WGS84,
        date_start=config.PERIOD_START,
        date_end=config.PERIOD_END,
        mode="era5_then_scl",
        max_aoi_cloud=config.MAX_AOI_CLOUD,
    )
    print(f"\ncandidates_per_stage: {plan.n_candidates_after}")
    print(f"elapsed: {plan.elapsed_s}")
    print(f"FINAL: {len(plan.dates)} datum")
    for d in plan.dates:
        print(f"  - {d}")

    return {
        "mode": plan.mode,
        "dates": list(plan.dates),
        "n_candidates_after": plan.n_candidates_after,
        "elapsed_s": plan.elapsed_s,
        "notes": plan.notes,
        "aoi": config.BBOX_WGS84,
        "period": [config.PERIOD_START, config.PERIOD_END],
    }


def fetch_l1c_for_date(date: str) -> dict:
    """Hämta L1C SAFE-arkiv för ett datum från GCP public bucket.

    `cloud_max=100` (disable filter) eftersom optimal_fetch_dates redan har
    pre-filtrerat datumen med ERA5+SCL-stack på AOI-skala. STAC `eo:cloud_cover`
    är en granul-snitt-metrik (~110×110 km), meningslös för en 22 km AOI.
    UTM-zon-prefer auto-deriveras från AOI-centrum (Lilla Karlsö 17.925°E → 33).
    """
    t0 = time.time()
    try:
        safe_path = fetch_l1c_safe_from_gcp(
            date=date,
            coords=config.BBOX_WGS84,
            dest_dir=config.SAFE_CACHE,
            cloud_max=100.0,  # disable: pre-filtrerat upstream
            max_workers=config.N_WORKERS_FETCH,
        )
        n_files = sum(1 for _ in safe_path.rglob("*"))
        size_mb = sum(f.stat().st_size for f in safe_path.rglob("*") if f.is_file()) / 1024**2
        return {
            "date": date,
            "safe_path": str(safe_path.relative_to(config.SAFE_CACHE.parent.parent)),
            "n_files": n_files,
            "size_mb": round(size_mb, 1),
            "elapsed_s": round(time.time() - t0, 1),
            "success": True,
        }
    except Exception as e:
        return {
            "date": date,
            "elapsed_s": round(time.time() - t0, 1),
            "success": False,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Kör bara optimal_fetch och skriv plan.json. Skippa L1C-fetch.",
    )
    args = parser.parse_args()

    config.SAFE_CACHE.mkdir(parents=True, exist_ok=True)
    plan_path = config.SAFE_CACHE.parent / "plan.json"

    plan = run_optimal_fetch()
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    print(f"\nplan written: {plan_path}")

    if args.plan_only:
        print("\n--plan-only — skippar L1C-fetch")
        return 0

    if not plan["dates"]:
        print("0 dates — inget att fetcha")
        return 1

    print(f"\n=== L1C SAFE-fetch ({len(plan['dates'])} datum) ===")
    fetch_records = []
    for i, date in enumerate(plan["dates"], 1):
        print(f"\n[{i}/{len(plan['dates'])}] {date}")
        rec = fetch_l1c_for_date(date)
        fetch_records.append(rec)
        if rec["success"]:
            print(f"   ok — {rec['n_files']} files, {rec['size_mb']} MB, {rec['elapsed_s']}s")
        else:
            print(f"   ERR: {rec['error']}")

    # Skriv totalrapport
    fetch_summary = config.SAFE_CACHE.parent / "fetch_summary.json"
    fetch_summary.write_text(json.dumps({
        "plan": plan,
        "fetch_records": fetch_records,
        "n_success": sum(1 for r in fetch_records if r["success"]),
        "n_failed": sum(1 for r in fetch_records if not r["success"]),
        "total_size_mb": round(sum(r.get("size_mb", 0) for r in fetch_records), 1),
    }, indent=2, ensure_ascii=False))
    print(f"\nfetch_summary: {fetch_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
