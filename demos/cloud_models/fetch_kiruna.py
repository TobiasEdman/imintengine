"""
demos/cloud_models/fetch_kiruna.py

Fetches a curated mix of clear, partly-cloudy and cloudy Sentinel-2 L2A
scenes over Kiruna for the cloud-model comparison. Uses the SCL stack to
sort candidate dates by AOI cloud fraction, then picks 4 scenes spanning
the full clarity range — that's what stress-tests the models.

Per scene we fetch the 11-band BOA spectral set + SCL + CLD (Sen2Cor's
cloud-probability layer) in a single openEO call. The auxiliary bands are
saved as separate npz keys so the comparison driver can run all five
models against the exact same pixels.

Run:
    python demos/cloud_models/fetch_kiruna.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import zipfile
import gzip
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from imint.config.env import load_env
load_env()

from imint.fetch import S2L2A_SPECTRAL_BANDS, fetch_des_data  # noqa: E402
from imint.training.optimal_fetch import (  # noqa: E402
    _connect_des_openeo, scl_stack_screen,
)

# ── AOI / period ───────────────────────────────────────────────────────────
AOI_NAME = "Kiruna"
BBOX = {
    # Centred ~67.85°N 20.22°E (Kiruna town + Kirunavaara mine).
    # ~12 km × 8 km — covers town, mine, surrounding tundra.
    "west":  20.10, "south": 67.81,
    "east":  20.36, "north": 67.89,
}
PERIOD_START = "2024-04-01"
PERIOD_END   = "2024-09-30"
N_SCENES_TO_FETCH = 4   # one near each quartile of AOI cloud-fraction
N_WORKERS = 6

# ── Output paths ───────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
SPEC_CACHE = HERE / "cache_kiruna"
SPEC_CACHE.mkdir(parents=True, exist_ok=True)

# Lower-case band names DES openEO expects
DES_BANDS = ["b02", "b03", "b04", "b05", "b06", "b07", "b08",
             "b8a", "b09", "b11", "b12", "scl", "cld"]


def fetch_scene(date_str: str, conn) -> dict:
    """Fetch 11 BOA bands + SCL via existing fetch_des_data. Sen2Cor's
    SCL band gives us all the per-pixel cloud-class info we need to
    compare against the other detectors — the separate CLD probability
    band would just be a continuous version of the same Sen2Cor model,
    and DES openEO refuses single-band CLD requests anyway.
    """
    cache = SPEC_CACHE / f"{date_str}.npz"
    if cache.exists():
        return {"date": date_str, "from_cache": True, "elapsed_s": 0.0}

    t0 = time.time()
    fr = fetch_des_data(
        date=date_str, coords=BBOX,
        cloud_threshold=1.0,    # don't reject; cloudy scenes are the point
        include_scl=True,
        date_window=0,
    )
    spectral = np.stack(
        [fr.bands[b] for b in S2L2A_SPECTRAL_BANDS], axis=0,
    ).astype(np.float32)
    scl = fr.scl.astype(np.uint8)

    np.savez_compressed(cache, arr=spectral, scl=scl)
    return {
        "date": date_str,
        "from_cache": False,
        "elapsed_s": round(time.time() - t0, 2),
        "shape": list(spectral.shape),
    }


def main() -> int:
    print(f"AOI: {AOI_NAME}  bbox={BBOX}")
    print(f"Period: {PERIOD_START} → {PERIOD_END}")

    # ── Step 1: SCL-stack to find cloud-fraction-per-date ───────────────
    print("\n[1/3] SCL-stack screen för hela perioden …")
    fracs = scl_stack_screen(BBOX, PERIOD_START, PERIOD_END)
    print(f"  {len(fracs)} datum med data; AOI-cloud-fraction:")
    sorted_dates = sorted(fracs.items(), key=lambda x: x[1])
    for d, f in sorted_dates[:5]:
        print(f"    {d}  {f:.0%}  (klar)")
    for d, f in sorted_dates[-3:]:
        print(f"    {d}  {f:.0%}  (molnig)")

    # ── Step 2: Curate 4 scenes spanning quartiles ──────────────────────
    if len(sorted_dates) < N_SCENES_TO_FETCH:
        picked = sorted_dates
    else:
        # Pick at quartiles: clearest, ~25 %, ~50 %, ~75 %
        idxs = [
            0,
            len(sorted_dates) // 4,
            len(sorted_dates) // 2,
            3 * len(sorted_dates) // 4,
        ]
        picked = [sorted_dates[i] for i in idxs]
    print(f"\n[2/3] Valda {len(picked)} scener över hela molnskalan:")
    for d, f in picked:
        print(f"    {d}  AOI-cloud {f:.0%}")

    # ── Step 3: Fetch ───────────────────────────────────────────────────
    print(f"\n[3/3] Hämtar full spektral + SCL + CLD per scen "
          f"({N_WORKERS} workers) …")
    conn = _connect_des_openeo()
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(fetch_scene, d, conn): d for d, _ in picked}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = "cache" if r["from_cache"] else "OK"
            print(f"  {r['date']}  {tag:<5}  ({r['elapsed_s']}s)")
    wall = time.time() - t0

    manifest = {
        "aoi":       AOI_NAME,
        "bbox":      BBOX,
        "period":    [PERIOD_START, PERIOD_END],
        "scl_fracs": {d: round(f, 4) for d, f in picked},
        "fetches":   results,
        "fetch_wall_s": round(wall, 2),
    }
    with open(SPEC_CACHE / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Klart på {wall:.1f}s. Manifest: {SPEC_CACHE / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
