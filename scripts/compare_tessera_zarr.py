#!/usr/bin/env python3
"""Measure how far Zarr-read TESSERA windows are from the cohort's stored ones.

The cohort's ``tessera`` came from geotessera 0.10.2 NPY tiles (dataset v1
"vultr") through ``enrich_tiles_tessera._fetch_tessera_for_tile``, for the
bbox ``resolve_tile_bbox`` gives and the tile's ``tessera_year``. This reads
the same bbox and year from the Zarr store with the same resampling and merge
(``merge="training"``), so what differs is the store itself: tile origins
rounded to the zone lattice and last-written-wins overlaps.

The sample is deterministic: tiles in sha256(name) order whose
``tessera_source`` is ``geotessera-0.10.2``. Per tile it reports validity
agreement and, on pixels both call valid, the share of bit-identical vectors,
absolute differences and cosine-similarity percentiles.

    python scripts/compare_tessera_zarr.py --data-dir /cephfs/unified_v2_512 \\
        --n 20 --cache-dir /cephfs/tessera_zarr_cache --report report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from imint.inference import lclu_zarr  # noqa: E402
from imint.inference.lclu import InputUnavailable  # noqa: E402

COHORT_SOURCE = "geotessera-0.10.2"
LOCATION_KEYS = ("easting", "northing", "bbox_3006")


def sample(data_dir: Path, n: int) -> list[Path]:
    """The first ``n`` tiles in sha256(name) order with a geotessera-0.10.2 embedding."""
    chosen = []
    for path in sorted(data_dir.glob("*.npz"),
                       key=lambda p: hashlib.sha256(p.stem.encode()).hexdigest()):
        try:
            with np.load(path, allow_pickle=False) as z:
                if ("tessera" in z.files and "has_tessera" in z.files
                        and int(z["has_tessera"]) == 1
                        and str(z["tessera_source"]) == COHORT_SOURCE):
                    chosen.append(path)
        except (KeyError, OSError, ValueError):
            continue
        if len(chosen) == n:
            break
    return chosen


def compare_arrays(stored: np.ndarray, zarr_emb: np.ndarray, zarr_valid: np.ndarray) -> dict:
    """Validity agreement and, on pixels both call valid, how close the vectors are."""
    stored_valid = np.any(stored != 0, axis=0)
    both = stored_valid & zarr_valid
    row = {"stored_valid_px": int(stored_valid.sum()), "zarr_valid_px": int(zarr_valid.sum()),
           "both_valid_px": int(both.sum()),
           "stored_only_px": int((stored_valid & ~zarr_valid).sum()),
           "zarr_only_px": int((zarr_valid & ~stored_valid).sum())}
    if not both.any():
        return row
    a = stored[:, both].astype(np.float32).T
    b = zarr_emb[:, both].astype(np.float32).T
    diff = np.abs(a - b)
    cosine = np.sum(a * b, axis=1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    p01, p05, p50 = np.percentile(cosine, [1, 5, 50])
    return {**row, "identical_px": int(np.all(a == b, axis=1).sum()),
            "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "cosine_p01": float(p01), "cosine_p05": float(p05), "cosine_p50": float(p50)}


def compare_tile(path: Path, store_url: str, cache_dir: Path) -> dict:
    from imint.training.tile_bbox import resolve_tile_bbox
    from imint.training.tile_config import TileConfig

    with np.load(path, allow_pickle=False) as z:
        stored = z["tessera"]
        year = int(z["tessera_year"])
        location = {k: z[k] for k in LOCATION_KEYS if k in z.files}
    size = stored.shape[-1]
    row = {"tile": path.stem, "year": year}
    bbox = resolve_tile_bbox(name=path.stem, tile=TileConfig(size_px=size), npz_data=location)
    if bbox is None:
        return {**row, "status": "failed", "reason": "no_bbox"}
    bounds = (bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    try:
        emb, valid, provenance = lclu_zarr.read_embedding(
            store_url, bounds, size, year, cache_dir=cache_dir, merge="training")
    except InputUnavailable as exc:
        return {**row, "status": "unavailable", "reason": str(exc)}
    return {**row, "status": "ok", "zones": [z["zone"] for z in provenance["zones"]],
            **compare_arrays(stored, emb, valid)}


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    compared = [r for r in ok if r["both_valid_px"]]
    total = sum(r["both_valid_px"] for r in compared)
    union = sum(r["both_valid_px"] + r["stored_only_px"] + r["zarr_only_px"] for r in ok)
    summary = {"tiles": len(rows), "ok": len(ok),
               "unavailable": sum(r["status"] == "unavailable" for r in rows),
               "failed": sum(r["status"] == "failed" for r in rows),
               "valid_agreement": (sum(r["both_valid_px"] for r in ok) / union) if union else None}
    if compared:
        summary.update(
            identical_fraction=sum(r["identical_px"] for r in compared) / total,
            max_abs_diff=max(r["max_abs_diff"] for r in compared),
            mean_abs_diff=sum(r["mean_abs_diff"] * r["both_valid_px"] for r in compared) / total,
            worst_cosine_p01=min(r["cosine_p01"] for r in compared))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--store", default=lclu_zarr.STORE_URL)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args(argv)

    tiles = sample(args.data_dir, args.n)
    if not tiles:
        print(f"no tiles with a {COHORT_SOURCE} embedding in {args.data_dir}")
        return 1
    rows = []
    for path in tiles:
        rows.append(compare_tile(path, args.store, args.cache_dir))
        print(json.dumps(rows[-1]), flush=True)
    import zarr
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True).stdout.strip() or None
    report = {"store": args.store, "embedding": lclu_zarr.ZARR_EMBEDDING,
              "cohort_source": COHORT_SOURCE, "merge": "training", "git_sha": git_sha,
              "zarr": zarr.__version__, "summary": summarise(rows), "tiles": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
