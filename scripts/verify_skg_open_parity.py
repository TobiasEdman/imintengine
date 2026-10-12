#!/usr/bin/env python3
"""Compare the open-data mirror with proxy-fetched cache arrays (IM-022).

Every ``.npy`` the retired fetchers cached under ``<root>/cache/<channel>``
is a 512×512 (or 256×256) array on a known bbox. Reading the same bbox
through the mirror-backed fetchers and comparing shows, per channel, how
far new tiles will be from stored ones. Grunddata should be exact;
markfukt and height are different product versions (see
docs/data/skg_open_mirror.md).

Usage::

    python scripts/verify_skg_open_parity.py --roots /data/unified_v2_512 \\
        /data/unified_v2_512_nfi2024_staging --sample 200 --out report.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.skg_grunddata import (  # noqa: E402
    fetch_basal_area_tile, fetch_diameter_tile, fetch_tree_height_tile, fetch_volume_tile,
)
from imint.training.skg_open import SkgMirrorMissingError  # noqa: E402
from imint.training.slu_markfukt import fetch_markfukt_tile  # noqa: E402

# cache dir -> (file prefix, fetcher, reference unit note)
CHANNELS = {
    "volume": ("volume", fetch_volume_tile),
    "basal_area": ("basal_area", fetch_basal_area_tile),
    "diameter": ("diameter", fetch_diameter_tile),
    "markfukt": ("markfukt", fetch_markfukt_tile),
    "height": ("tree_height", lambda *a, **k: fetch_tree_height_tile(*a, **k) * 10.0),  # cache holds dm
}
_KEY = re.compile(r"_(\d+)_(\d+)_(\d+)_(\d+)\.npy$")


def compare(ref: np.ndarray, cand: np.ndarray) -> dict:
    ref = ref.astype(np.float64)
    cand = cand.astype(np.float64)
    both = (ref > 0) & (cand > 0)
    d = ref - cand
    return {
        "exact": round(float((ref == cand).mean()), 4),
        "exact_where_both_valid": round(float((ref[both] == cand[both]).mean()), 4) if both.any() else None,
        "mae": round(float(np.abs(d).mean()), 3),
        "bias": round(float(d.mean()), 3),
        "ref_nonzero": round(float((ref > 0).mean()), 4),
        "cand_nonzero": round(float((cand > 0).mean()), 4),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", type=Path, required=True)
    p.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=list(CHANNELS))
    p.add_argument("--sample", type=int, default=100, help="arrays per root and channel (random, seeded)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rng = random.Random(args.seed)

    report: dict = {"roots": [str(r) for r in args.roots], "sample": args.sample, "per_root": {}}
    total_rows = total_errors = 0
    for root in args.roots:
        for ch in args.channels:
            prefix, fetch = CHANNELS[ch]
            files = sorted((root / "cache" / ch).glob(f"{prefix}_*.npy"))
            if not files:
                # explicit: this root has no reference arrays for the channel
                report["per_root"].setdefault(str(root), {})[ch] = {
                    "summary": {"n": 0, "errors": 0, "n_available": 0}, "tiles": []}
                print(f"{root.name:40s} {ch:11s} no reference arrays", flush=True)
                continue
            picks = rng.sample(files, min(args.sample, len(files)))
            rows, errors = [], 0
            for f in picks:
                m = _KEY.search(f.name)
                w, s, e, n = map(int, m.groups())
                ref = np.load(f)
                try:
                    cand = fetch(w, s, e, n, size_px=ref.shape)
                except SkgMirrorMissingError:
                    raise
                except Exception as ex:          # noqa: BLE001 — report, keep going
                    errors += 1
                    print(f"  {ch} {f.name}: {type(ex).__name__}: {ex}", flush=True)
                    continue
                if ch == "markfukt":
                    ref = np.where(ref > 101, 0, ref)
                    cand = np.where(cand > 101, 0, cand)
                rows.append({"bbox": [w, s, e, n], **compare(ref, cand)})
            total_rows += len(rows)
            total_errors += errors
            agg = {"n": len(rows), "errors": errors, "n_available": len(files)}
            if rows:
                agg.update({k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 4)
                            for k in ("exact", "mae", "bias", "ref_nonzero", "cand_nonzero")})
                agg["exact_min"] = round(min(r["exact"] for r in rows), 4)
            report["per_root"].setdefault(str(root), {})[ch] = {"summary": agg, "tiles": rows}
            if rows:
                print(f"{root.name:40s} {ch:11s} n={len(rows):4d} exact={agg['exact']:.4f} "
                      f"(min {agg['exact_min']:.4f}) mae={agg['mae']:.3f} bias={agg['bias']:+.3f} "
                      f"nonzero ref/cand {agg['ref_nonzero']:.3f}/{agg['cand_nonzero']:.3f} errors={errors}",
                      flush=True)
            else:
                print(f"{root.name:40s} {ch:11s} n=0 errors={errors} — every read failed", flush=True)
    report["summary"] = {"compared": total_rows, "errors": total_errors}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}: {total_rows} compared, {total_errors} errors")
    if total_errors or not total_rows:
        print("FAIL: parity did not validate every sample", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
