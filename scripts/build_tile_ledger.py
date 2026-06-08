#!/usr/bin/env python3
"""Build a ledger (index) over all unified tiles across resolutions.

One row per tile (256 and 512), capturing what each tile *is* and *has* —
so cohort/densify-reason, coverage, duplicates, and "what's missing" are
queryable from data instead of inferred from filename prefixes. This is the
foundation for moving cohort out of the filename and into metadata.

Per-tile fields (cheap — uses has_* flags + temporal_mask + the small label
array; does NOT load full spectral):
  dataset, name, cohort, center_e, center_n, tile_px, year,
  frames_filled (temporal_mask sum), slot0_filled,
  has_b08, has_rededge, has_tessera, has_s1, has_frame_2016,
  n_classes, dominant_class, has_crop, has_wetland, has_urban,
  dates

Outputs:
  <out>.jsonl         — one JSON object per tile (the ledger)
  <out>.summary.json  — aggregates: counts by dataset x cohort, cross-res
                        coverage (256-only / 512-only / both by center),
                        same-center multi-tile groups (the "conflicts").

Usage (k8s, both PVCs, co-scheduled on the 256 RWO node):
  python scripts/build_tile_ledger.py \\
    --dirs 256=/data/unified_v2 512=/cephfs/unified_v2_512 \\
    --out /cephfs/audits/tile_ledger
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

# Unified-schema class groups (imint/training/unified_schema.py v5)
_CROP = set(range(11, 22))      # 11..21 LPIS crops
_WETLAND = {5, 7}              # sumpskog, våtmark
_URBAN = {9}                   # bebyggelse


def cohort_of(name: str) -> str:
    if name.startswith("tile_"):
        return "lulc"
    if name.startswith("crop_"):
        return "crop"
    if name.startswith("urban_"):
        return "urban"
    if name[:1].isdigit():
        return "numeric"
    return "other"


def center_of(d):
    if "easting" in d and "northing" in d:
        try:
            return int(d["easting"]), int(d["northing"])
        except Exception:
            pass
    if "bbox_3006" in d:
        w, s, e, n = [float(x) for x in np.asarray(d["bbox_3006"]).ravel()[:4]]
        return int(round((w + e) / 2)), int(round((s + n) / 2))
    return None, None


def _flag(d, key):
    if key not in d:
        return False
    try:
        return bool(np.asarray(d[key]).ravel()[0])
    except Exception:
        return False


def _year(d):
    for k in ("tessera_year", "lpis_year", "year"):
        if k in d:
            try:
                return int(d[k])
            except Exception:
                pass
    return None


def row_for(dataset, path):
    name = os.path.basename(path)[:-4]
    try:
        # Context-manage the NpzFile: np.load mmaps the zip, so not closing
        # it leaks a file handle + mapping per tile (OOMs over ~15k tiles).
        with np.load(path, allow_pickle=True) as d:
            return _extract(dataset, name, d)
    except Exception as e:
        return {"dataset": dataset, "name": name, "error": str(e)[:120]}


def _extract(dataset, name, d):
    e, n = center_of(d)
    tm = [int(x) for x in d["temporal_mask"]] if "temporal_mask" in d else []
    lbl_hist = {}
    n_classes = dominant = has_crop = has_wet = has_urb = None
    if "label" in d:
        lab = np.asarray(d["label"])
        vals, cnts = np.unique(lab, return_counts=True)
        lbl_hist = {int(v): int(c) for v, c in zip(vals, cnts)}
        present = set(lbl_hist) - {0}
        n_classes = len(present)
        dominant = int(vals[int(np.argmax(cnts))])
        has_crop = bool(present & _CROP)
        has_wet = bool(present & _WETLAND)
        has_urb = bool(present & _URBAN)
    return {
        "dataset": dataset, "name": name, "cohort": cohort_of(name),
        "center_e": e, "center_n": n,
        "tile_px": int(np.asarray(d["label"]).shape[-1]) if "label" in d else None,
        "year": _year(d),
        "frames_filled": int(sum(tm)), "slot0_filled": bool(tm[:1] == [1]),
        "has_b08": _flag(d, "has_b08"), "has_rededge": _flag(d, "has_rededge"),
        "has_tessera": _flag(d, "has_tessera"), "has_s1": _flag(d, "has_s1"),
        "has_frame_2016": _flag(d, "has_frame_2016"),
        "n_classes": n_classes, "dominant_class": dominant,
        "has_crop": has_crop, "has_wetland": has_wet, "has_urban": has_urb,
        "dates": [str(x) for x in d["dates"]] if "dates" in d else [],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dirs", nargs="+", required=True,
                   help="label=path pairs, e.g. 256=/data/unified_v2")
    p.add_argument("--out", required=True, help="output prefix (no extension)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    dirs = dict(s.split("=", 1) for s in args.dirs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    with open(args.out + ".jsonl", "w") as fh:
        for dataset, d in dirs.items():
            files = sorted(f for f in glob.glob(os.path.join(d, "*.npz"))
                           if not f.endswith("_tmp.npz"))
            if args.limit:
                files = files[: args.limit]
            print(f"{dataset}: {len(files)} tiles in {d}", flush=True)
            for i, f in enumerate(files):
                r = row_for(dataset, f)
                rows.append(r)
                fh.write(json.dumps(r) + "\n")
                if (i + 1) % 2000 == 0:
                    print(f"  {dataset} [{i+1}/{len(files)}]", flush=True)

    # ── summary ────────────────────────────────────────────────────────
    by_cohort = Counter((r["dataset"], r.get("cohort")) for r in rows if "error" not in r)
    # cross-resolution coverage keyed by center
    centers = defaultdict(set)
    multi = defaultdict(list)  # center -> [(dataset,name,cohort)] when >1 tile
    for r in rows:
        if "error" in r or r["center_e"] is None:
            continue
        c = (r["center_e"], r["center_n"])
        centers[c].add(r["dataset"])
        multi[c].append((r["dataset"], r["name"], r["cohort"]))
    only256 = sum(1 for c, ds in centers.items() if ds == {"256"})
    only512 = sum(1 for c, ds in centers.items() if ds == {"512"})
    both = sum(1 for c, ds in centers.items() if {"256", "512"} <= ds)
    # same-center, same-dataset, multiple tiles (the dup/conflict groups)
    samecell = {}
    for c, lst in multi.items():
        per = Counter(d for d, _, _ in lst)
        if any(v > 1 for v in per.values()):
            samecell[f"{c[0]}_{c[1]}"] = lst
    summary = {
        "n_rows": len(rows),
        "errors": sum(1 for r in rows if "error" in r),
        "by_dataset_cohort": {f"{k[0]}/{k[1]}": v for k, v in sorted(by_cohort.items())},
        "centers_only_256": only256,
        "centers_only_512": only512,
        "centers_both": both,
        "same_center_multi_tile_groups": len(samecell),
        "same_center_examples": dict(list(samecell.items())[:10]),
    }
    with open(args.out + ".summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n=== ledger summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}.jsonl + {args.out}.summary.json", flush=True)


if __name__ == "__main__":
    main()
