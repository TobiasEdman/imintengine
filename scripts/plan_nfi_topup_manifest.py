#!/usr/bin/env python3
"""Plan the step-3 top-up tiles for the multi-year NFI evaluation (read-only planner).

For each requested year this prespecifies, before any image exists, the
512 px tiles that lift the evaluation to ``--target`` eligible plot-years,
under the rules decided on 2026-10-11/12:

  * identity is (TractID, PlotID, Year); the teacher-pool plot-years of the
    task-0103 sizing (``imint.eval.fieldtruth`` at its process commit, 841
    plot-years) are excluded;
  * plot-years already on a kept evaluation-copy tile (in the 496 px common
    crop) count as covered and are not fetched again;
  * a plot inside the footprint of a TRAINING tile of the same year cannot
    get an evaluation tile (every tile containing it would overlap that
    footprint), and a candidate tile that would overlap a same-year training
    tile is dropped; other-year overlap is allowed (D6);
  * candidates are ranked by the sizing's identity hash (sha256 of
    ``seed:TractID:PlotID:Year``) so the choice is prespecified and
    independent of field class or any model output; tiles are taken in rank
    order of their best plot until the gap is closed;
  * placement is the 2024 campaign's: a 5,120 m lattice cell decides which
    plots share a tile, the tile is centred on their bounding box and
    snapped to the 10 m grid, and a group that does not fit the 496 px crop
    is split recursively (the sizing's ``tile_cover`` rule).

Outputs, per year: a manifest in the ``nfi2024_512.json`` shape (tile name,
bbox_3006, year, source, n_plots), a WEkEO VPP gap report in the
``nfi2024_vpp_gaps.json`` shape, and aggregate counts. No plot identity or
coordinate is printed; the manifest carries tile bboxes and counts only.
Nothing under the data root is read for writing; the two output files are the
only writes and they go where ``--out-dir`` says.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from imint.training.tile_config import TileConfig  # noqa: E402
from scripts.census_holdout_val_512 import (  # noqa: E402
    CROP_PX, PAD, build_index, exposure_union, load_fieldtruth, read_tile,
    require_cohort_geometry, valid_rows,
)

TILE = TileConfig(size_px=512)
SEED = 20261008  # the sizing's identity-hash seed (NFI_YEAR_SAMPLING)
SOURCE = "lulc"  # forest truth; the year key forces strict matching downstream


def rank_key(tract: int, plot: int, year: int, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}:{tract}:{plot}:{year}".encode("ascii")).hexdigest()


def crop_mask(frame: pd.DataFrame, bbox: dict) -> np.ndarray:
    rows = np.floor((bbox["north"] - frame["Northing"].to_numpy()) / TILE.gsd_m)
    cols = np.floor((frame["Easting"].to_numpy() - bbox["west"]) / TILE.gsd_m)
    return (rows >= PAD) & (rows < PAD + CROP_PX) & (cols >= PAD) & (cols < PAD + CROP_PX)


def centred_bbox(group: pd.DataFrame) -> dict:
    return TILE.bbox_from_center((group["Easting"].min() + group["Easting"].max()) / 2,
                                 (group["Northing"].min() + group["Northing"].max()) / 2)


def overlaps(bbox: dict, geo: pd.DataFrame) -> bool:
    if geo.empty:
        return False
    return bool(((geo["west"] < bbox["east"]) & (geo["east"] > bbox["west"])
                 & (geo["south"] < bbox["north"]) & (geo["north"] > bbox["south"])).any())


def inside_any(frame: pd.DataFrame, geo: pd.DataFrame) -> np.ndarray:
    """Plots that lie inside some tile of ``geo`` (inclusive edges, like plots_in_bbox)."""
    out = np.zeros(len(frame), dtype=bool)
    if geo.empty or frame.empty:
        return out
    e, n = frame["Easting"].to_numpy(), frame["Northing"].to_numpy()
    for w, s, ea, no in geo[["west", "south", "east", "north"]].to_numpy(float):
        out |= (e >= w) & (e <= ea) & (n >= s) & (n <= no)
    return out


def place_tiles(candidates: pd.DataFrame, same_year_training: pd.DataFrame, key_cols: list[str]) -> list[dict]:
    """Centred-lattice placement with crop splits; drops tiles on a same-year training footprint.

    Returns tile records sorted by the rank of their best plot. Each record
    keeps its plot keys in ``_keys`` for the caller's bookkeeping; they never
    leave this process.
    """
    if candidates.empty:
        return []
    groups = candidates.assign(_x=np.floor(candidates["Easting"] / TILE.size_m).astype("int64"),
                               _y=np.floor(candidates["Northing"] / TILE.size_m).astype("int64"))
    pending = [g for _, g in groups.groupby(["_x", "_y"], sort=True)]
    tiles = []
    while pending:
        group = pending.pop()
        bbox = centred_bbox(group)
        TILE.assert_bbox_matches(bbox)
        if not crop_mask(group, bbox).all():
            assert len(group) > 1, "a centred singleton must lie in every native crop"
            axis = max(("Easting", "Northing"), key=lambda k: group[k].max() - group[k].min())
            ordered = group.sort_values([axis, *key_cols])
            mid = len(group) // 2
            pending.extend([ordered.iloc[:mid], ordered.iloc[mid:]])
            continue
        cell_w, cell_s = int(group["_x"].iloc[0]) * TILE.size_m, int(group["_y"].iloc[0]) * TILE.size_m
        tiles.append({
            "name": f"nfi{int(group['Year'].iloc[0])}_{bbox['west']}_{bbox['south']}",
            "cell": [cell_w, cell_s],
            "bbox_3006": bbox,
            "year": int(group["Year"].iloc[0]),
            "source": SOURCE,
            "n_plots": int(len(group)),
            "_rank": min(group["_rank"]),
            "_keys": [tuple(int(v) for v in row) for row in group[key_cols].to_numpy()],
            "_same_year_training_overlap": overlaps(bbox, same_year_training),
        })
    return sorted(tiles, key=lambda t: (t["_rank"], t["name"]))


def plan_year(year: int, plots: pd.DataFrame, ft, exposure, covered_keys, cohort: pd.DataFrame,
              target: int, covered_count: int) -> tuple[list[dict], dict]:
    key_cols = list(ft.NFI_KEY)
    p = plots[plots["Year"] == year].copy()
    valid = valid_rows(p)
    p = p.loc[valid].copy()
    keys = ft.observation_keys(p)
    exposed = keys.isin(exposure)
    covered = keys.isin(covered_keys)
    same_year = cohort[(cohort["year"] == year)].dropna(subset=["west"]) if len(cohort) else cohort
    on_training = inside_any(p, same_year)
    candidates = p.loc[~exposed & ~covered & ~on_training].copy()
    candidates["_rank"] = [rank_key(t, pl, y) for t, pl, y in ft.observation_keys(candidates)]
    need = max(0, target - covered_count)
    tiles = place_tiles(candidates, same_year, key_cols)
    dropped_overlap = [t for t in tiles if t["_same_year_training_overlap"]]
    feasible = [t for t in tiles if not t["_same_year_training_overlap"]]
    chosen, gained = [], 0
    for t in feasible:
        if gained >= need:
            break
        chosen.append(t)
        gained += t["n_plots"]
    summary = {
        "year": year,
        "target": target,
        "covered_on_evaluation_copy": covered_count,
        "need": need,
        "plot_years_valid": int(len(p)),
        "invalid_values": int((~valid).sum()),
        "excluded_teacher_pool": int(exposed.sum()),
        "already_covered": int(covered.sum()),
        "inside_same_year_training_footprint": int((~exposed & ~covered & on_training).sum()),
        "candidates": int(len(candidates)),
        "candidate_tiles": len(tiles),
        "candidate_tiles_dropped_same_year_overlap": len(dropped_overlap),
        "plots_on_dropped_tiles": int(sum(t["n_plots"] for t in dropped_overlap)),
        "feasible_tiles": len(feasible),
        "feasible_plot_years": int(sum(t["n_plots"] for t in feasible)),
        "chosen_tiles": len(chosen),
        "chosen_plot_years": gained,
        "gap_after_plan": max(0, need - gained),
        "crop_splits": sum(1 for _ in tiles) - len({tuple(t["cell"]) for t in tiles}),
    }
    return chosen, summary


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("/data"))
    ap.add_argument("--eval-dir", type=Path, default=None, help="default <data-root>/holdout_eval_512")
    ap.add_argument("--cohort-dir", type=Path, default=None, help="default <data-root>/unified_v2_512")
    ap.add_argument("--fieldtruth-file", type=Path, default=REPO / "imint/eval/fieldtruth.py")
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--vpp-cache", type=Path, default=None, help="default <data-root>/vpp_wekeo")
    ap.add_argument("--out-dir", type=Path, required=True, help="where nfi{Y}_topup_512.json and gaps go")
    ap.add_argument("--summary-json", type=Path, required=True)
    args = ap.parse_args(argv)
    root = args.data_root
    eval_dir = args.eval_dir or root / "holdout_eval_512"
    cohort_dir = args.cohort_dir or root / "unified_v2_512"
    vpp_cache = args.vpp_cache or root / "vpp_wekeo"
    t0 = time.time()

    # Fail before any input is read or any output written: without a complete
    # training-cohort inventory the footprint rule would silently exclude nothing.
    cohort = require_cohort_geometry(cohort_dir)

    ft = load_fieldtruth(args.fieldtruth_file)
    from imint.training.slu_nfi import load_nfi_plots
    plots = load_nfi_plots(data_dir=root / "nfi", years=args.years)
    inputs: dict = {}
    exposure = exposure_union(ft, root, inputs)
    print(f"teacher union: {len(exposure)} plot-years", flush=True)

    # what the evaluation copy already covers, with the census's own colocation
    tiles = [read_tile(p, None) for p in sorted(eval_dir.glob("*.npz")) if not p.name.endswith(".tmp.npz")]
    index = build_index(tiles, plots)
    if len(index):
        in_crop = index["row"].between(PAD, PAD + CROP_PX - 1) & index["col"].between(PAD, PAD + CROP_PX - 1)
        idx = index.loc[in_crop]
        idx = idx.loc[~ft.observation_keys(idx).isin(exposure) & valid_rows(idx)]
        covered_keys = ft.observation_keys(idx).unique()
        covered_by_year = {int(y): int(len(ft.observation_keys(g).unique())) for y, g in idx.groupby("Year")}
    else:
        covered_keys, covered_by_year = ft.observation_keys(plots.iloc[0:0]), {}
    print(f"evaluation copy: {len(tiles)} tiles, covered eligible plot-years {covered_by_year}", flush=True)

    from derive_missing_vpp_mgrs import cached_pairs, mgrs_tiles_for_bbox
    have = cached_pairs(str(vpp_cache)) if vpp_cache.is_dir() else set()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries, requests = [], 0
    for year in sorted(set(args.years)):
        chosen, summary = plan_year(year, plots, ft, exposure, covered_keys, cohort, args.target,
                                    covered_by_year.get(year, 0))
        manifest = [{k: v for k, v in t.items() if not k.startswith("_")} for t in chosen]
        manifest.sort(key=lambda t: -t["n_plots"])
        need_mgrs = set()
        for t in manifest:
            need_mgrs |= mgrs_tiles_for_bbox(t["bbox_3006"])
        missing = sorted(m for m in need_mgrs if (m, year) not in have)
        summary["mgrs_cells"] = len(need_mgrs)
        summary["mgrs_missing_in_wekeo_cache"] = len(missing)
        summary["wekeo_requests_estimate"] = 5 * len(missing)  # five metrics per (cell, year)
        requests += 5 * len(missing)
        (args.out_dir / f"nfi{year}_topup_512.json").write_text(json.dumps(manifest, indent=1))
        (args.out_dir / f"nfi{year}_topup_vpp_gaps.json").write_text(json.dumps(
            {"year": year, "needed": sorted(need_mgrs), "missing": missing}, indent=1))
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    out = {"schema": "nfi-topup-plan-v1", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "elapsed_s": round(time.time() - t0, 1), "seed": SEED, "target": args.target,
           "identity": list(ft.NFI_KEY), "common_crop_px": CROP_PX, "teacher_feature_union": int(len(exposure)),
           "inputs": inputs, "evaluation_copy_tiles": len(tiles), "by_year": summaries,
           "wekeo_requests_estimate_total": requests, "wekeo_quota_per_window": 500}
    args.summary_json.write_text(json.dumps(out, indent=1))
    print(f"\nWEkEO requests for the whole plan: ~{requests} (quota 500 per window)")
    return out


if __name__ == "__main__":
    main()
