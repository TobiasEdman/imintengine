#!/usr/bin/env python3
"""Scene selector for the sen2cor frame_2016 back-fill pipeline.

Architecture (decided 2026-05-16)
---------------------------------
COT cloud scoring needs 12-band TOA pixels, and every candidate L1C
scene must be downloaded as a SAFE archive to run sen2cor anyway. So
COT is NOT a pre-selection fetch — it moves into the *runner*
(run_sen2cor_per_scene.py), which reads the SAFE it already has and
gates each tile on cot_l1c before paying sen2cor compute.

This selector therefore does the cheap, pixel-free part:

  1. ERA5 atmosphere prefilter per tile (Open-Meteo, free) →
     candidate ISO dates where weather is plausibly clear.
  2. STAC L1C catalogue lookup — *existence only*, NO eo:cloud_cover
     filter (COT replaces that). Maps each candidate (tile, date) to
     the L1C scene id + MGRS granule covering it.
  3. Greedy set-cover on COVERAGE: pick the scene covering the most
     still-unassigned tiles, repeat. ERA5-date order breaks ties so
     weather-better dates win.
  4. For every tile, also emit a ranked fallback list of OTHER scenes
     that cover it — the runner uses this when COT rejects a tile's
     primary scene as too cloudy.

Output JSON (consumed by run_sen2cor_per_scene.py)::

    {
      "year_primary": 2016, "year_fallback": 2015,
      "scenes": [
        {"scene_id": "...", "mgrs_tile": "33VUD",
         "datetime": "2016-08-21T...", "tile_names": ["tile_...", ...]}
      ],
      "tile_fallbacks": {
        "tile_385060_6345060": ["S2A_MSIL1C_2016...", "S2B_..."]
      },
      "unassigned": ["tile_..."]
    }

Usage:
    python scripts/sen2cor_pipeline/select_scenes.py \\
        --data-dir /data/unified_v2_512 \\
        --year 2016 --fallback-year 2015 \\
        --out /data/debug/sen2cor_plan_2016.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_CDSE_STAC_ROOT = "https://stac.dataspace.copernicus.eu/v1"
_L1C_COLLECTION = "sentinel-2-l1c"


# ── STAC catalogue lookup (existence only, no cloud filter) ──────────────

def _stac_l1c_scenes(
    bbox_wgs84: dict,
    date_start: str,
    date_end: str,
) -> list[dict]:
    """All L1C scenes intersecting bbox in the window. No cloud filter.

    cloud_pct is still captured (free, comes with the item) and used
    only as a tie-breaker in set-cover — the real cloud gate is COT in
    the runner.
    """
    from pystac_client import Client

    client = Client.open(_CDSE_STAC_ROOT)
    search = client.search(
        collections=[_L1C_COLLECTION],
        bbox=[bbox_wgs84["west"], bbox_wgs84["south"],
              bbox_wgs84["east"], bbox_wgs84["north"]],
        datetime=f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
        limit=300,
    )
    out: list[dict] = []
    for item in search.items():
        props = item.properties or {}
        mgrs = props.get("s2:mgrs_tile") or props.get("mgrs_tile") or ""
        if not mgrs:
            seg = item.id.split("_T")
            mgrs = seg[-1].split("_")[0] if len(seg) > 1 else ""
        out.append({
            "scene_id": item.id,
            "datetime": props.get("datetime") or props.get("start_datetime"),
            "cloud_pct": float(props.get("eo:cloud_cover", 100.0)),
            "mgrs_tile": mgrs,
        })
    return out


# ── ERA5 prefilter ───────────────────────────────────────────────────────

def _era5_dates(bbox_wgs84: dict, year: int) -> set[str]:
    """ISO summer dates passing the atmosphere rules for this bbox."""
    from imint.training.optimal_fetch import era5_prefilter_dates
    return set(era5_prefilter_dates(
        bbox_wgs84, f"{year}-06-01", f"{year}-08-31",
    ))


# ── Tile inventory ───────────────────────────────────────────────────────

def _missing_frame_2016_tiles(data_dir: str) -> list[tuple[str, dict]]:
    """(tile_name, wgs84_bbox) for tiles lacking has_frame_2016==1."""
    from imint.training.tile_bbox import resolve_tile_bbox
    from imint.training.tile_config import TileConfig
    from pyproj import Transformer

    tx = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)
    missing: list[tuple[str, dict]] = []
    for npz_path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        name = Path(npz_path).stem
        try:
            with np.load(npz_path, allow_pickle=True) as d:
                if "frame_2016" in d.files and int(d.get("has_frame_2016", 0)) == 1:
                    continue
                cfg = TileConfig(size_px=int(d.get("tile_size_px", 512)))
                bbox3006 = resolve_tile_bbox(name=name, tile=cfg, npz_data=d)
        except Exception:
            continue
        if bbox3006 is None:
            continue
        w, s = tx.transform(bbox3006["west"], bbox3006["south"])
        e, n = tx.transform(bbox3006["east"], bbox3006["north"])
        missing.append((name, {"west": w, "south": s, "east": e, "north": n}))
    return missing


# ── Greedy set-cover ─────────────────────────────────────────────────────

def _set_cover(
    tile_to_scenes: dict[str, list[dict]],
) -> tuple[list[dict], dict[str, list[str]], list[str]]:
    """Greedy coverage set-cover.

    Returns:
        selected   — scene dicts, each with a ``tile_names`` list
        fallbacks  — {tile_name: [other scene_ids covering it, ranked
                     by cloud_pct ascending]}
        unassigned — tiles no scene covers
    """
    scene_to_tiles: dict[str, set[str]] = defaultdict(set)
    scene_meta: dict[str, dict] = {}
    for tile, scenes in tile_to_scenes.items():
        for sc in scenes:
            scene_to_tiles[sc["scene_id"]].add(tile)
            scene_meta[sc["scene_id"]] = sc

    uncovered = set(tile_to_scenes.keys())
    selected: list[dict] = []
    while uncovered:
        best_id, best_score, best_set = None, -1.0, set()
        for sid, tiles in scene_to_tiles.items():
            covered = tiles & uncovered
            if not covered:
                continue
            # Coverage count, cloud_pct only as a sub-integer tiebreak
            score = len(covered) - 0.001 * scene_meta[sid]["cloud_pct"]
            if score > best_score:
                best_score, best_id, best_set = score, sid, covered
        if best_id is None:
            break
        selected.append({**scene_meta[best_id], "tile_names": sorted(best_set)})
        uncovered -= best_set

    # Fallback lists: every scene covering a tile, minus its primary,
    # ranked by cloud_pct ascending.
    primary_of: dict[str, str] = {}
    for sc in selected:
        for t in sc["tile_names"]:
            primary_of[t] = sc["scene_id"]
    fallbacks: dict[str, list[str]] = {}
    for tile, scenes in tile_to_scenes.items():
        ranked = sorted(scenes, key=lambda s: s["cloud_pct"])
        others = [s["scene_id"] for s in ranked
                  if s["scene_id"] != primary_of.get(tile)]
        if others:
            fallbacks[tile] = others

    return selected, fallbacks, sorted(uncovered)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="sen2cor scene selector (ERA5 + STAC catalogue)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--year", type=int, default=2016)
    p.add_argument("--fallback-year", type=int, default=2015)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tiles", type=int, default=None)
    args = p.parse_args()

    t0 = time.time()
    print("=== sen2cor scene selector ===")
    print(f"  data-dir: {args.data_dir}")
    print(f"  years:    {args.year} (fallback {args.fallback_year})")

    tiles = _missing_frame_2016_tiles(args.data_dir)
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    print(f"  tiles missing frame_2016: {len(tiles)}")

    plan_scenes: list[dict] = []
    all_fallbacks: dict[str, list[str]] = {}

    for pass_idx, year in enumerate([args.year, args.fallback_year]):
        if not tiles:
            break
        print(f"\n--- Pass {pass_idx + 1}: year={year}, remaining={len(tiles)} ---")
        tile_to_scenes: dict[str, list[dict]] = {}
        for i, (name, bbox) in enumerate(tiles):
            if i and i % 100 == 0:
                print(f"  era5+stac {i}/{len(tiles)}  ({time.time() - t0:.0f}s)")
            try:
                ok = _era5_dates(bbox, year)
                if not ok:
                    continue
                scenes = _stac_l1c_scenes(bbox, f"{year}-06-01", f"{year}-08-31")
                hit = [s for s in scenes if (s["datetime"] or "")[:10] in ok]
                if hit:
                    tile_to_scenes[name] = hit
            except Exception as ex:
                print(f"  WARN {name}: {type(ex).__name__}: {ex}")

        selected, fallbacks, unassigned = _set_cover(tile_to_scenes)
        n_cov = sum(len(s["tile_names"]) for s in selected)
        print(f"  {len(selected)} scenes cover {n_cov} tiles; {len(unassigned)} unassigned")
        plan_scenes.extend(selected)
        all_fallbacks.update(fallbacks)

        covered = {t for sc in selected for t in sc["tile_names"]}
        tiles = [(n, b) for n, b in tiles if n not in covered]

    plan = {
        "year_primary": args.year,
        "year_fallback": args.fallback_year,
        "scenes": plan_scenes,
        "tile_fallbacks": all_fallbacks,
        "unassigned": [n for n, _ in tiles],
        "n_scenes": len(plan_scenes),
        "n_tiles_covered": sum(len(s["tile_names"]) for s in plan_scenes),
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"\n=== Plan → {args.out} ===")
    print(f"  scenes={plan['n_scenes']}  covered={plan['n_tiles_covered']}  "
          f"unassigned={len(plan['unassigned'])}  ({plan['elapsed_s']}s)")


if __name__ == "__main__":
    main()
