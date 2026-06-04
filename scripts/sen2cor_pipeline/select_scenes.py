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
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    from imint.training.optimal_fetch import retry_on_rate_limit

    def _query() -> list:
        client = Client.open(_CDSE_STAC_ROOT)
        search = client.search(
            collections=[_L1C_COLLECTION],
            bbox=[bbox_wgs84["west"], bbox_wgs84["south"],
                  bbox_wgs84["east"], bbox_wgs84["north"]],
            datetime=f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
            limit=300,
        )
        return list(search.items())

    out: list[dict] = []
    for item in retry_on_rate_limit(_query):
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

def _era5_dates(bbox_wgs84: dict, date_start: str, date_end: str) -> set[str]:
    """ISO dates in the given window passing the ERA5 atmosphere rules."""
    from imint.training.optimal_fetch import era5_prefilter_dates
    return set(era5_prefilter_dates(bbox_wgs84, date_start, date_end))


# ── Target window helpers ───────────────────────────────────────────────


def _window_for_target(target: str, year: int) -> tuple[str, str]:
    """Per-target date window for ERA5 + STAC lookup.

    - ``frame_2016`` → June 1 .. August 31 of ``year`` (legacy summer
      background frame).
    - ``slot:0`` → August 15 .. October 31 of ``year`` (autumn-y-1 of
      year=2018 audit tiles, year-arg here = 2017 for that case).
    - ``slot:1`` / ``slot:2`` / ``slot:3`` → growing-season windows
      consistent with the temporal-stack default layout.
    """
    if target == "frame_2016":
        return f"{year}-06-01", f"{year}-08-31"
    if target == "slot:0":
        return f"{year}-08-15", f"{year}-10-31"
    if target == "slot:1":
        return f"{year}-04-01", f"{year}-05-31"
    if target == "slot:2":
        return f"{year}-06-01", f"{year}-07-31"
    if target == "slot:3":
        return f"{year}-08-01", f"{year}-09-15"
    raise ValueError(
        f"unknown target {target!r}; expected 'frame_2016' or 'slot:N' (N=0..3)"
    )


# ── Tile inventory ───────────────────────────────────────────────────────

def _missing_slot_tiles(
    data_dir: str,
    slot_idx: int,
    audit_needs: dict[str, set[str]] | None = None,
    tile_year_filter: int | None = None,
) -> list[tuple[str, dict]]:
    """``(tile_name, wgs84_bbox)`` for tiles whose temporal slot ``slot_idx``
    needs (re)fetch.

    Two complementary criteria, both required to catch the full audit
    cohort (2026-06-04 diagnostic — pre-fix the selector caught 39 of
    2 352 audit-flagged tiles because it only looked at the empty-slot
    case and missed the PR #15 ``cap_doy=244`` victims whose slot 0
    held non-empty but wrong-time data):

    * **Audit-driven (preferred):** when ``audit_needs`` is supplied,
      a tile counts as missing iff ``f"refetch_slot_{slot_idx}"`` is
      in its needs set. The audit's verdict supersedes the local
      disk check because the audit captures the broader notion of
      "wrong" (empty OR wrong-DOY OR empty-date OR mask-clear),
      whereas the local check sees only the empty cases.
    * **Disk-derived (fallback, no audit):** ``temporal_mask[slot_idx]``
      ``== 0`` OR the corresponding spectral slice is all-zero.

    ``tile_year_filter`` (optional) drops tiles whose ``tessera_year``
    doesn't match. Required for the slot-N backfill so e.g. a 2017
    autumn pass only targets year=2018 tiles (slot 0 of a 2022 tile
    is autumn 2021, not 2017).

    The frame_2016-equivalent inventory function is
    :func:`_missing_frame_2016_tiles`; both share the bbox extraction
    + WGS84 transform.
    """
    from imint.training.tile_bbox import resolve_tile_bbox
    from imint.training.tile_config import TileConfig
    from pyproj import Transformer

    n_bands = 6
    need_key = f"refetch_slot_{slot_idx}"
    tx = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)
    missing: list[tuple[str, dict]] = []
    for npz_path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        name = Path(npz_path).stem
        if audit_needs is not None:
            needs = audit_needs.get(name)
            if needs is None or need_key not in needs:
                continue
        try:
            with np.load(npz_path, allow_pickle=True) as d:
                if tile_year_filter is not None:
                    ty_raw = d.get("tessera_year")
                    if ty_raw is None or int(ty_raw) != tile_year_filter:
                        continue
                # Disk-derived empty check only when we don't have the
                # audit's verdict — the audit-driven path above already
                # passed membership in ``needs``.
                if audit_needs is None:
                    tmask_raw = d.get("temporal_mask")
                    tmask = list(tmask_raw) if tmask_raw is not None else []
                    tmask_set = (slot_idx < len(tmask)
                                 and int(tmask[slot_idx]) == 1)
                    spec = d.get("spectral")
                    if spec is not None and spec.shape[0] >= (slot_idx + 1) * n_bands:
                        slot_slice = spec[slot_idx * n_bands:(slot_idx + 1) * n_bands]
                        has_data = bool(np.any(slot_slice))
                    else:
                        has_data = False
                    if tmask_set and has_data:
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


def _missing_frame_2016_tiles(data_dir: str) -> list[tuple[str, dict]]:
    """(tile_name, wgs84_bbox) for tiles needing a frame_2016 (re-)fetch.

    A tile counts as missing unless it has ``has_frame_2016==1`` *and* a
    ``frame_2016_bands`` matching the canonical ``PRITHVI_BANDS`` order.
    Tiles written before the band field existed, or with a stale/wrong
    order (e.g. B08 in slot 3), are re-fetched — self-healing.
    """
    from imint.training.tile_bbox import resolve_tile_bbox
    from imint.training.tile_config import TileConfig
    from imint.training.tile_fetch import PRITHVI_BANDS
    from pyproj import Transformer

    tx = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)
    missing: list[tuple[str, dict]] = []
    for npz_path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        name = Path(npz_path).stem
        try:
            with np.load(npz_path, allow_pickle=True) as d:
                has = (
                    "frame_2016" in d.files
                    and int(d.get("has_frame_2016", 0)) == 1
                )
                bands_ok = (
                    "frame_2016_bands" in d.files
                    and [str(b) for b in d["frame_2016_bands"]] == PRITHVI_BANDS
                )
                if has and bands_ok:
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


def _resolve_tile(
    name: str, bbox: dict, date_start: str, date_end: str,
) -> tuple[str, list[dict] | None]:
    """ERA5 prefilter + STAC lookup for one tile, in the given window.

    Returns ``(name, hit_scenes)`` — ``hit_scenes`` is the L1C scenes
    covering the tile on an ERA5-clear date, or ``None`` if the weather
    prefilter rejects every date or no scene matches. Pure per-tile work
    with no shared state, so it parallelises cleanly.
    """
    ok = _era5_dates(bbox, date_start, date_end)
    if not ok:
        return name, None
    scenes = _stac_l1c_scenes(bbox, date_start, date_end)
    hit = [s for s in scenes if (s["datetime"] or "")[:10] in ok]
    return name, (hit or None)


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
    p.add_argument("--fallback-year", type=int, default=None,
                   help="Optional second year to try for tiles still uncovered "
                        "after the primary pass. Legacy frame_2016 default 2015 "
                        "is only applied if --target=frame_2016 and this is "
                        "left unset.")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--workers", type=int, default=16,
                   help="Parallel ERA5+STAC lookups. Each tile is one "
                        "Open-Meteo call + one CDSE STAC query.")
    p.add_argument("--target", default="frame_2016",
                   help="What slot this plan fills: 'frame_2016' (default — "
                        "legacy background backfill) or 'slot:N' (N=0..3 — "
                        "temporal-stack slot, e.g. slot:0 for the 2017 autumn "
                        "→ year=2018 audit-tile backfill).")
    p.add_argument("--audit-json", default=None,
                   help="Optional audit JSON; restricts the tile inventory "
                        "to the names listed in its unique_affected_tiles / "
                        "unique_problem_tiles field.")
    args = p.parse_args()

    # Default fallback-year only for legacy frame_2016 backfill.
    if args.fallback_year is None and args.target == "frame_2016":
        args.fallback_year = 2015

    t0 = time.time()
    print("=== sen2cor scene selector ===")
    print(f"  data-dir:    {args.data_dir}")
    print(f"  target:      {args.target}")
    print(f"  year:        {args.year}" +
          (f" (fallback {args.fallback_year})"
           if args.fallback_year else " (no fallback)"))

    # Audit filter — two parallel representations:
    #   audit_names  — flat set, used by the frame_2016 path (no
    #                  per-tile needs concept on background frames).
    #   audit_needs  — {name: set(needs)}, used by the slot-N path so
    #                  the selector trusts the audit's verdict on what
    #                  "missing" means (catches the PR #15 cap_doy=244
    #                  victims whose slot is non-empty but wrong-time).
    audit_names: set[str] | None = None
    audit_needs: dict[str, set[str]] | None = None
    if args.audit_json:
        with open(args.audit_json) as f:
            audit = json.load(f)
        # Prefer the rich tiles_with_issues schema; fall back to the
        # flat unique_problem_tiles list for older audits.
        tiles_with_issues = audit.get("tiles_with_issues") if isinstance(audit, dict) else None
        if tiles_with_issues:
            audit_needs = {}
            for t in tiles_with_issues:
                nm = t.get("name", "")
                if nm.endswith(".npz"):
                    nm = nm[:-4]
                if nm:
                    audit_needs[nm] = set(t.get("needs") or [])
            audit_names = set(audit_needs)
        else:
            names = (audit.get("unique_affected_tiles")
                     or audit.get("unique_problem_tiles") or [])
            audit_names = {n[:-4] if n.endswith(".npz") else n for n in names}
        print(f"  audit-json:  {args.audit_json} → {len(audit_names)} tiles"
              + (" (with per-tile needs)" if audit_needs else ""))

    # Target-aware inventory + window selection.
    if args.target == "frame_2016":
        tiles = _missing_frame_2016_tiles(args.data_dir)
        # If audit-json supplied, intersect.
        if audit_names is not None:
            tiles = [(n, b) for n, b in tiles if n in audit_names]
        inventory_label = "tiles missing frame_2016"
    elif args.target.startswith("slot:"):
        try:
            slot_idx = int(args.target.split(":", 1)[1])
        except ValueError:
            sys.exit(f"--target slot:N — N must be an integer, "
                     f"got {args.target!r}")
        if slot_idx < 0 or slot_idx > 3:
            sys.exit(f"--target slot:{slot_idx} out of range; valid 0..3")
        # Derive tile_year filter from (target, --year). Slot 0 holds the
        # autumn-y-1 frame for a tile of year=y, so a 2017 autumn pass
        # targets year=2018 tiles. Slots 1-3 are same-year growing season.
        tile_year_filter = None
        if args.year is not None:
            tile_year_filter = args.year + 1 if slot_idx == 0 else args.year
        tiles = _missing_slot_tiles(
            args.data_dir, slot_idx,
            audit_needs=audit_needs,
            tile_year_filter=tile_year_filter,
        )
        inventory_label = (f"tiles missing slot {slot_idx}"
                           + (f" (tile_year={tile_year_filter})"
                              if tile_year_filter is not None else ""))
    else:
        sys.exit(f"--target must be 'frame_2016' or 'slot:N', got {args.target!r}")

    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    print(f"  {inventory_label}: {len(tiles)}")

    plan_scenes: list[dict] = []
    all_fallbacks: dict[str, list[str]] = {}

    # Years to try. Fallback only applies if explicitly set.
    pass_years = [args.year]
    if args.fallback_year:
        pass_years.append(args.fallback_year)

    for pass_idx, year in enumerate(pass_years):
        if not tiles:
            break
        date_start, date_end = _window_for_target(args.target, year)
        print(f"\n--- Pass {pass_idx + 1}: year={year} "
              f"({date_start}..{date_end}), remaining={len(tiles)} ---")
        tile_to_scenes: dict[str, list[dict]] = {}
        done = 0
        done_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_resolve_tile, name, bbox,
                                date_start, date_end): name
                    for name, bbox in tiles}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    _, hit = fut.result()
                    if hit:
                        tile_to_scenes[name] = hit
                except Exception as ex:
                    print(f"  WARN {name}: {type(ex).__name__}: {ex}")
                with done_lock:
                    done += 1
                    if done % 200 == 0:
                        print(f"  era5+stac {done}/{len(tiles)}  "
                              f"({time.time() - t0:.0f}s)", flush=True)

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
