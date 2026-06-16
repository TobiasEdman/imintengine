#!/usr/bin/env python3
"""Stage 0 — pre-2018 granule map builder (decoupled re-coreg campaign).

For every existing ``unified_v2_512`` tile, enumerate which Sentinel-2 L1C
granules its PRE-2018 frames need, so that:

  * **Stage 2a** can sen2cor each *unique* granule exactly once (granule dedup —
    many tiles share a granule), and
  * **Stage 2b** can window + coreg-to-anchor per tile from the processed granule.

Date sourcing (per the campaign design):

  * **PRIMARY — reuse the STORED dates.** A frame is pre-2018 iff its stored date
    is strictly before ``DES_L2A_FLOOR`` (2018-01-01): always the 2016 summer
    background (``frame_2016_date``), and slot-0 only on 2018-labelled tiles
    (autumn = year-1 = 2017). ≥2018 frames stay on the Phase-1/des path.
  * **FALLBACK — only when a needed date is MISSING** (a tile with no stored
    2016 frame, or a fresh orphan): select a clean date via the **ERA5**
    atmospheric pipeline ``optimal_fetch_dates(mode="era5_then_scl")``. Never
    STAC-for-date-selection.

Granule *resolution* — "given this known date, which L1C product covers this
bbox" — reads the L1C catalogue via ``stac_l1c_scenes``. That is a lookup, not a
selection. Granules are dedup'd by ``scene_id``.

Outputs (JSON, into ``--out-dir``):

  * ``tile_granule_map.json`` — ``{tile_name: [{label, date, scene_id, mgrs_tile}]}``
  * ``unique_granules.json``  — the dedup'd Stage-2a worklist:
    ``[{scene_id, mgrs_tile, date, n_tiles, tiles: [...]}]``
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Callable

import numpy as np

from imint.training.fetch_spectral import DES_L2A_FLOOR

# Label for the 2016 summer background frame (matches the npz ``frame_2016*`` keys).
BG_LABEL = "frame_2016"
# ERA5 fallback window for the 2016 summer background (only used when no date is stored).
WINDOW_2016 = ("2016-06-01", "2016-08-31")


def pre2018_frames_for_tile(
    slot_dates: list[str],
    frame_2016_date: str,
) -> list[tuple[str, str | None]]:
    """Frames a tile needs sen2cor for, derived purely from its STORED dates.

    Returns ``(label, date)`` pairs whose date is strictly before
    ``DES_L2A_FLOOR``. The 2016 background is always required: a present date is
    used directly; an empty one yields ``(BG_LABEL, None)`` to signal the ERA5
    fallback. Temporal slots always carry a stored date once fetched, so only
    a *missing* background ever needs the fallback.
    """
    out: list[tuple[str, str | None]] = []
    # Temporal slots (0 = autumn y-1, 1-3 = VPP). Only a 2018-labelled tile has a
    # pre-2018 (2017) slot 0; ≥2018 slots are des-able and handled by Phase 1.
    for i, d in enumerate(slot_dates):
        d = (d or "").strip()
        if d and d < DES_L2A_FLOOR:
            out.append((f"slot{i}", d))
    bg = (frame_2016_date or "").strip()
    out.append((BG_LABEL, bg or None))
    return out


def resolve_granules(
    coords_wgs84: dict,
    date: str,
    *,
    stac_fn: Callable | None = None,
) -> list[dict]:
    """L1C granule(s) covering ``coords_wgs84`` on a *known* ``date`` (lookup)."""
    if stac_fn is None:  # lazy: keep network deps out of module import
        from imint.training.sen2cor_l2a import stac_l1c_scenes as stac_fn
    scenes = stac_fn(coords_wgs84, date, date)
    same_day = [s for s in scenes if (s.get("datetime") or "")[:10] == date]
    return same_day or scenes


def select_missing_date(
    coords_wgs84: dict,
    window_start: str,
    window_end: str,
    *,
    era5_fn: Callable | None = None,
) -> str | None:
    """ERA5 fallback: pick a clean date when none is stored. Never STAC-selects."""
    if era5_fn is None:  # lazy import
        from imint.training.optimal_fetch import optimal_fetch_dates as era5_fn
    plan = era5_fn(coords_wgs84, window_start, window_end, mode="era5_then_scl")
    dates = getattr(plan, "dates", None)
    return dates[0] if dates else None


def build_granule_map(
    tiles: list[dict],
    *,
    stac_fn: Callable | None = None,
    era5_fn: Callable | None = None,
) -> tuple[dict, list[dict]]:
    """Build ``(tile_granule_map, unique_granules)``.

    ``tiles`` items: ``{name, coords_wgs84, slot_dates, frame_2016_date}``.
    """
    tile_map: dict[str, list[dict]] = {}
    granule_tiles: dict[str, set[str]] = defaultdict(set)
    granule_meta: dict[str, dict] = {}

    for t in tiles:
        entries: list[dict] = []
        for label, date in pre2018_frames_for_tile(t["slot_dates"], t.get("frame_2016_date", "")):
            if date is None:  # missing 2016 background date -> ERA5 fallback
                date = select_missing_date(t["coords_wgs84"], *WINDOW_2016, era5_fn=era5_fn)
                if not date:
                    continue  # no clean 2016 date available -> logged gap
            for s in resolve_granules(t["coords_wgs84"], date, stac_fn=stac_fn):
                sid = s["scene_id"]
                mgrs = s.get("mgrs_tile", "")
                entries.append({"label": label, "date": date, "scene_id": sid, "mgrs_tile": mgrs})
                granule_tiles[sid].add(t["name"])
                granule_meta[sid] = {"mgrs_tile": mgrs, "date": date}
        tile_map[t["name"]] = entries

    unique = [
        {"scene_id": sid, **granule_meta[sid], "n_tiles": len(names), "tiles": sorted(names)}
        for sid, names in sorted(granule_tiles.items())
    ]
    return tile_map, unique


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles-dir", required=True, help="dir of existing unified_v2_512 .npz tiles")
    ap.add_argument("--out-dir", required=True, help="where to write the two JSON maps")
    ap.add_argument("--tile-size-px", type=int, default=512)
    ap.add_argument("--max-tiles", type=int, default=None, help="cap for dry-runs")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/ for fetch_unified_tiles
    from fetch_unified_tiles import gen_from_existing, _npz_str
    from imint.training.tile_config import TileConfig

    locs = gen_from_existing(args.tiles_dir, TileConfig(size_px=args.tile_size_px), max_tiles=args.max_tiles)
    tiles: list[dict] = []
    for loc in locs:
        data = np.load(loc["_existing_path"], allow_pickle=True)
        slot_dates = [_npz_str(d) for d in data["dates"]] if "dates" in data else []
        bg = _npz_str(data["frame_2016_date"]) if "frame_2016_date" in data else ""
        tiles.append({
            "name": loc["name"],
            "coords_wgs84": loc["coords_wgs84"],
            "slot_dates": slot_dates,
            "frame_2016_date": bg,
        })

    tile_map, unique = build_granule_map(tiles)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "tile_granule_map.json"), "w") as f:
        json.dump(tile_map, f, indent=2)
    with open(os.path.join(args.out_dir, "unique_granules.json"), "w") as f:
        json.dump(unique, f, indent=2)

    n_refs = sum(len(v) for v in tile_map.values())
    print(
        f"tiles={len(tile_map)}  pre2018_frame_refs={n_refs}  "
        f"unique_granules={len(unique)}  "
        f"dedup={n_refs / max(len(unique), 1):.1f}x  -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
