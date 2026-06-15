#!/usr/bin/env python3
"""Re-grid unified_v2_512 tiles onto the Swedish national 10 m NMD lattice.

One DES openEO download per tile (all 4 frames, all 12 ``ALL_BANDS``) at the
tile's OWN stored per-frame dates, fetched on a 520 px HALO bbox, then:

  * M1 — grid-snap each frame's transform to the national-lattice bbox extent
          (done inside ``fetch_tile_at_specific_dates`` via ``_snap_to_target_grid``;
          the bbox edges are exact 10 m NMD multiples because the centre is snapped
          by ``TileConfig.bbox_from_center``). After M1 every frame shares ONE grid,
          so the inter-frame integer offset is 0 by construction.
  * M2 — inter-frame coregistration on that shared 520 grid: each frame is
          registered to the clearest (reference) frame on B04 by MUTUAL INFORMATION
          (``estimate_mi_offset``) and shifted onto it. MI is used (not phase
          correlation) because the frames span seasons — autumn stubble vs summer
          canopy — so the same ground point looks radiometrically different;
          intensity correlation chases that phenology (~0.75 px error, even on
          clean structured imagery) while MI scores joint-histogram dependence and
          recovers the geometry (~0.05 px on a synthetic season-change fixture).
          Inter-frame alignment is the priority; absolute placement vs NMD is a
          separate, secondary concern handled (if needed) downstream.
  * CROP — 520 → 512 (centre crop, 4 px/side). The halo absorbs the sinc
          wrap-around that ``subpixel_shift`` leaves at the frame edges, so the
          stored 512 is clean. Store-fork A: the cropped 512 is canonical; the 520
          is an in-orchestrator scratch buffer that never reaches disk.

Fresh-for-all: unlike ``fill_tiles_l2a.py`` (keep-if-clean, same bbox), every frame
is re-fetched because the NEW national bbox differs from the old grid — there is
nothing on the new grid to keep. The cropped inner 512 is co-centred with the
national 512 bbox, so ``build_labels`` (run next on the new dir, no logic change)
overlays its regenerated labels pixel-exact.

Scope is S2 SPECTRAL ONLY. The new tile carries the re-gridded ``spectral`` cube,
the all-band extras (B08/red-edge/B01/B09), dates, the new national bbox/centre,
and grid-independent identity (``year``/``lpis_year``/``tessera_year``). Old-grid
labels and aux (``vpp_*``/``dem``) and the 2016 background frame are NOT carried:
``build_labels`` regenerates labels on the national bbox, and aux re-fetch is a
separate concern (out of this spectral scope).

Never mutates existing data: ``--out-dir`` is required and must differ from the
source dir. Writes are atomic (temp + ``os.replace``). Idempotent: a re-gridded
tile carries ``national_grid=1`` and is skipped on re-run unless ``--no-skip-existing``.

Usage (dry-run to scratch, inspect, THEN scale on explicit go):
    python scripts/regrid_national_512.py --data-dir /data/unified_v2_512 \\
        --out-dir /data/_national_dryrun --max-tiles 5 --workers 4

Credentials: DES_USER + DES_PASSWORD (basic auth), DES_TOKEN, or .des_token —
see ``imint.fetch._get_des_token``.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.fetch_spectral import fetch_tile_spectral
from imint.training.tile_config import TileConfig

# Reuse the proven atomic-write helper from the fill script (same dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_tiles_l2a import _atomic_savez  # noqa: E402

HALO_PX = 520           # fetch size: 512 canonical + 4 px halo per side
CANON_PX = 512          # stored canonical size
CROP = (HALO_PX - CANON_PX) // 2   # 4 px centre-crop per side

# Grid-independent identity carried verbatim. Everything else (spectral, extras,
# dates, mask, doy, bbox, centre, size, source, national_grid) is freshly set;
# labels + aux + 2016 bg are regenerated/re-fetched downstream (see docstring).
_CARRY_KEYS = ("year", "lpis_year", "tessera_year")


def centre_of(data: dict) -> tuple[int, int] | None:
    """Recover the tile centre (EPSG:3006 easting, northing).

    Prefers the explicit ``easting``/``northing`` keys the fetcher persists;
    falls back to the ``bbox_3006`` midpoint. Returns ``None`` when neither
    is present.
    """
    if "easting" in data and "northing" in data:
        return int(data["easting"]), int(data["northing"])
    if "bbox_3006" in data:
        bb = [float(x) for x in np.asarray(data["bbox_3006"]).reshape(-1)[:4]]
        return int(round((bb[0] + bb[2]) / 2)), int(round((bb[1] + bb[3]) / 2))
    return None


def regrid_one_tile(
    tile_path: str, out_dir: str, *, skip_existing: bool = True,
    debug_precoreg: bool = False,
) -> dict:
    """Re-grid one tile onto the national lattice via the canonical fetch entry.

    Thin wrapper over ``imint.training.fetch_spectral.fetch_tile_spectral`` (the
    one production M1+M2 path): read the source tile's centre + stored dates, hand
    them to the entry, then persist the entry's result with the regrid-specific
    ``national_grid`` flag + grid-independent carry-keys. ``debug_precoreg`` adds
    the pre-M2 (raw M1, cropped) spectral under ``spectral_precoreg`` for a
    before/after coreg viz (dry-run only; ~doubles tile size).
    """
    name = Path(tile_path).stem
    try:
        data = dict(np.load(tile_path, allow_pickle=True))
    except Exception as e:  # noqa: BLE001 — a corrupt .npz must not kill the run
        return {"name": name, "status": "failed", "reason": f"load:{type(e).__name__}"}

    if skip_existing and int(data.get("national_grid", 0)) == 1:
        return {"name": name, "status": "skipped", "reason": "national_grid"}

    n_frames = int(data.get("num_frames", 0))
    if not n_frames and "spectral" in data:
        n_frames = int(np.asarray(data["spectral"]).shape[0] // 6)
    if not n_frames:
        return {"name": name, "status": "failed", "reason": "no_num_frames"}

    raw = data.get("dates", [])
    dates = {fi: str(raw[fi])[:10] for fi in range(n_frames)
             if fi < len(raw) and raw[fi]}
    if not dates:
        return {"name": name, "status": "failed", "reason": "no_dates"}

    centre = centre_of(data)
    if centre is None:
        return {"name": name, "status": "failed", "reason": "no_centre"}

    # The canonical entry owns M1 (grid-snap) + M2 (inter-frame MI coreg) + crop +
    # assemble; the source tile's stored dates drive the re-fetch (no reselection).
    try:
        res = fetch_tile_spectral(
            centre, tile=TileConfig(size_px=CANON_PX), dates=dates, n_frames=n_frames,
            backend="des", halo_px=HALO_PX - CANON_PX, coregister=True,
            return_precoreg=debug_precoreg,
        )
    except Exception as e:  # noqa: BLE001 — one tile's fetch must not kill the run
        return {"name": name, "status": "failed", "reason": f"fetch:{type(e).__name__}:{e}"}
    if res is None:
        return {"name": name, "status": "failed", "reason": "fetch_empty_all_slots"}

    # Persist with the regrid-specific national-grid flag + grid-independent carry-keys.
    save = {**res, "national_grid": np.int32(1)}
    for k in _CARRY_KEYS:
        if k in data:
            save[k] = data[k]

    dest = os.path.join(out_dir, name + ".npz")
    try:
        _atomic_savez(dest, save)
    except Exception as e:  # noqa: BLE001
        return {"name": name, "status": "failed", "reason": f"write:{type(e).__name__}:{e}"}

    return {"name": name, "status": "ok", "frames": int(res["temporal_mask"].sum()),
            "ref": int(res["coreg_ref_frame"]),
            "cx": int(res["easting"]), "cy": int(res["northing"])}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-grid unified_v2_512 tiles onto the national 10 m lattice")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="Glob <dir>/*.npz for source tiles")
    src.add_argument("--tiles-file", help="Newline-separated source .npz paths")
    ap.add_argument("--out-dir", required=True,
                    help="Write re-gridded tiles here (MUST differ from source dir)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--debug-save-precoreg", action="store_true",
                    help="Persist the pre-coreg (raw M1) spectral as spectral_precoreg "
                         "for before/after viz (dry-run only — ~doubles tile size)")
    args = ap.parse_args()

    if args.data_dir:
        tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
        src_dir = os.path.abspath(args.data_dir)
    else:
        tiles = [ln.strip() for ln in Path(args.tiles_file).read_text().splitlines() if ln.strip()]
        src_dir = None
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]

    out_dir = os.path.abspath(args.out_dir)
    if src_dir is not None and out_dir == src_dir:
        print("ERROR: --out-dir must differ from --data-dir (never mutate existing data).",
              flush=True)
        return 2
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== National 512 re-grid ===  tiles={len(tiles)}  workers={args.workers}  "
          f"out={out_dir}  skip_existing={args.skip_existing}", flush=True)

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()
    done = 0
    t0 = time.time()

    def _run(path: str) -> None:
        nonlocal done
        r = regrid_one_tile(path, out_dir, skip_existing=args.skip_existing,
                            debug_precoreg=args.debug_save_precoreg)
        with lock:
            done += 1
            stats[r["status"]] = stats.get(r["status"], 0) + 1
            rate = done / (time.time() - t0) * 3600 if time.time() > t0 else 0
            extra = (f" frames={r['frames']} ref={r['ref']} c=({r['cx']},{r['cy']})"
                     if r["status"] == "ok" else f" {r.get('reason', '')}")
            print(f"  [{done}/{len(tiles)}] {r['name']}: {r['status']}{extra} | {rate:.0f}/h",
                  flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_run, t): t for t in tiles}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print(f"  worker error: {type(e).__name__}: {e}", flush=True)

    print(f"\n=== Done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    print(f"  OK={stats['ok']}  Skipped={stats['skipped']}  Failed={stats['failed']}",
          flush=True)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
