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
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.coregistration import clearest_frame_idx, coregister_interframe
from imint.training.openeo_tile_graph import (
    ALL_BANDS,
    ALL_BANDS_INDEX,
    fetch_tile_at_specific_dates,
)
from imint.training.tile_config import TileConfig

# Reuse the proven atomic-write + has-helper from the fill script (same dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_tiles_l2a import _atomic_savez, _has  # noqa: E402

HALO_PX = 520           # fetch size: 512 canonical + 4 px halo per side
CANON_PX = 512          # stored canonical size
CROP = (HALO_PX - CANON_PX) // 2   # 4 px centre-crop per side

_I_PRITHVI = list(ALL_BANDS_INDEX["prithvi"])   # (0,1,2,7,8,9) B02,B03,B04,B8A,B11,B12
_I_B08 = ALL_BANDS_INDEX["b08"][0]              # 3
_I_RE = list(ALL_BANDS_INDEX["rededge"])        # (4,5,6) B05,B06,B07
_I_B01 = ALL_BANDS_INDEX["b01"][0]              # 10
_I_B09 = ALL_BANDS_INDEX["b09"][0]              # 11

# Grid-independent identity carried verbatim. Everything else (spectral, extras,
# dates, mask, doy, bbox, centre, size, source, national_grid) is freshly set;
# labels + aux + 2016 bg are regenerated/re-fetched downstream (see docstring).
_CARRY_KEYS = ("year", "lpis_year", "tessera_year")


def _doy(date_str: str) -> int:
    """Day-of-year for an ISO date, or 0 for an empty/invalid string."""
    if not date_str:
        return 0
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").timetuple().tm_yday
    except ValueError:
        return 0


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


def crop_halo(arr: np.ndarray) -> np.ndarray:
    """Centre-crop a ``(..., 520, 520)`` array to ``(..., 512, 512)``."""
    return arr[..., CROP:CROP + CANON_PX, CROP:CROP + CANON_PX]


def assemble_fresh(
    frames512: dict[int, np.ndarray], dates: list[str], n_frames: int
) -> tuple[np.ndarray, dict]:
    """Assemble cropped fresh all-band frames into the spectral cube + extras.

    Every frame is fresh (re-gridded), so there is no keep-if-clean branch: a
    fetched+cropped frame's 6 Prithvi bands go straight into the cube and its
    B08/red-edge/B01/B09 into the extras. A frame absent from ``frames512``
    failed to fetch and is zero-filled with an empty date (downstream QC drops
    <3/4-frame tiles). Output keys/shapes mirror ``stack_extra_frames`` exactly.
    """
    h = w = CANON_PX
    spec, b08_f, re_f, b01_f, b09_f = [], [], [], [], []
    b08_d, re_d, b01_d, b09_d = [], [], [], []
    for fi in range(n_frames):
        f = frames512.get(fi)
        d = dates[fi] if fi < len(dates) else ""
        if f is not None:
            spec.append(f[_I_PRITHVI])
            b08_f.append(f[_I_B08]); re_f.append(f[_I_RE])
            b01_f.append(f[_I_B01]); b09_f.append(f[_I_B09])
            b08_d.append(d); re_d.append(d); b01_d.append(d); b09_d.append(d)
        else:
            spec.append(np.zeros((6, h, w), np.float32))
            b08_f.append(np.zeros((h, w), np.float32))
            re_f.append(np.zeros((3, h, w), np.float32))
            b01_f.append(np.zeros((h, w), np.float32))
            b09_f.append(np.zeros((h, w), np.float32))
            b08_d.append(""); re_d.append(""); b01_d.append(""); b09_d.append("")
    spectral = np.concatenate(spec, axis=0).astype(np.float32)   # (T*6, H, W)
    extras = {
        "b08": np.stack(b08_f, 0), "b08_dates": np.array(b08_d), "has_b08": _has(b08_f),
        "rededge": np.concatenate(re_f, 0), "rededge_dates": np.array(re_d),
        "has_rededge": _has(re_f),
        "b01": np.stack(b01_f, 0), "b01_dates": np.array(b01_d), "has_b01": _has(b01_f),
        "b09": np.stack(b09_f, 0), "b09_dates": np.array(b09_d), "has_b09": _has(b09_f),
    }
    return spectral, extras


def regrid_one_tile(
    tile_path: str, out_dir: str, *, skip_existing: bool = True,
    debug_precoreg: bool = False,
) -> dict:
    """Re-grid one tile onto the national lattice. See module docstring.

    ``debug_precoreg`` additionally stores the pre-M2 (raw M1, cropped) spectral
    cube under ``spectral_precoreg`` so a dry-run can render a before/after
    coregistration comparison. Off by default — it ~doubles the tile size.
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
    dates = [str(raw[fi])[:10] if fi < len(raw) and raw[fi] else "" for fi in range(n_frames)]
    slot_dates = {fi: dates[fi] for fi in range(n_frames) if dates[fi]}
    if not slot_dates:
        return {"name": name, "status": "failed", "reason": "no_dates"}

    centre = centre_of(data)
    if centre is None:
        return {"name": name, "status": "failed", "reason": "no_centre"}
    cx0, cy0 = centre

    # Snap the centre to the national lattice; 512 and 520 share that centre
    # (TileConfig snaps the SAME input to the SAME 10 m multiple), so the inner
    # 512 crop is co-centred with the canonical national bbox.
    bbox512 = TileConfig(size_px=CANON_PX).bbox_from_center(cx0, cy0)
    bbox520 = TileConfig(size_px=HALO_PX).bbox_from_center(cx0, cy0)
    cx_new = (bbox512["west"] + bbox512["east"]) // 2
    cy_new = (bbox512["south"] + bbox512["north"]) // 2

    try:
        res = fetch_tile_at_specific_dates(bbox520, slot_dates, source="des")
    except Exception as e:  # noqa: BLE001 — one tile's fetch must not kill the run
        return {"name": name, "status": "failed", "reason": f"fetch:{type(e).__name__}:{e}"}

    fresh: dict[int, np.ndarray] = {}
    for fi, entry in res.items():
        if entry is None or entry[0] is None:
            continue
        arr = np.asarray(entry[0], np.float32)
        want = (len(ALL_BANDS), HALO_PX, HALO_PX)
        if arr.shape != want:
            return {"name": name, "status": "failed",
                    "reason": f"shape:{tuple(arr.shape)}!={want}"}
        fresh[fi] = arr
    if not fresh:
        return {"name": name, "status": "failed", "reason": "fetch_empty_all_slots"}

    # M2 inter-frame coreg on the shared 520 grid, then crop the halo to 512.
    ref_idx = clearest_frame_idx(fresh)
    # Capture the pre-coreg (raw M1) cropped spectral for the before/after viz
    # BEFORE M2 rebinds `fresh` — this is the "before coregistration" state.
    precoreg = (
        assemble_fresh({fi: crop_halo(a) for fi, a in fresh.items()}, dates, n_frames)[0]
        if debug_precoreg else None
    )
    fresh = coregister_interframe(fresh, ref_idx, search_px=float(CROP))
    cropped = {fi: crop_halo(arr) for fi, arr in fresh.items()}

    spectral, extras = assemble_fresh(cropped, dates, n_frames)
    temporal_mask = np.array(
        [1 if fi in cropped else 0 for fi in range(n_frames)], np.uint8)
    doy = np.array([_doy(dates[fi]) for fi in range(n_frames)], np.int32)
    out_dates = np.array([dates[fi] if fi in cropped else "" for fi in range(n_frames)])

    save = {
        "spectral": spectral,
        "temporal_mask": temporal_mask,
        "doy": doy,
        "dates": out_dates,
        "multitemporal": np.int32(1),
        "num_frames": np.int32(n_frames),
        "num_bands": np.int32(6),
        "bbox_3006": np.array(
            [bbox512["west"], bbox512["south"], bbox512["east"], bbox512["north"]],
            dtype=np.int32),
        "easting": np.int32(cx_new),
        "northing": np.int32(cy_new),
        "tile_size_px": np.int32(CANON_PX),
        "source": "des",
        "national_grid": np.int32(1),
        "coreg_ref_frame": np.int32(ref_idx),
        **extras,
    }
    for k in _CARRY_KEYS:
        if k in data:
            save[k] = data[k]
    if precoreg is not None:
        save["spectral_precoreg"] = precoreg   # raw M1 (pre-M2) — dry-run viz only

    dest = os.path.join(out_dir, name + ".npz")
    try:
        _atomic_savez(dest, save)
    except Exception as e:  # noqa: BLE001
        return {"name": name, "status": "failed", "reason": f"write:{type(e).__name__}:{e}"}

    return {"name": name, "status": "ok", "frames": len(cropped), "ref": int(ref_idx),
            "cx": int(cx_new), "cy": int(cy_new)}


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
