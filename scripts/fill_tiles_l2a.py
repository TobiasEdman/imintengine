#!/usr/bin/env python3
"""Fill existing unified_v2_512 tiles with the full Sentinel-2 L2A band set.

One DES openEO download per tile (all frames, all 12 ``ALL_BANDS``) at the
tile's OWN per-frame dates, then:

  * ADD   B01 (coastal aerosol, 60m) + B09 (water vapour, 60m) — absent on
          every existing tile.
  * WRITE B08 (broad NIR, 10m) fresh for every fetched frame — fixes the
          pre-fix +0.1-high DES dequant (51% of tiles) and re-aligns
          ``b08_dates`` to the spectral ``dates``.
  * KEEP-IF-CLEAN the 6-band ``spectral`` cube + red-edge: a frame's block is
          overwritten ONLY when its existing min-positive reflectance floor is
          >= ``CORRUPT_FLOOR`` (the +0.1 offset-not-applied signature, ~2% of
          spectral frames). Clean frames are left byte-identical, which avoids
          a needless scene-swap on the model input and avoids disk churn.

All other keys (labels, aux, provenance) are preserved verbatim. Writes are
atomic (temp + ``os.replace``) so an interrupted/evicted pod can never leave a
half-written .npz — the ``BadZipFile`` failure mode the direct-``savez`` enrich
scripts are vulnerable to. Idempotent: a filled tile carries ``l2a_filled=1``
and is skipped on re-run unless ``--no-skip-existing``.

The fetch path applies the DES -1000 BOA offset (``imint.utils.dn_to_reflectance``
via ``fetch_tile_all_slots_des_openeo``); prove it first with
``scripts/verify_all_bands_gate.py`` before any bulk run.

Usage (dry-run to scratch, inspect, THEN in-place):
    python scripts/fill_tiles_l2a.py --data-dir /cephfs/unified_v2_512 \\
        --out-dir /cephfs/_fill_dryrun --max-tiles 5 --workers 4
    python scripts/fill_tiles_l2a.py --data-dir /cephfs/unified_v2_512 --workers 6

Credentials: DES_USER + DES_PASSWORD (basic auth), DES_TOKEN, or .des_token —
see ``imint.fetch._get_des_token``.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.openeo_tile_graph import (
    ALL_BANDS,
    ALL_BANDS_INDEX,
    fetch_tile_at_specific_dates,
)

# min-positive reflectance at/above this == the +0.1 offset-not-applied bug
# (DES bakes the -1000 BOA offset in, so a correct tile reaches ~0 at dark
# water/shadow; an offset-not-applied tile cannot dip below ~0.10).
CORRUPT_FLOOR = 0.095

_I_PRITHVI = list(ALL_BANDS_INDEX["prithvi"])   # (0,1,2,7,8,9) B02,B03,B04,B8A,B11,B12
_I_B08 = ALL_BANDS_INDEX["b08"][0]              # 3
_I_RE = list(ALL_BANDS_INDEX["rededge"])        # (4,5,6) B05,B06,B07
_I_B01 = ALL_BANDS_INDEX["b01"][0]              # 10
_I_B09 = ALL_BANDS_INDEX["b09"][0]              # 11


def _minpos(a: np.ndarray) -> float:
    flat = np.asarray(a, np.float32).reshape(-1)
    pos = flat[flat > 1e-6]
    return float(pos.min()) if pos.size else float("nan")


def _is_corrupt(block: np.ndarray) -> bool:
    """True if the block reads +0.1 high (DES -1000 offset not applied)."""
    mn = _minpos(block)
    return (not np.isnan(mn)) and mn >= CORRUPT_FLOOR


def _has(frames: list[np.ndarray]) -> np.int32:
    return np.int32(1 if any(bool(np.any(f)) for f in frames) else 0)


def assemble_bands(
    spectral: np.ndarray,
    fresh: dict[int, np.ndarray],
    dates: list[str],
    n_frames: int,
    height: int,
    width: int,
    ex_b08: np.ndarray | None,
    ex_b08_dates: list[str] | None,
    ex_re: np.ndarray | None,
    ex_re_dates: list[str] | None,
) -> tuple[np.ndarray, dict, dict]:
    """Pure per-frame band assembly (no fetch, no IO — unit-testable).

    Args:
        spectral: existing (T*6, H, W) cube. Copied; corrupt frames replaced.
        fresh: ``{frame_idx: (12, H, W)}`` fresh DES all-band arrays. A frame
            absent from the dict failed to fetch and is preserved.
        dates: canonical per-frame ISO dates (len ``n_frames``).
        ex_b08 / ex_re: existing extras or ``None`` if the key was absent.
        ex_b08_dates / ex_re_dates: existing per-frame date lists or ``None``.

    Returns:
        ``(spectral_out, extras, stats)``. ``extras`` holds the npz keys
        b08/rededge/b01/b09 (+ ``*_dates`` + ``has_*``); ``stats`` reports
        frames fetched + spectral/red-edge frames fixed.
    """
    # Copy preserving the stored dtype so KEPT (clean) frames stay byte-identical;
    # a fresh prithvi-6 insert (float32) upcasts cleanly into the slice if needed.
    spectral = np.array(spectral)
    b08_frames: list[np.ndarray] = []
    re_frames: list[np.ndarray] = []   # each (3, H, W)
    b01_frames: list[np.ndarray] = []
    b09_frames: list[np.ndarray] = []
    b08_dts: list[str] = []
    re_dts: list[str] = []
    b01_dts: list[str] = []
    b09_dts: list[str] = []
    spec_fixed = 0
    re_fixed = 0

    for fi in range(n_frames):
        f = fresh.get(fi)
        date_fi = dates[fi] if fi < len(dates) else ""

        # spectral — keep-if-clean: only swap a corrupt frame for fresh.
        if f is not None and _is_corrupt(spectral[fi * 6:(fi + 1) * 6]):
            spectral[fi * 6:(fi + 1) * 6] = f[_I_PRITHVI]
            spec_fixed += 1

        # B08 — always fresh when fetched (corrects 51% +0.1); else preserve.
        if f is not None:
            b08_frames.append(f[_I_B08])
            b08_dts.append(date_fi)
        elif ex_b08 is not None and fi < ex_b08.shape[0]:
            b08_frames.append(np.asarray(ex_b08[fi], np.float32))
            b08_dts.append(ex_b08_dates[fi] if ex_b08_dates and fi < len(ex_b08_dates) else "")
        else:
            b08_frames.append(np.zeros((height, width), np.float32))
            b08_dts.append("")

        # red-edge — keep-if-clean when present; fill fresh when key absent.
        if ex_re is not None and ex_re.shape[0] >= (fi + 1) * 3:
            re_block = np.asarray(ex_re[fi * 3:(fi + 1) * 3], np.float32)
            if f is not None and _is_corrupt(re_block):
                re_frames.append(f[_I_RE])
                re_dts.append(date_fi)
                re_fixed += 1
            else:
                re_frames.append(re_block)
                re_dts.append(ex_re_dates[fi] if ex_re_dates and fi < len(ex_re_dates) else date_fi)
        elif f is not None:
            re_frames.append(f[_I_RE])
            re_dts.append(date_fi)
        else:
            re_frames.append(np.zeros((3, height, width), np.float32))
            re_dts.append("")

        # B01 / B09 — new bands: fresh when fetched, else zero-filled.
        if f is not None:
            b01_frames.append(f[_I_B01])
            b09_frames.append(f[_I_B09])
            b01_dts.append(date_fi)
            b09_dts.append(date_fi)
        else:
            b01_frames.append(np.zeros((height, width), np.float32))
            b09_frames.append(np.zeros((height, width), np.float32))
            b01_dts.append("")
            b09_dts.append("")

    extras = {
        "b08": np.stack(b08_frames, axis=0),
        "b08_dates": np.array(b08_dts),
        "has_b08": _has(b08_frames),
        "rededge": np.concatenate(re_frames, axis=0),
        "rededge_dates": np.array(re_dts),
        "has_rededge": _has(re_frames),
        "b01": np.stack(b01_frames, axis=0),
        "b01_dates": np.array(b01_dts),
        "has_b01": _has(b01_frames),
        "b09": np.stack(b09_frames, axis=0),
        "b09_dates": np.array(b09_dts),
        "has_b09": _has(b09_frames),
    }
    stats = {"frames_fetched": len(fresh), "spec_fixed": spec_fixed, "re_fixed": re_fixed}
    return spectral, extras, stats


def _atomic_savez(dest: str, data: dict) -> None:
    """Write ``data`` to ``dest`` atomically (temp in same dir + os.replace).

    A file handle is passed to savez_compressed (NOT a path) so numpy does not
    append a second ``.npz`` to the temp name.
    """
    dest_dir = os.path.dirname(dest) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **data)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def fill_one_tile(tile_path: str, out_dir: str | None = None,
                  skip_existing: bool = True) -> dict:
    """Fill one tile with the full L2A band set. See module docstring."""
    name = Path(tile_path).stem
    try:
        data = dict(np.load(tile_path, allow_pickle=True))
    except Exception as e:  # noqa: BLE001 — corrupt .npz must not kill the run
        return {"name": name, "status": "failed", "reason": f"load:{type(e).__name__}"}

    if skip_existing and int(data.get("l2a_filled", 0)) == 1:
        return {"name": name, "status": "skipped", "reason": "l2a_filled"}

    spectral = data.get("spectral")
    if spectral is None:
        return {"name": name, "status": "failed", "reason": "no_spectral"}
    spectral = np.asarray(spectral, np.float32)
    n_frames = spectral.shape[0] // 6
    height, width = spectral.shape[1], spectral.shape[2]

    if "bbox_3006" not in data:
        return {"name": name, "status": "failed", "reason": "no_bbox_3006"}
    bb = [float(x) for x in data["bbox_3006"]]
    bbox = {"west": bb[0], "south": bb[1], "east": bb[2], "north": bb[3]}

    raw = data.get("dates", [])
    dates = [str(raw[fi])[:10] if fi < len(raw) and raw[fi] else "" for fi in range(n_frames)]
    slot_dates = {fi: dates[fi] for fi in range(n_frames) if dates[fi]}
    if not slot_dates:
        return {"name": name, "status": "failed", "reason": "no_dates"}

    try:
        res = fetch_tile_at_specific_dates(bbox, slot_dates, source="des")
    except Exception as e:  # noqa: BLE001 — one tile's fetch error must not kill the run
        return {"name": name, "status": "failed", "reason": f"fetch:{type(e).__name__}:{e}"}

    fresh: dict[int, np.ndarray] = {}
    for fi, entry in res.items():
        if entry is None or entry[0] is None:
            continue
        arr = np.asarray(entry[0], np.float32)
        if arr.shape != (len(ALL_BANDS), height, width):
            return {"name": name, "status": "failed",
                    "reason": f"bands:{tuple(arr.shape)}!=({len(ALL_BANDS)},{height},{width})"}
        fresh[fi] = arr
    if not fresh:
        return {"name": name, "status": "failed", "reason": "fetch_empty_all_slots"}

    ex_b08 = np.asarray(data["b08"], np.float32) if "b08" in data else None
    ex_re = np.asarray(data["rededge"], np.float32) if "rededge" in data else None
    ex_b08_dates = [str(x)[:10] for x in data["b08_dates"]] if "b08_dates" in data else None
    ex_re_dates = [str(x)[:10] for x in data["rededge_dates"]] if "rededge_dates" in data else None

    spectral_out, extras, stats = assemble_bands(
        spectral, fresh, dates, n_frames, height, width,
        ex_b08, ex_b08_dates, ex_re, ex_re_dates,
    )

    data["spectral"] = spectral_out
    data.update(extras)
    data["l2a_filled"] = np.int32(1)

    dest = os.path.join(out_dir, name + ".npz") if out_dir else tile_path
    try:
        _atomic_savez(dest, data)
    except Exception as e:  # noqa: BLE001
        return {"name": name, "status": "failed", "reason": f"write:{type(e).__name__}:{e}"}

    return {"name": name, "status": "ok", **stats}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fill tiles with full S2 L2A bands")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="Glob <dir>/*.npz")
    src.add_argument("--tiles-file", help="Newline-separated .npz paths (dry-run subset)")
    ap.add_argument("--out-dir", default=None,
                    help="Write filled copies here instead of in-place (dry-run)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    if args.data_dir:
        tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    else:
        tiles = [ln.strip() for ln in Path(args.tiles_file).read_text().splitlines() if ln.strip()]
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]

    print(f"=== L2A Fill ===  tiles={len(tiles)}  workers={args.workers}  "
          f"out={'in-place' if not args.out_dir else args.out_dir}  "
          f"skip_existing={args.skip_existing}", flush=True)

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    fixed_spec = fixed_re = 0
    lock = threading.Lock()
    done = 0
    t0 = time.time()

    def _run(path: str) -> None:
        nonlocal done, fixed_spec, fixed_re
        r = fill_one_tile(path, out_dir=args.out_dir, skip_existing=args.skip_existing)
        with lock:
            done += 1
            stats[r["status"]] = stats.get(r["status"], 0) + 1
            fixed_spec += r.get("spec_fixed", 0)
            fixed_re += r.get("re_fixed", 0)
            rate = done / (time.time() - t0) * 3600 if time.time() > t0 else 0
            extra = (f" fetched={r['frames_fetched']} spec_fix={r['spec_fixed']} "
                     f"re_fix={r['re_fixed']}" if r["status"] == "ok" else f" {r.get('reason','')}")
            print(f"  [{done}/{len(tiles)}] {r['name']}: {r['status']}{extra} | {rate:.0f}/h",
                  flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_run, t): t for t in tiles}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print(f"  worker error: {type(e).__name__}: {e}", flush=True)

    print(f"\n=== Done in {(time.time()-t0)/60:.1f} min ===", flush=True)
    print(f"  OK={stats['ok']}  Skipped={stats['skipped']}  Failed={stats['failed']}  "
          f"| spectral frames fixed={fixed_spec}  red-edge frames fixed={fixed_re}", flush=True)
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
