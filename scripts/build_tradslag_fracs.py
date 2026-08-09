#!/usr/bin/env python3
"""Build per-tile Trädslag fraction sidecars from NMD2023's tilläggsskikt.

NMD2023's Trädslag layer gives, per species, a continuous 0-100 crown-cover
indication raster (EPSG:3006, 10 m, same national lattice as the NMD basskikt).
This is the *pre-threshold* signal behind NMD2023's hard forest classes. We
window-read the 4 main species at each tile's bbox and store them as a sidecar
npz so a fraction head can be supervised against them (multi-task, alongside
the existing hard label loss). Coverage follows basskikt v2.1 (~94.5%); outside
coverage the rasters read 0 and are masked out (no supervision), NOT treated as
"0% of everything present".

Species used (the 4 main, in this fixed order):
    0 tall     (Scots pine)
    1 gran     (Norway spruce)
    2 trivial  (trivial deciduous — birch/aspen/alder etc.)
    3 adel     (noble/ädellöv deciduous — oak/beech/ash etc.)

The ``osakerklassning`` raster is an uncertainty mask (1 band): where it flags
a pixel unreliable, the fraction loss should mask it out. We store it so the
training loss can honour it.

Sidecar schema — ``<out-dir>/<tile>.npz``:
    frac            (4, H, W) uint8  0-100 crown-cover per species (order above)
    frac_unreliable (H, W)    uint8  1 = osakerklassning flags this pixel
    tile_size_px    int32

Reuses the repo's own lattice-aligned windowed-read pattern
(``TileConfig.native_window`` via ``resolve_tile_bbox``), mirroring
``build_labels.py`` for bbox resolution, the per-thread rasterio-handle
pattern, atomic writes, and the skip-existing / running-count logging.

Usage:
    python scripts/build_tradslag_fracs.py \
        --data-dir /data/unified_v2_512 \
        --tradslag-dir /data/nmd2023_tradslag \
        --out-dir /data/tradslag_fracs --workers 4
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

from imint.training.tile_bbox import resolve_tile_bbox
from imint.training.tile_config import TileConfig

# Fixed species order for the frac stack. tall/gran/trivial(löv)/adel(löv) —
# the 4 main NMD2023 Trädslag species. The remaining species (lärk, contorta,
# bok, ekovrädel) are deliberately not used: the NFI dominance rule collapses
# to conifer=(tall+gran) vs deciduous=(trivial+adel), so only these 4 carry
# supervision signal aligned with the validation target.
SPECIES = ("tall", "gran", "trivial", "adel")
UNCERTAINTY = "osakerklassning"

# Raster filename template. v1_1 files on the PVC:
#   NMD2023_tradslag_<species>_v1_1.tif
_FILENAME_TMPL = "NMD2023_tradslag_{species}_v1_1.tif"


# ── Per-thread rasterio handles ──────────────────────────────────────────────
# Rasterio's DatasetReader.read() is NOT thread-safe: two threads reading
# different windows on one handle can race and return data from the wrong
# location (the exact bug that corrupted build_labels' NMD reads in 2026-04).
# Fix: one handle per (thread, path) via threading.local(), keyed by path so a
# thread touching all 5 rasters keeps a distinct handle per file.
_TLS = threading.local()


def _get_handle(path: str):
    """Return a per-thread rasterio handle for ``path`` (lazy-opened, cached)."""
    cache = getattr(_TLS, "by_path", None)
    if cache is None:
        cache = {}
        _TLS.by_path = cache
    src = cache.get(path)
    if src is None:
        import rasterio
        src = rasterio.open(path)
        cache[path] = src
    return src


def _raster_path(tradslag_dir: str, species: str) -> str:
    return os.path.join(tradslag_dir, _FILENAME_TMPL.format(species=species))


def _windowed_read(
    src, bbox_3006: dict, tile: TileConfig,
) -> np.ndarray | None:
    """Lattice-aligned native windowed read of ``src`` at the tile bbox.

    Returns (H, W) or None if the tile bbox falls outside the raster's bounds.
    Raises (via ``native_window``) if the bbox is off the raster's 10 m
    lattice — that is a caller/data bug and must surface loudly, not be
    papered over with resampling.
    """
    w = bbox_3006["west"]; s = bbox_3006["south"]
    e = bbox_3006["east"]; n = bbox_3006["north"]

    b = src.bounds
    if w < b.left or e > b.right or s < b.bottom or n > b.top:
        return None

    window = tile.native_window(src.transform, w, s, e, n)
    return src.read(1, window=window)


def build_tile_fracs(
    tile_path: str,
    tradslag_dir: str,
    out_dir: str,
    *,
    skip_existing: bool = True,
) -> dict:
    """Build the Trädslag fraction sidecar for one tile.

    Returns a status dict: ``{"name", "status", ["reason"], ["nonzero"]}``.
    ``status`` is one of ``ok`` / ``skipped`` / ``failed``. ``nonzero`` is
    True when any of the 4 species reads a nonzero fraction (coverage signal).
    """
    name = os.path.basename(tile_path).replace(".npz", "")
    out_path = os.path.join(out_dir, name + ".npz")
    if skip_existing and os.path.exists(out_path):
        return {"name": name, "status": "skipped"}

    try:
        data = dict(np.load(tile_path, allow_pickle=True))

        sp = data.get("spectral", data.get("image"))
        size_px = int(data.get("tile_size_px", sp.shape[-1] if sp is not None else 512))
        tile = TileConfig(size_px=size_px)

        bbox_3006 = resolve_tile_bbox(name=name, tile=tile, npz_data=data)
        if bbox_3006 is None:
            return {"name": name, "status": "failed", "reason": "no_bbox"}

        # Read the 4 species rasters. A tile outside a raster's bounds reads
        # as all-zero (no supervision there — masked at loss time via the
        # all-zero test), NOT as an error.
        frac = np.zeros((len(SPECIES), size_px, size_px), dtype=np.uint8)
        any_nonzero = False
        for i, species in enumerate(SPECIES):
            src = _get_handle(_raster_path(tradslag_dir, species))
            arr = _windowed_read(src, bbox_3006, tile)
            if arr is None:
                continue  # outside coverage → leave zeros
            # Clip to [0, 100] and store as uint8. The rasters are 0-100
            # crown-cover; anything above 100 (nodata sentinels like 255)
            # is clamped so it can't masquerade as a valid 100% fraction —
            # such pixels are then caught by the all-zero / uncertainty mask
            # at loss time only if genuinely zero, so we additionally zero
            # out sentinel-high nodata here.
            a = arr.astype(np.int32)
            a[a > 100] = 0  # nodata sentinel (e.g. 255) → 0 (no supervision)
            a[a < 0] = 0
            frac[i] = a.astype(np.uint8)
            if a.any():
                any_nonzero = True

        # Uncertainty mask (1 band). Its own coverage may differ; outside →
        # not-unreliable (0). Any nonzero code is treated as "flagged".
        frac_unreliable = np.zeros((size_px, size_px), dtype=np.uint8)
        unc_path = _raster_path(tradslag_dir, UNCERTAINTY)
        if os.path.exists(unc_path):
            src = _get_handle(unc_path)
            arr = _windowed_read(src, bbox_3006, tile)
            if arr is not None:
                frac_unreliable = (arr != 0).astype(np.uint8)

        # ── Invariant assertions (catch bugs at write time) ──
        if sp is not None:
            sp_h, sp_w = sp.shape[-2], sp.shape[-1]
            if frac.shape[1:] != (sp_h, sp_w):
                raise AssertionError(
                    f"frac HW={frac.shape[1:]} != spectral HW=({sp_h},{sp_w})"
                )
        if int(frac.max()) > 100:
            raise AssertionError(f"frac out of range: max={int(frac.max())} (>100)")

        payload = {
            "frac": frac,
            "frac_unreliable": frac_unreliable,
            "tile_size_px": np.int32(size_px),
        }

        # Atomic write: tmp + os.replace, mirroring build_labels. savez_compressed
        # appends ".npz" to a suffix-less base, so we rename that onto the target.
        tmp_base = out_path + ".tmp"
        np.savez_compressed(tmp_base, **payload)
        os.replace(tmp_base + ".npz", out_path)
        return {"name": name, "status": "ok", "nonzero": any_nonzero}

    except Exception as e:
        stale = out_path + ".tmp.npz"
        if os.path.exists(stale):
            try:
                os.unlink(stale)
            except OSError:
                pass
        return {"name": name, "status": "failed", "reason": str(e)[:120]}


def main() -> None:
    p = argparse.ArgumentParser(description="Build Trädslag fraction sidecars")
    p.add_argument("--data-dir", required=True, help="Directory with .npz tiles")
    p.add_argument("--tradslag-dir", required=True,
                   help="Directory with NMD2023_tradslag_<species>_v1_1.tif")
    p.add_argument("--out-dir", required=True, help="Sidecar output directory")
    p.add_argument("--tile-ids", nargs="+",
                   help="Only process these tile IDs (filename stems)")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel threads (per-thread rasterio handles)")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Rebuild sidecars that already exist")
    p.add_argument("--max-failed", type=int, default=0,
                   help="Exit non-zero if more than this many tiles fail "
                        "(default 0 — any failure fails the run)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Verify the 4 species rasters exist and share the same CRS/resolution up
    # front — fail loud rather than silently reading zeros for a mis-pathed run.
    import rasterio
    ref_crs = ref_res = None
    for species in SPECIES:
        rp = _raster_path(args.tradslag_dir, species)
        if not os.path.exists(rp):
            print(f"FATAL: missing raster {rp}")
            sys.exit(2)
        with rasterio.open(rp) as src:
            crs = src.crs
            res = (round(src.res[0], 6), round(src.res[1], 6))
            if ref_crs is None:
                ref_crs, ref_res = crs, res
                if crs is None or crs.to_epsg() != 3006:
                    print(f"FATAL: {species} CRS is {crs}, expected EPSG:3006")
                    sys.exit(2)
                if abs(res[0] - 10.0) > 1e-6 or abs(res[1] - 10.0) > 1e-6:
                    print(f"FATAL: {species} res is {res}, expected ~10 m")
                    sys.exit(2)
            elif crs != ref_crs or res != ref_res:
                print(f"FATAL: {species} CRS/res {crs}/{res} != "
                      f"reference {ref_crs}/{ref_res}")
                sys.exit(2)

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.tile_ids:
        ids = set(args.tile_ids)
        tiles = [t for t in tiles if os.path.basename(t).replace(".npz", "") in ids]

    print("=== Build Trädslag Fraction Sidecars ===")
    print(f"  Tiles:     {len(tiles)}")
    print(f"  Trädslag:  {args.tradslag_dir}")
    print(f"  Out:       {args.out_dir}")
    print(f"  Species:   {', '.join(SPECIES)} (+ {UNCERTAINTY} mask)")
    print(f"  Workers:   {args.workers}")
    print()

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    n_nonzero = 0
    fail_examples: list[str] = []
    t0 = time.time()

    skip_existing = not args.no_skip_existing
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(build_tile_fracs, t, args.tradslag_dir,
                        args.out_dir, skip_existing=skip_existing): t
            for t in tiles
        }
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            st = r.get("status", "failed")
            stats[st] = stats.get(st, 0) + 1
            if r.get("nonzero"):
                n_nonzero += 1
            if st == "failed" and len(fail_examples) < 10:
                fail_examples.append(f"{r['name']} — {r.get('reason', '?')}")
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(tiles)}] ok={stats['ok']} "
                      f"skip={stats['skipped']} fail={stats['failed']} "
                      f"nonzero={n_nonzero} | {(i+1)/elapsed*3600:.0f}/h",
                      flush=True)

    elapsed = time.time() - t0
    built = stats["ok"]
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  OK={stats['ok']}  Skipped={stats['skipped']}  Failed={stats['failed']}")
    print(f"  Coverage: {n_nonzero}/{built} newly-built tiles have any nonzero "
          f"frac ({(n_nonzero/built*100 if built else 0):.1f}%)")
    if fail_examples:
        print("  First failures:")
        for line in fail_examples:
            print(f"    {line}")
    if stats["failed"] > args.max_failed:
        print(f"FAILED: {stats['failed']} tiles failed (> --max-failed "
              f"{args.max_failed})")
        sys.exit(1)


if __name__ == "__main__":
    main()
