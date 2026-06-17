#!/usr/bin/env python3
"""Backfill HR-VPP phenology into ``_recoreg`` tiles missing it.

~36% of the tiles in ``unified_v2_512_recoreg`` carry empty/missing VPP —
the five channels ``vpp_{sosd,eosd,length,maxv,minv}`` are absent or all
zero, inherited from the original dataset (the re-coreg refetch preserves
aux but never fetches VPP). Root cause: VPP via the CDSE Sentinel Hub
Process API hit PU exhaustion (HTTP 403); only partially backfilled in the
2026-05-08 audit. This script fills the gap via WEkEO (PU-free), CDSE only
as a last-resort fallback.

What it does, per tile:
  1. Skip tiles that already carry non-empty VPP (idempotent / resumable).
  2. Resolve the EPSG:3006 bbox (``bbox_3006`` key) and the tile year
     (``tessera_year`` primary, then ``lpis_year`` → ``year`` → ``dates``).
  3. Fetch the five VPP bands via ``imint.training.cdse_vpp.fetch_vpp_tiles``
     for that bbox + year. Source routing is the function's own
     ``$VPP_SOURCE`` env switch (NOT a kwarg): we force ``wekeo`` first —
     PU-free — and only fall back to ``cdse`` on a WEkEO miss. A "miss" is
     either a hard ``RuntimeError`` (no WEkEO ``index.json`` cache at all) or
     a covered-but-empty read (coverage gap — caught via the same
     ``_has_sufficient_coverage`` floor the auto-router uses).
  4. Write ``vpp_*`` into the .npz ATOMICALLY (temp + ``os.replace``),
     preserving every other field byte-for-byte.

NEVER zero-fill: if BOTH WEkEO and CDSE miss, the tile's VPP is left absent
and the tile name is recorded in a ``vpp_known_empty.json`` sidecar with a
reason — genuinely no-phenology tiles (water / urban) and true coverage
gaps are catalogued, not fabricated (see
``memory/feedback_no_zerofills_reuse_downloaded``).

Fetched COGs/arrays are cached to disk under ``--cache-dir`` (default
``<data-dir>/.vpp_cache``) by ``fetch_vpp_tiles`` itself, so re-runs and
neighbouring tiles don't re-download.

Usage (build/unit-test only — DO NOT run against the cluster yet):
    python scripts/backfill_vpp.py --data-dir /data/unified_v2_512_recoreg \\
        --workers 2 --dry-run
    python scripts/backfill_vpp.py --data-dir /data/unified_v2_512_recoreg \\
        --workers 2

Credentials:
    WEkEO  — WEKEO_USERNAME / WEKEO_PASSWORD (or ~/.hdarc); the COG cache
             dir is ``$VPP_WEKEO_DIR`` (default /data/vpp_wekeo).
    CDSE   — CDSE_CLIENT_ID / CDSE_CLIENT_SECRET (or .cdse_credentials),
             fallback only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.cdse_vpp import _has_sufficient_coverage, fetch_vpp_tiles

# The five HR-VPP channels stored per tile. ``fetch_vpp_tiles`` returns the
# bare metric names (sosd, eosd, …); the .npz key is ``vpp_<name>`` — the
# exact mapping used by scripts/prefetch_aux.py + scripts/submit_vpp_jobs.py.
_VPP_RAW_NAMES = ("sosd", "eosd", "length", "maxv", "minv")
_VPP_CHANNEL_NAMES = tuple(f"vpp_{n}" for n in _VPP_RAW_NAMES)

# Year-resolution precedence — mirrors audit_vpp_window_displacement.tile_year_of
# and fetch_unified_tiles, but with tessera_year first (the SPEC's primary key).
_YEAR_KEYS = ("tessera_year", "lpis_year", "year")

_KNOWN_EMPTY_SIDECAR = "vpp_known_empty.json"


# ── VPP presence / emptiness ─────────────────────────────────────────────

def _vpp_is_empty(data) -> bool:
    """True if the tile needs VPP — any of the 5 channels absent OR all-zero.

    An all-zero band is the empty signature: ``fetch_vpp_tiles`` writes 0 for
    nodata, so a band that never had a real fetch is uniformly 0. We treat the
    VPP block as present only when EVERY channel exists and at least one
    carries a non-zero pixel (phenology is never identically zero for a tile
    with real vegetation).
    """
    for ch in _VPP_CHANNEL_NAMES:
        if ch not in data:
            return True
    return not any(bool(np.any(np.asarray(data[ch]))) for ch in _VPP_CHANNEL_NAMES)


# ── Tile metadata resolution ─────────────────────────────────────────────

def _tile_year(data) -> int | None:
    """Resolve the tile's VPP product year.

    Precedence: ``tessera_year`` (SPEC primary, LPIS-cross-checked) →
    ``lpis_year`` → ``year`` → the latest parseable year in ``dates``.
    Returns None when nothing is resolvable (caller treats as a hard skip —
    we will not guess a year for a VPP fetch).
    """
    for key in _YEAR_KEYS:
        if key in data:
            try:
                return int(np.asarray(data[key]).item())
            except (ValueError, TypeError):
                pass
    years = [
        int(str(s)[:4])
        for s in np.asarray(data.get("dates", [])).ravel()
        if len(str(s)) >= 4 and str(s)[:4].isdigit()
    ]
    return max(years) if years else None


def _tile_bbox_3006(data) -> tuple[float, float, float, float] | None:
    """Return the tile's EPSG:3006 (west, south, east, north) bbox.

    Reads the ``bbox_3006`` array (4 floats, W/S/E/N) written by the fetcher —
    the same key fill_tiles_l2a.py uses. The bbox already sits on the NMD 10 m
    lattice (M1 grid-snap), so it is passed to ``fetch_vpp_tiles`` verbatim; no
    re-snap. Returns None if absent/malformed.
    """
    if "bbox_3006" not in data:
        return None
    b = np.asarray(data["bbox_3006"]).flatten()
    if b.size < 4:
        return None
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def _spatial_size(data) -> tuple[int, int]:
    """(H, W) pixel size from the spectral/image cube; default 512 for _recoreg."""
    img = data.get("spectral", data.get("image"))
    if img is not None and np.asarray(img).ndim >= 2:
        shp = np.asarray(img).shape
        return int(shp[-2]), int(shp[-1])
    return 512, 512


# ── Fetch: WEkEO first (PU-free), CDSE fallback ──────────────────────────

# fetch_vpp_tiles routes its source via the process-global $VPP_SOURCE env var, so
# the wekeo→cdse switch below MUST be serialized: under ThreadPoolExecutor two
# threads interleaving the env mutation could make one read "cdse" when it meant
# "wekeo" → silent CDSE-PU spend, the exact 403 this campaign exists to avoid.
_VPP_SOURCE_LOCK = threading.Lock()


def _fetch_vpp_wekeo_then_cdse(
    west: float, south: float, east: float, north: float,
    *, size_px: tuple[int, int], year: int, cache_dir: Path | None,
) -> tuple[dict[str, np.ndarray] | None, str]:
    """Fetch the 5 VPP bands, WEkEO first then CDSE. Never fabricates data.

    ``fetch_vpp_tiles`` has no ``src=`` kwarg — source is its ``$VPP_SOURCE``
    env switch. We force ``wekeo`` (PU-free), and on a miss force ``cdse``.

    A WEkEO miss is either:
      * RuntimeError — no WEkEO COG cache (``index.json``) at all; or
      * a covered-but-insufficient read (coverage gap) — detected with the
        same ``_has_sufficient_coverage`` floor the auto-router uses.

    Returns ``(bands, source)`` where ``source`` is "wekeo" or "cdse"; or
    ``(None, reason)`` when BOTH miss — the caller records the reason in the
    known-empty sidecar and writes NOTHING (no zero-fill).
    """
    # Serialize the whole set→fetch→restore (see _VPP_SOURCE_LOCK above). returns
    # release the lock cleanly; the finally restores the env while still held.
    with _VPP_SOURCE_LOCK:
        prev = os.environ.get("VPP_SOURCE")
        try:
            # ── WEkEO (forced, PU-free) ──
            os.environ["VPP_SOURCE"] = "wekeo"
            wk: dict[str, np.ndarray] | None = None
            try:
                wk = fetch_vpp_tiles(
                    west, south, east, north,
                    size_px=size_px, year=year, cache_dir=cache_dir,
                )
            except RuntimeError:
                wk = None  # no WEkEO cache → treat as a miss, try CDSE
            if wk is not None and _has_sufficient_coverage(wk):
                return wk, "wekeo"

            # ── CDSE (forced, metered fallback) ──
            os.environ["VPP_SOURCE"] = "cdse"
            try:
                cd = fetch_vpp_tiles(
                    west, south, east, north,
                    size_px=size_px, year=year, cache_dir=cache_dir,
                )
            except RuntimeError as exc:
                return None, f"wekeo_miss+cdse_error:{str(exc)[:120]}"
            if _has_sufficient_coverage(cd):
                return cd, "cdse"

            # Both sources returned but neither covers the tile → genuine gap /
            # no-phenology (water, urban). Do NOT zero-fill; catalogue it.
            return None, "no_coverage_wekeo_or_cdse"
        finally:
            if prev is None:
                os.environ.pop("VPP_SOURCE", None)
            else:
                os.environ["VPP_SOURCE"] = prev


# ── Atomic write (temp in same dir + os.replace) ─────────────────────────

def _atomic_savez(dest: str, data: dict) -> None:
    """Write ``data`` to ``dest`` atomically (temp in same dir + os.replace).

    A file handle (not a path) is passed to savez_compressed so numpy does not
    append a second ``.npz`` to the temp name. Same pattern as
    scripts/fill_tiles_l2a.py — an interrupted/evicted pod can never leave a
    half-written .npz (the BadZipFile failure mode of plain savez).
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


# ── Per-tile worker ──────────────────────────────────────────────────────

def backfill_one_tile(
    tile_path: str,
    *,
    cache_dir: Path | None,
    dry_run: bool = False,
) -> dict:
    """Backfill VPP into one tile. Returns a status dict (never raises).

    Statuses: ``skipped`` (already non-empty / no year / no bbox),
    ``filled`` (VPP written, or would-be in dry-run), ``empty`` (both
    sources missed — recorded in the known-empty sidecar by the caller),
    ``failed`` (load/write error). The tile is rewritten only on ``filled``.
    """
    name = Path(tile_path).stem
    try:
        data = dict(np.load(tile_path, allow_pickle=True))
    except Exception as e:  # noqa: BLE001 — a corrupt .npz must not kill the run
        return {"name": name, "status": "failed", "reason": f"load:{type(e).__name__}"}

    if not _vpp_is_empty(data):
        return {"name": name, "status": "skipped", "reason": "vpp_present"}

    bbox = _tile_bbox_3006(data)
    if bbox is None:
        return {"name": name, "status": "skipped", "reason": "no_bbox_3006"}

    year = _tile_year(data)
    if year is None:
        return {"name": name, "status": "skipped", "reason": "no_year"}

    size_px = _spatial_size(data)
    west, south, east, north = bbox

    bands, source = _fetch_vpp_wekeo_then_cdse(
        west, south, east, north,
        size_px=size_px, year=year, cache_dir=cache_dir,
    )
    if bands is None:
        # NEVER zero-fill — leave VPP absent, record for the known-empty set.
        return {"name": name, "status": "empty", "reason": source, "year": year}

    if dry_run:
        return {"name": name, "status": "filled", "source": source,
                "year": year, "dry_run": True}

    # Write the 5 channels under their vpp_<name> keys; preserve everything else.
    for raw_name in _VPP_RAW_NAMES:
        data[f"vpp_{raw_name}"] = np.asarray(bands[raw_name], np.float32)
    try:
        _atomic_savez(tile_path, data)
    except Exception as e:  # noqa: BLE001
        return {"name": name, "status": "failed", "reason": f"write:{type(e).__name__}:{e}"}

    return {"name": name, "status": "filled", "source": source, "year": year}


# ── Known-empty sidecar (atomic JSON) ────────────────────────────────────

def _load_known_empty(path: Path) -> dict[str, str]:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_known_empty(path: Path, mapping: dict[str, str]) -> None:
    """Atomically write the ``{tile: reason}`` known-empty catalogue."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# ── Driver ───────────────────────────────────────────────────────────────

def run(
    data_dir: str,
    *,
    workers: int = 2,
    dry_run: bool = False,
    cache_dir: str | None = None,
    max_tiles: int | None = None,
) -> dict:
    """Backfill VPP across every ``*.npz`` under ``data_dir``.

    Returns the aggregate stats dict. The known-empty sidecar
    (``vpp_known_empty.json``) is updated under ``data_dir`` for every tile
    that both sources missed (skipped in dry-run — measure first).
    """
    tiles = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if max_tiles is not None:
        tiles = tiles[:max_tiles]

    cdir = Path(cache_dir) if cache_dir else Path(data_dir) / ".vpp_cache"

    print(f"=== VPP backfill ===  tiles={len(tiles)}  workers={workers}  "
          f"dry_run={dry_run}  cache={cdir}", flush=True)

    stats = {"filled": 0, "skipped": 0, "empty": 0, "failed": 0}
    by_source = {"wekeo": 0, "cdse": 0}
    known_empty_new: dict[str, str] = {}
    lock = threading.Lock()
    done = 0
    t0 = time.time()

    def _run_one(path: str) -> None:
        nonlocal done
        r = backfill_one_tile(path, cache_dir=cdir, dry_run=dry_run)
        with lock:
            done += 1
            stats[r["status"]] = stats.get(r["status"], 0) + 1
            if r["status"] == "filled" and r.get("source"):
                by_source[r["source"]] = by_source.get(r["source"], 0) + 1
            if r["status"] == "empty":
                known_empty_new[r["name"]] = r.get("reason", "unknown")
            rate = done / (time.time() - t0) * 3600 if time.time() > t0 else 0.0
            tail = (f" src={r.get('source', '-')}" if r["status"] == "filled"
                    else f" {r.get('reason', '')}")
            print(f"  [{done}/{len(tiles)}] {r['name']}: {r['status']}{tail} "
                  f"| {rate:.0f}/h", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run_one, t): t for t in tiles}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print(f"  worker error: {type(e).__name__}: {e}", flush=True)

    # Persist the known-empty catalogue (merge with any prior run). Skipped in
    # dry-run — a dry-run only measures, it must not mutate the dataset dir.
    if known_empty_new and not dry_run:
        sidecar = Path(data_dir) / _KNOWN_EMPTY_SIDECAR
        merged = _load_known_empty(sidecar)
        merged.update(known_empty_new)
        _save_known_empty(sidecar, merged)
        print(f"  recorded {len(known_empty_new)} known-empty tile(s) → "
              f"{sidecar}", flush=True)

    print(f"\n=== Done in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    print(f"  filled={stats['filled']} (wekeo={by_source['wekeo']} "
          f"cdse={by_source['cdse']})  skipped={stats['skipped']}  "
          f"empty={stats['empty']}  failed={stats['failed']}", flush=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Backfill HR-VPP phenology into _recoreg tiles via WEkEO "
                    "(CDSE fallback). Never zero-fills.")
    ap.add_argument("--data-dir", required=True,
                    help="Directory globbed as <dir>/*.npz (the _recoreg dir)")
    ap.add_argument("--workers", type=int, default=2,
                    help="Parallel worker threads (default: 2)")
    ap.add_argument("--cache-dir", default=None,
                    help="VPP COG/array cache dir (default: <data-dir>/.vpp_cache)")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="Cap tiles processed (smoke/dry-run subset)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Measure only — fetch + report, write nothing")
    args = ap.parse_args()

    stats = run(
        args.data_dir,
        workers=args.workers,
        dry_run=args.dry_run,
        cache_dir=args.cache_dir,
        max_tiles=args.max_tiles,
    )
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
