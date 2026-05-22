#!/usr/bin/env python3
"""Add Sentinel-2 red-edge bands (B05, B06, B07) to existing tiles.

Clay v1.5 expects 10 S2 bands; we already have 7 on disk (6-band
Prithvi spectral + B08). This script fills the remaining 3:
    B05 (rededge1, 705 nm, 20 m)
    B06 (rededge2, 740 nm, 20 m)
    B07 (rededge3, 783 nm, 20 m)

Sentinel Hub resamples 20 m bands to the tile's pixel grid (10 m), so
the output is 512×512 aligned with spectral. Uses a single evalscript
fetching all 3 bands per call → 3× API quota savings vs per-band.

Idempotent: skips tiles with has_rededge=1.

Keys written:
    rededge       (T*3, H, W) float32 — B05, B06, B07 per temporal frame
    has_rededge   int32 — 1 if any frame has red-edge data

Usage:
    python scripts/enrich_tiles_rededge.py \\
        --data-dir /data/unified_v2_512 \\
        --workers 4 \\
        --skip-existing
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


# Global rate limiter — sliding-window cap on the number of CDSE
# Process API fetches per hour. Used to spread PU consumption over time
# so a multi-thousand-frame completion run doesn't burn the entire
# monthly free tier in a single burst.
#
# Implementation: keep timestamps of the last N fetches (where N is the
# requested cap). Before each new fetch, drop entries older than 1 hour;
# if there are still N entries, sleep until the oldest one is > 1 hour
# old, then proceed. Thread-safe via a single module-level lock.
_RATE_LOCK = threading.Lock()
_RATE_TIMES: list[float] = []
_RATE_MAX_PER_HOUR = 0  # 0 = unlimited


def _set_rate_limit(max_per_hour: int) -> None:
    """Configure the global cap. 0 disables throttling."""
    global _RATE_MAX_PER_HOUR
    _RATE_MAX_PER_HOUR = int(max_per_hour)


def _wait_for_rate_limit() -> None:
    """Block until a fresh fetch is allowed under the configured cap."""
    if _RATE_MAX_PER_HOUR <= 0:
        return
    while True:
        with _RATE_LOCK:
            now = time.time()
            cutoff = now - 3600.0
            # Drop entries outside the trailing-hour window
            i = 0
            while i < len(_RATE_TIMES) and _RATE_TIMES[i] <= cutoff:
                i += 1
            if i:
                del _RATE_TIMES[:i]
            if len(_RATE_TIMES) < _RATE_MAX_PER_HOUR:
                _RATE_TIMES.append(now)
                return
            # Have to wait for the oldest in-window entry to age out
            wait_s = _RATE_TIMES[0] + 3600.0 - now
        # Cap the actual sleep to keep workers responsive on shutdown,
        # but never sleep less than 1 s to avoid a tight retry loop.
        time.sleep(max(1.0, min(wait_s + 0.5, 60.0)))


_REDEDGE_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B05", "B06", "B07"], units: "DN" }],
    output: { bands: 3, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.B05 / 10000, sample.B06 / 10000, sample.B07 / 10000];
}
"""


def _fetch_rededge_frame_cdse(
    west: float, south: float, east: float, north: float,
    date_str: str, size_px: int,
) -> np.ndarray | None:
    """Fetch B05/B06/B07 (3 bands) for one date via CDSE Process API.

    Returns (3, H, W) float32 or None on failure. PU-billed.
    Honours the global ``--max-per-hour`` cap configured via
    ``_set_rate_limit`` — blocks here before any network I/O if the
    sliding-window quota would be exceeded.
    """
    from imint.training.cdse_s2 import _fetch_s2_tiff, _parse_multiband_tiff, _get_token
    from imint.training.tile_fetch import _CDSE_SEMAPHORE

    _wait_for_rate_limit()
    _CDSE_SEMAPHORE.acquire()
    try:
        token = _get_token()
        tiff_bytes = _fetch_s2_tiff(
            west, south, east, north, size_px, size_px,
            date=date_str, token=token,
            evalscript=_REDEDGE_EVALSCRIPT,
        )
        bands = _parse_multiband_tiff(tiff_bytes, size_px, size_px, 3)
        if bands and len(bands) >= 3:
            _CDSE_SEMAPHORE.report_success()
            return np.stack(bands[:3], axis=0).astype(np.float32)
        _CDSE_SEMAPHORE.report_success()
    except Exception:
        _CDSE_SEMAPHORE.report_failure()
    finally:
        _CDSE_SEMAPHORE.release()
    return None


# Re-use a single openEO connection across worker calls — each
# ``_connect()`` does an OAuth round-trip that is wasted on per-frame use.
# threading.local because the openEO client isn't documented as thread-safe.
_des_conn_tls = threading.local()


def _get_des_conn():
    conn = getattr(_des_conn_tls, "conn", None)
    if conn is None:
        from imint.fetch import _connect
        conn = _connect()
        _des_conn_tls.conn = conn
    return conn


def _fetch_rededge_frame_des(
    west: float, south: float, east: float, north: float,
    date_str: str, size_px: int,
) -> np.ndarray | None:
    """Fetch B05/B06/B07 (3 bands) for one date via DES openEO.

    DES openEO bills compute time, not Processing Units — so this path
    is unaffected by the CDSE PU quota that previously blocked Stage B
    256 enrichment until 1 May. Routes through the unified
    ``_load_s2_cube`` helper introduced in commit 357d390.

    Returns (3, H, W) float32 or None on failure.
    """
    from datetime import datetime, timedelta
    from imint.fetch import (
        _load_s2_cube,
        _snap_to_target_grid,
        dn_to_reflectance,
        BANDS_10M,
    )

    try:
        conn = _get_des_conn()
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        temporal = [
            dt.strftime("%Y-%m-%d"),
            (dt + timedelta(days=1)).strftime("%Y-%m-%d"),
        ]
        projected_coords = {
            "west": west, "south": south, "east": east, "north": north,
            "crs": "EPSG:3006",
        }
        # Reference grid = a single 10 m band; we want only B05/B06/B07
        # (20 m, resampled to the 10 m reference internally) in the output.
        raw, crs, transform = _load_s2_cube(
            conn,
            projected_coords=projected_coords,
            temporal=temporal,
            collection_id="s2_msi_l2a",
            bands_10m=[BANDS_10M[1]],  # b03 — arbitrary 10 m anchor
            bands_20m=["b05", "b06", "b07"],
            merge_reference_into_output=False,
            empty_msg=f"DES openEO returned empty rededge cube for {date_str}",
        )
    except Exception as e:
        # Surface the failure mode so we can distinguish auth / network /
        # empty-cube — silent ``except: return None`` was masking real bugs
        # (e.g. missing DES_USER env, openEO 401) on the first job run.
        print(f"    [rededge-des] {date_str}: {type(e).__name__}: {e}", flush=True)
        return None

    if raw is None or raw.shape[0] != 3:
        print(
            f"    [rededge-des] {date_str}: unexpected cube shape "
            f"{getattr(raw, 'shape', None)}",
            flush=True,
        )
        return None

    # DES openEO routinely returns 257×257 / 256×257 grids whose origin
    # is shifted by sub-pixel from the requested bbox. Snap to the canonical
    # 10 m NMD grid — same call ``_fetch_s2_via_openeo`` makes after every
    # cube download.
    target_bounds = {"west": west, "south": south, "east": east, "north": north}
    raw, _ = _snap_to_target_grid(
        raw, transform, crs, target_bounds, pixel_size=10,
    )

    if raw.shape[1] != size_px or raw.shape[2] != size_px:
        print(
            f"    [rededge-des] {date_str}: post-snap grid {raw.shape[1:]} != "
            f"({size_px},{size_px})",
            flush=True,
        )
        return None

    bands = np.stack(
        [dn_to_reflectance(raw[i], source="des") for i in range(3)],
        axis=0,
    )
    return bands.astype(np.float32)


def _fetch_rededge_frame(
    west: float, south: float, east: float, north: float,
    date_str: str, size_px: int,
    *,
    source: str = "des",
) -> np.ndarray | None:
    """Dispatch rededge fetch to the requested backend."""
    if source == "des":
        return _fetch_rededge_frame_des(west, south, east, north, date_str, size_px)
    if source == "cdse":
        return _fetch_rededge_frame_cdse(west, south, east, north, date_str, size_px)
    raise ValueError(f"Unknown rededge source: {source!r} (expected 'des' or 'cdse')")


def enrich_one_tile(
    tile_path: str,
    skip_existing: bool = True,
    *,
    source: str = "des",
    year_filter: str | None = None,
) -> dict:
    """Add red-edge (B05/B06/B07) to one tile .npz file.

    D3 design (date-aware mismatch detection):
    - Per-frame `rededge_dates[i]` is compared against `dates[i]`.
      Any slot where they disagree (or where rededge slice is all-zeros)
      gets re-fetched.
    - Slots whose rededge_dates match dates AND have non-zero data are
      preserved untouched.
    - `skip_existing=True` (default) fast-paths when all slots are aligned.
    - `year_filter`: when set (e.g. ``"2017"``), only re-fetch frames
      whose date starts with that year — others preserved.

    Self-healing: when refetch_to_canonical_layout changes the spectral
    `dates` for a slot, the next enrich-rededge pass detects the
    rededge_dates[i] != dates[i] mismatch and re-fetches that slot. The
    refetch path doesn't need to know about rededge.
    """
    from imint.training.tile_config import TileConfig
    from imint.training.tile_bbox import resolve_tile_bbox

    name = Path(tile_path).stem
    try:
        data = dict(np.load(tile_path, allow_pickle=True))
    except Exception as e:
        return {"name": name, "status": "failed", "reason": str(e)}

    dates = data.get("dates", [])
    spectral = data.get("spectral", data.get("image"))
    if spectral is None:
        return {"name": name, "status": "failed", "reason": "no_spectral"}

    h, w = spectral.shape[1], spectral.shape[2]
    n_frames = spectral.shape[0] // 6
    new_dates = [
        str(dates[fi])[:10] if fi < len(dates) and dates[fi] else ""
        for fi in range(n_frames)
    ]

    existing_rededge = data.get("rededge")
    existing_re_dates = data.get("rededge_dates")
    if existing_re_dates is not None:
        existing_re_dates = [str(x)[:10] for x in existing_re_dates]

    # Per-slot decision: keep (date-aligned + non-zero) or fetch.
    frames_to_fetch: list[int] = []
    for fi in range(n_frames):
        already_aligned = (
            existing_rededge is not None
            and existing_re_dates is not None
            and fi < len(existing_re_dates)
            and existing_re_dates[fi] == new_dates[fi]
            and existing_rededge.shape[0] >= (fi + 1) * 3
            and bool(np.any(existing_rededge[fi*3:(fi+1)*3]))
        )
        if not already_aligned:
            # year_filter: only re-fetch frames in this year-window
            if year_filter is not None and not new_dates[fi].startswith(year_filter):
                continue
            frames_to_fetch.append(fi)

    if skip_existing and not frames_to_fetch:
        return {"name": name, "status": "skipped",
                "reason": "all_slots_date_aligned"}

    size_px = int(data.get("tile_size_px", h))
    tile_cfg = TileConfig(size_px=size_px)

    bbox = resolve_tile_bbox(name=name, tile=tile_cfg, npz_data=data)
    if bbox is None:
        return {"name": name, "status": "failed", "reason": "no_bbox"}
    tile_cfg.assert_bbox_matches(bbox)

    rededge_frames = []  # each (3, H, W) float32
    re_dates_out: list[str] = []
    for fi in range(n_frames):
        if fi not in frames_to_fetch:
            if (existing_rededge is not None
                    and existing_rededge.shape[0] >= (fi + 1) * 3):
                rededge_frames.append(
                    existing_rededge[fi*3:(fi+1)*3].astype(np.float32)
                )
                re_dates_out.append(
                    existing_re_dates[fi] if existing_re_dates is not None
                    else ""
                )
            else:
                rededge_frames.append(np.zeros((3, h, w), dtype=np.float32))
                re_dates_out.append("")
            continue

        date_str = new_dates[fi]
        if not date_str:
            rededge_frames.append(np.zeros((3, h, w), dtype=np.float32))
            re_dates_out.append("")
            continue

        frame = _fetch_rededge_frame(
            bbox["west"], bbox["south"], bbox["east"], bbox["north"],
            date_str, size_px,
            source=source,
        )
        if frame is not None and frame.shape == (3, h, w):
            rededge_frames.append(frame)
            re_dates_out.append(date_str)
        else:
            rededge_frames.append(np.zeros((3, h, w), dtype=np.float32))
            re_dates_out.append("")

    # Stack along band × frame axis → (T*3, H, W) matching spectral convention
    # Frame 0: B05, B06, B07; Frame 1: B05, B06, B07; ...
    data["rededge"] = np.concatenate(rededge_frames, axis=0)   # (T*3, H, W)
    data["rededge_dates"] = np.array(re_dates_out)             # (T,)
    valid = sum(1 for f in rededge_frames if bool(np.any(f)))
    data["has_rededge"] = np.int32(1 if valid > 0 else 0)

    # Atomic write: tmp + os.replace. ``np.savez_compressed`` writing
    # directly to ``tile_path`` was the same anti-pattern that produced
    # 188 BadZipFile-truncated tiles when the previous S1 enrichment job
    # was killed mid-write. Same pattern as build_labels.py:344 and
    # enrich_tiles_s1.py. ``np.savez_compressed`` unconditionally appends
    # ``.npz`` to its path argument unless the path already ends in
    # ``.npz``; pass a path WITHOUT the suffix so the produced file
    # lands at ``tile.npz.tmp.npz``, then rename onto ``tile.npz``.
    tmp_base = tile_path + ".tmp"
    np.savez_compressed(tmp_base, **data)
    os.replace(tmp_base + ".npz", tile_path)
    return {"name": name, "status": "ok", "valid_frames": valid}


def main():
    parser = argparse.ArgumentParser(description="Enrich tiles with S2 red-edge (B05/B06/B07)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--source", choices=["des", "cdse"], default="des",
        help="Backend for the rededge fetch. 'des' is PU-free (default); "
             "'cdse' uses the Process API and consumes Processing Units.",
    )
    parser.add_argument(
        "--year-filter", default=None,
        help="Only fetch frames whose dates[i] string starts with this "
             "prefix (e.g. '2017'). Frames outside the filter are "
             "preserved as-is from the existing rededge array. Used by "
             "the CDSE 2017-completion pass to fill in frames that DES "
             "openEO refuses (no 2017 L2A indexed) without disturbing "
             "non-2017 frames already populated by the DES rerun.",
    )
    parser.add_argument(
        "--max-per-hour", type=int, default=0,
        help="Global cap on CDSE Process API fetches per hour (sliding "
             "window across all worker threads). 0 (default) = no cap. "
             "Use to spread PU consumption across days when the total "
             "fetch count exceeds the monthly free-tier budget. "
             "Only applies to --source cdse; DES openEO is not throttled.",
    )
    args = parser.parse_args()

    _set_rate_limit(args.max_per_hour)

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    print(f"=== Red-Edge Enrichment (B05/B06/B07) ===")
    print(f"  Tiles:        {len(tiles)}")
    print(f"  Workers:      {args.workers}")
    print(f"  Source:       {args.source}")
    if args.max_per_hour > 0:
        print(f"  Rate limit:   {args.max_per_hour} fetches/hour (CDSE only)")

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()
    completed = 0
    t0 = time.time()

    def _run(path):
        nonlocal completed
        r = enrich_one_tile(
            path,
            skip_existing=args.skip_existing,
            source=args.source,
            year_filter=args.year_filter,
        )
        with lock:
            completed += 1
            stats[r.get("status", "failed")] = stats.get(r.get("status", "failed"), 0) + 1
            elapsed = time.time() - t0
            rate = completed / elapsed * 3600 if elapsed > 0 else 0
            valid = r.get("valid_frames", "")
            reason = r.get("reason", "")
            reason_str = f" [{reason}]" if r.get("status") == "failed" and reason else ""
            print(f"  [{completed}/{len(tiles)}] {r['name']}: {r['status']}"
                  f"{f' ({valid}/4 frames)' if valid != '' else ''}"
                  f"{reason_str}"
                  f" | {rate:.0f}/h", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_run, t): t for t in tiles}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"  Error: {e}")

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed/60:.1f} min ===")
    print(f"  OK={stats['ok']}  Skipped={stats['skipped']}  Failed={stats['failed']}")


if __name__ == "__main__":
    main()
