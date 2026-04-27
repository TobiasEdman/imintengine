#!/usr/bin/env python3
"""Prefetch auxiliary raster channels for training tiles.

Scans an existing tile directory (data/lulc_full/tiles/) and adds
auxiliary channels to every .npz tile that doesn't already have them.

Supported channels:
  height      — tree height in meters (Skogsstyrelsen trädhöjd)
  volume      — timber volume in m³sk/ha (Skogliga grunddata)
  basal_area  — basal area / grundyta in m²/ha (Skogliga grunddata)
  diameter    — mean stem diameter / medeldiameter in cm (Skogliga grunddata)
  dem         — terrain elevation in meters (Copernicus DEM GLO-30)
  vpp         — HR-VPP vegetation phenology (5 bands: sosd, eosd, length,
                maxv, minv) via CDSE Sentinel Hub Process API

Runs independently of the main training pipeline so data can be
downloaded in the background while training continues.

Usage:
    # Fetch all standard channels
    python scripts/prefetch_aux.py --channels height volume basal_area diameter dem

    # Fetch VPP phenology channels
    python scripts/prefetch_aux.py --channels vpp

    # Fetch everything including VPP
    python scripts/prefetch_aux.py --channels height volume basal_area diameter dem vpp

    # Backward compatible: height only (same as prefetch_height.py)
    python scripts/prefetch_aux.py --channels height

The script is fully resumable — tiles that already contain the
requested channels are skipped, and progress is printed every 50 tiles.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.skg_height import fetch_height_tile
from imint.training.skg_grunddata import (
    fetch_volume_tile,
    fetch_basal_area_tile,
    fetch_diameter_tile,
)
from imint.training.copernicus_dem import fetch_dem_tile
from imint.training.cdse_vpp import fetch_vpp_tiles

# ── Channel registry ──────────────────────────────────────────────────────
# Standard channels: each fetcher returns a single (H, W) float32 array.

_CHANNEL_FETCHERS = {
    "height": fetch_height_tile,
    "volume": fetch_volume_tile,
    "basal_area": fetch_basal_area_tile,
    "diameter": fetch_diameter_tile,
    "dem": fetch_dem_tile,
}

# VPP is a multi-band channel: one fetch returns 5 bands stored as
# vpp_sosd, vpp_eosd, vpp_length, vpp_maxv, vpp_minv in the .npz.
_VPP_BAND_NAMES = ["vpp_sosd", "vpp_eosd", "vpp_length", "vpp_maxv", "vpp_minv"]

ALL_CHANNELS = list(_CHANNEL_FETCHERS.keys()) + ["vpp"]

# ── Helpers ──────────────────────────────────────────────────────────────

_TILE_RE = re.compile(r"tile_(\d+)_(\d+)\.npz$")


def _bbox_from_tile(tile_path: Path, half_m: int) -> tuple[int, int, int, int]:
    """Extract EPSG:3006 bbox from tile filename.

    Tile names are ``tile_{easting}_{northing}.npz`` where easting and
    northing are the grid-cell center.  The bbox extends ±half_m around
    that center.

    Returns (west, south, east, north) in EPSG:3006 meters.
    """
    m = _TILE_RE.search(tile_path.name)
    if not m:
        raise ValueError(f"Cannot parse easting/northing from {tile_path.name}")
    easting = int(m.group(1))
    northing = int(m.group(2))
    return (
        easting - half_m,
        northing - half_m,
        easting + half_m,
        northing + half_m,
    )


def _tile_missing_channels(
    tile_path: Path, channels: list[str],
) -> list[str]:
    """Return list of requested channels missing from the tile.

    For "vpp", checks whether all 5 VPP sub-bands are present.
    """
    try:
        with np.load(tile_path, allow_pickle=True) as d:
            missing = []
            for ch in channels:
                if ch == "vpp":
                    # VPP is present only if all 5 sub-bands exist
                    if not all(b in d for b in _VPP_BAND_NAMES):
                        missing.append("vpp")
                elif ch not in d:
                    missing.append(ch)
            return missing
    except Exception:
        return channels


def _add_channels_to_tile(
    tile_path: Path,
    channels: list[str],
    half_m: int,
    cache_dirs: dict[str, Path | None],
) -> dict:
    """Fetch missing channels and write them into an existing .npz tile.

    The tile is rewritten atomically (tmp → rename) to avoid corruption.
    """
    west, south, east, north = _bbox_from_tile(tile_path, half_m)

    # Load existing tile data
    with np.load(tile_path, allow_pickle=True) as d:
        # Re-check which channels are actually missing
        missing = [ch for ch in channels if ch not in d]
        if not missing:
            return {"status": "skip", "tile": tile_path.name}
        existing = dict(d)

    # Determine output pixel size from the image shape (C, H, W)
    img = existing.get("spectral", existing.get("image"))
    if img is not None and img.ndim >= 2:
        h_px, w_px = img.shape[-2], img.shape[-1]
    else:
        h_px, w_px = 256, 256

    # Fetch each missing channel
    fetched = {}
    for ch_name in missing:
        if ch_name == "vpp":
            # VPP: multi-band fetch → 5 sub-bands
            vpp_data = fetch_vpp_tiles(
                west, south, east, north,
                size_px=(h_px, w_px),
                cache_dir=cache_dirs.get("vpp"),
            )
            for raw_name, arr in vpp_data.items():
                key = f"vpp_{raw_name}"
                existing[key] = arr
                fetched[key] = {
                    "mean": float(arr.mean()),
                    "max": float(arr.max()),
                    "nonzero_pct": float((arr > 0).mean() * 100),
                }
        else:
            # Standard single-band channel
            fetcher = _CHANNEL_FETCHERS[ch_name]
            data = fetcher(
                west, south, east, north,
                size_px=(h_px, w_px),
                cache_dir=cache_dirs.get(ch_name),
            )
            existing[ch_name] = data
            fetched[ch_name] = {
                "mean": float(data.mean()),
                "max": float(data.max()),
                "nonzero_pct": float((data > 0).mean() * 100),
            }

    # Atomic rewrite
    tmp = tile_path.parent / (tile_path.stem + "_tmp.npz")
    np.savez_compressed(tmp, **existing)
    tmp.rename(tile_path)

    return {
        "status": "ok",
        "tile": tile_path.name,
        "channels": fetched,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prefetch auxiliary channels for training tiles",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/lulc_full",
        help="Training data directory (default: data/lulc_full)",
    )
    parser.add_argument(
        "--channels", nargs="+", default=ALL_CHANNELS,
        choices=ALL_CHANNELS,
        help=f"Channels to prefetch (default: all = {' '.join(ALL_CHANNELS)})",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel download threads (default: 4)",
    )
    parser.add_argument(
        "--patch-size-m", type=int, default=2560,
        help="Tile spatial extent in meters (default: 2560 = 256px × 10m)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable tile caching",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only count tiles needing channels, don't fetch",
    )
    args = parser.parse_args()

    channels = args.channels
    data_dir = Path(args.data_dir)
    # Support both layouts:
    #   1. legacy "lulc_full/tiles/<tile>.npz" (data_dir = data/lulc_full)
    #   2. unified "unified_v2/<tile>.npz"     (data_dir = data/unified_v2)
    # Pick whichever has .npz files. The cache dir always lives under
    # data_dir.
    tiles_dir = data_dir / "tiles"
    if not tiles_dir.exists() or not any(tiles_dir.glob("*.npz")):
        # Fall back to data_dir itself — the unified layout.
        tiles_dir = data_dir
    half_m = args.patch_size_m // 2

    # Build per-channel cache dirs (always under data_dir, not tiles_dir,
    # so caches are shared across crop_/urban_/tile_ naming conventions).
    cache_dirs: dict[str, Path | None] = {}
    for ch in channels:
        if args.no_cache:
            cache_dirs[ch] = None
        else:
            cache_dirs[ch] = data_dir / "cache" / ch

    if not tiles_dir.exists():
        print(f"ERROR: Tiles directory not found: {tiles_dir}")
        sys.exit(1)

    # ── Scan tiles ──
    # Match all naming conventions: tile_*, crop_*, urban_*, plus legacy
    # numeric-only names (e.g. "43963942.npz" from the original 256 dataset).
    all_tiles = sorted(tiles_dir.glob("*.npz"))
    print(f"Found {len(all_tiles)} tiles in {tiles_dir}")
    print(f"Channels: {', '.join(channels)}")

    # Filter to tiles missing at least one requested channel
    print("Scanning for tiles with missing channels...")
    todo: list[tuple[Path, list[str]]] = []
    already = 0
    for tp in all_tiles:
        missing = _tile_missing_channels(tp, channels)
        if missing:
            todo.append((tp, missing))
        else:
            already += 1

    print(f"  Already complete: {already}")
    print(f"  Need fetching:    {len(todo)}")

    if todo:
        # Show breakdown by channel
        ch_counts: dict[str, int] = {ch: 0 for ch in channels}
        for _, missing in todo:
            for ch in missing:
                ch_counts[ch] += 1
        for ch, count in ch_counts.items():
            print(f"    {ch}: {count} tiles")

    if not todo:
        print(f"\nAll tiles already have all requested channels. Nothing to do!")
        return

    if args.dry_run:
        print(f"\n[dry-run] Would fetch channels for {len(todo)} tiles.")
        return

    # ── Progress tracking ──
    progress_file = data_dir / "aux_prefetch_progress.json"
    progress_lock = Lock()
    stats = {"ok": 0, "skip": 0, "fail": 0, "total": len(todo)}
    t_start = time.time()

    def _save_progress():
        elapsed = time.time() - t_start
        done = stats["ok"] + stats["skip"]
        rate = done / max(elapsed, 0.01)
        remaining = (len(todo) - done) / max(rate, 0.001)
        with open(progress_file, "w") as f:
            json.dump({
                **stats,
                "channels": channels,
                "elapsed_s": round(elapsed, 1),
                "rate_tiles_per_s": round(rate, 2),
                "eta_s": round(remaining, 0),
            }, f, indent=2)

    # ── Parallel fetch ──
    print(f"\nFetching {', '.join(channels)} for {len(todo)} tiles "
          f"({args.workers} workers)...")
    for ch in channels:
        print(f"  {ch} cache: {cache_dirs[ch] or 'disabled'}")
    print()

    def _worker(item):
        tile_path, missing = item
        try:
            return _add_channels_to_tile(
                tile_path, missing, half_m, cache_dirs)
        except Exception as e:
            return {
                "status": "fail",
                "tile": tile_path.name,
                "error": f"{type(e).__name__}: {e}",
            }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, item): item for item in todo}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status = result["status"]

            with progress_lock:
                stats[status] = stats.get(status, 0) + 1

            if status == "fail":
                print(f"  ✗ {result['tile']}: {result.get('error', '?')}")
            elif status == "ok" and i <= 5:
                # Show first few successes
                ch_info = result.get("channels", {})
                parts = []
                for ch_name, ch_stats in ch_info.items():
                    parts.append(
                        f"{ch_name}={ch_stats['mean']:.1f}"
                        f"(nz={ch_stats['nonzero_pct']:.0f}%)"
                    )
                print(f"  ✓ {result['tile']}: {', '.join(parts)}")

            # Progress every 50 tiles
            if i % 50 == 0 or i == len(todo):
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 0.01)
                eta = (len(todo) - i) / max(rate, 0.001)
                print(f"  [{i}/{len(todo)}] "
                      f"{stats['ok']} ok, {stats['fail']} fail — "
                      f"{rate:.1f} tiles/s, "
                      f"ETA {eta/60:.0f}min")
                _save_progress()

    # ── Summary ──
    elapsed = time.time() - t_start
    _save_progress()
    print(f"\n{'='*60}")
    print(f"  Aux channel prefetch complete!")
    print(f"  Channels: {', '.join(channels)}")
    print(f"  OK:       {stats['ok']}")
    print(f"  Skipped:  {stats['skip']}")
    print(f"  Failed:   {stats['fail']}")
    print(f"  Time:     {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"  Rate:     {stats['ok']/max(elapsed, 1):.1f} tiles/s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
