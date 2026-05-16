#!/usr/bin/env python3
"""sen2cor runner — download L1C SAFE, COT-gate, sen2cor, crop frame_2016.

Consumes the plan JSON from select_scenes.py. Runs inside the
ghcr.io/tobiasedman/imint-sen2cor image (which carries L2A_Process)
with python3 + numpy + rasterio + torch installed at pod startup.

Per scene in the plan:
  1. Download the L1C SAFE archive from the Google Cloud public bucket
     via imint.fetch.fetch_l1c_safe_from_gcp (anonymous, PU-free).
  2. COT gate — for each tile assigned to this scene, read the 12-band
     TOA window from the L1C SAFE, run imint.analyzers.cot_l1c, and
     skip the tile if mean COT exceeds --cot-max. Skipped tiles are
     recorded as 'deferred' so a fallback pass can retry them against
     another scene.
  3. If at least one tile passed the gate, run sen2cor (L2A_Process)
     once on the whole SAFE → L2A SAFE.
  4. For each passing tile, read the 6-band L2A reflectance window
     (B02/B03/B04/B08/B11/B12, the Prithvi spectral order) and write
     it as frame_2016 + has_frame_2016 + metadata into the tile .npz,
     atomically (tmp + os.replace).
  5. Purge the SAFE + L2A dirs so disk stays bounded.

Disk: one SAFE (~0.8 GB) + its L2A (~0.6 GB) at a time per worker;
purged after each scene. Peak ≈ workers × 1.5 GB.

Usage:
    python scripts/sen2cor_pipeline/run_sen2cor_per_scene.py \\
        --plan /data/debug/sen2cor_plan_2016.json \\
        --data-dir /data/unified_v2_512 \\
        --safe-cache /data/sen2cor_cache \\
        --cot-max 5.0 \\
        --workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Prithvi 6-band spectral order — what frame_2016 must contain.
_FRAME_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

# L1C band → JP2 resolution group (for window reads from the SAFE).
_BAND_RES = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20,
    "B07": 20, "B08": 10, "B8A": 20, "B09": 60, "B10": 60, "B11": 20,
    "B12": 20,
}

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ── SAFE band window reads ───────────────────────────────────────────────

def _find_band_jp2(safe_dir: Path, band: str) -> Path | None:
    """Locate the JP2 for a band inside an L1C SAFE GRANULE/.../IMG_DATA."""
    # L1C layout: <SAFE>/GRANULE/<granule>/IMG_DATA/*_<BAND>.jp2
    matches = list(safe_dir.glob(f"GRANULE/*/IMG_DATA/*_{band}.jp2"))
    return matches[0] if matches else None


def _read_window(
    jp2_path: Path,
    bbox_3006: dict,
    out_px: int,
) -> np.ndarray | None:
    """Read an out_px×out_px window at the EPSG:3006 bbox, bilinear.

    Returns float32 reflectance (DN / 10000) or None on failure.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds
    from rasterio.enums import Resampling

    try:
        with rasterio.open(jp2_path) as ds:
            dst_bounds = transform_bounds(
                "EPSG:3006", ds.crs,
                bbox_3006["west"], bbox_3006["south"],
                bbox_3006["east"], bbox_3006["north"],
                densify_pts=21,
            )
            win = window_from_bounds(*dst_bounds, transform=ds.transform)
            dn = ds.read(
                1, window=win, out_shape=(out_px, out_px),
                resampling=Resampling.bilinear, boundless=True, fill_value=0,
            ).astype(np.float32)
    except Exception:
        return None
    return dn / 10000.0


def _read_12band_toa(safe_dir: Path, bbox_3006: dict, out_px: int) -> dict | None:
    """Read the 12 COT bands (all except B01) as TOA reflectance."""
    from imint.analyzers.cot_l1c import COT_L1C_BAND_ORDER
    bands = {}
    for b in COT_L1C_BAND_ORDER:
        jp2 = _find_band_jp2(safe_dir, b)
        if jp2 is None:
            return None
        arr = _read_window(jp2, bbox_3006, out_px)
        if arr is None:
            return None
        bands[b] = arr
    return bands


# ── sen2cor invocation ───────────────────────────────────────────────────

def _run_sen2cor(safe_dir: Path, work_dir: Path) -> Path | None:
    """Run L2A_Process on an L1C SAFE; return the produced L2A SAFE dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "L2A_Process",
        "--resolution", "10",
        "--output_dir", str(work_dir),
        str(safe_dir),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        _log(f"    sen2cor failed: {type(e).__name__}")
        return None
    l2a = sorted(work_dir.glob("*MSIL2A*.SAFE"))
    return l2a[0] if l2a else None


def _l2a_band_path(l2a_dir: Path, band: str, res: int = 10) -> Path | None:
    """Find an L2A band JP2 at the given resolution."""
    matches = list(l2a_dir.glob(f"GRANULE/*/IMG_DATA/R{res}m/*_{band}_*.jp2"))
    if not matches:
        # B11/B12 are native 20 m; sen2cor also writes them under R10m
        matches = list(l2a_dir.glob(f"GRANULE/*/IMG_DATA/R20m/*_{band}_*.jp2"))
    return matches[0] if matches else None


# ── Per-tile output ──────────────────────────────────────────────────────

def _write_frame_2016(
    tile_path: Path,
    frame: np.ndarray,
    scene_id: str,
    scene_datetime: str,
) -> None:
    """Atomically add frame_2016 + metadata to a tile .npz."""
    with np.load(tile_path, allow_pickle=True) as d:
        data = {k: d[k] for k in d.files}
    data["frame_2016"] = frame.astype(np.float32)
    data["has_frame_2016"] = np.int32(1)
    data["frame_2016_scene"] = np.str_(scene_id)
    data["frame_2016_date"] = np.str_((scene_datetime or "")[:10])
    data["frame_2016_year"] = np.int32(
        int(scene_datetime[:4]) if scene_datetime else 0
    )
    tmp_base = str(tile_path) + ".tmp"
    np.savez_compressed(tmp_base, **data)
    os.replace(tmp_base + ".npz", tile_path)


# ── Per-scene processing ─────────────────────────────────────────────────

def _process_scene(
    scene: dict,
    data_dir: Path,
    safe_cache: Path,
    cot_max: float,
    cot_models,
    stats: dict,
    stats_lock: threading.Lock,
) -> None:
    """Download SAFE, COT-gate tiles, sen2cor, write frame_2016."""
    from imint.fetch import fetch_l1c_safe_by_name
    from imint.analyzers.cot_l1c import cloud_score_l1c
    from imint.training.tile_bbox import resolve_tile_bbox
    from imint.training.tile_config import TileConfig

    scene_id = scene["scene_id"]
    tile_names = scene["tile_names"]
    _log(f"[scene {scene_id}] {len(tile_names)} tiles")

    # 1. Download exactly the L1C SAFE the selector chose (by name —
    #    no STAC re-resolution that could pick a different scene).
    try:
        safe_dir = fetch_l1c_safe_by_name(scene_id, str(safe_cache))
    except Exception as e:
        _log(f"    SAFE download failed: {e}")
        with stats_lock:
            stats["scene_dl_fail"] += 1
            stats["deferred"] += len(tile_names)
        return

    # 2. COT gate per tile
    passing: list[tuple[str, dict, int]] = []
    for name in tile_names:
        npz_path = data_dir / f"{name}.npz"
        if not npz_path.exists():
            continue
        try:
            with np.load(npz_path, allow_pickle=True) as d:
                size_px = int(d.get("tile_size_px", 512))
                cfg = TileConfig(size_px=size_px)
                bbox = resolve_tile_bbox(name=name, tile=cfg, npz_data=d)
            if bbox is None:
                continue
            toa = _read_12band_toa(safe_dir, bbox, size_px)
            if toa is None:
                with stats_lock:
                    stats["deferred"] += 1
                continue
            score = cloud_score_l1c(toa, cot_models)
            if score["mean_cot"] > cot_max:
                with stats_lock:
                    stats["cot_rejected"] += 1
                continue
            passing.append((name, bbox, size_px))
        except Exception as e:
            _log(f"    COT gate {name}: {type(e).__name__}: {e}")
            with stats_lock:
                stats["deferred"] += 1

    if not passing:
        _log(f"    no tiles passed COT gate — purging SAFE")
        shutil.rmtree(safe_dir, ignore_errors=True)
        return

    # 3. sen2cor once for the whole SAFE
    work_dir = safe_cache / f"{scene_id}_l2a"
    l2a_dir = _run_sen2cor(safe_dir, work_dir)
    if l2a_dir is None:
        with stats_lock:
            stats["sen2cor_fail"] += 1
            stats["deferred"] += len(passing)
        shutil.rmtree(safe_dir, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)
        return

    # 4. Crop L2A 6-band → frame_2016 per passing tile
    band_paths = {b: _l2a_band_path(l2a_dir, b) for b in _FRAME_BANDS}
    if any(p is None for p in band_paths.values()):
        _log(f"    L2A missing bands: "
             f"{[b for b, p in band_paths.items() if p is None]}")
        with stats_lock:
            stats["sen2cor_fail"] += 1
            stats["deferred"] += len(passing)
    else:
        for name, bbox, size_px in passing:
            try:
                chans = []
                ok = True
                for b in _FRAME_BANDS:
                    arr = _read_window(band_paths[b], bbox, size_px)
                    if arr is None:
                        ok = False
                        break
                    chans.append(arr)
                if not ok:
                    with stats_lock:
                        stats["deferred"] += 1
                    continue
                frame = np.stack(chans, axis=0)  # (6, H, W)
                _write_frame_2016(
                    data_dir / f"{name}.npz", frame,
                    scene_id, scene.get("datetime", ""),
                )
                with stats_lock:
                    stats["ok"] += 1
            except Exception as e:
                _log(f"    write {name}: {type(e).__name__}: {e}")
                with stats_lock:
                    stats["deferred"] += 1

    # 5. Purge
    shutil.rmtree(safe_dir, ignore_errors=True)
    shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description="sen2cor per-scene runner")
    p.add_argument("--plan", required=True, help="select_scenes.py output JSON")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--safe-cache", default="/data/sen2cor_cache")
    p.add_argument("--cot-max", type=float, default=5.0,
                   help="Max mean COT (physical units) for a tile to pass the gate")
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel scenes. Each holds ~1.5 GB of SAFE+L2A.")
    p.add_argument("--max-scenes", type=int, default=None)
    args = p.parse_args()

    with open(args.plan) as f:
        plan = json.load(f)
    scenes = plan["scenes"]
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]

    data_dir = Path(args.data_dir)
    safe_cache = Path(args.safe_cache)
    safe_cache.mkdir(parents=True, exist_ok=True)

    print(f"=== sen2cor runner ===")
    print(f"  plan:       {args.plan}")
    print(f"  scenes:     {len(scenes)}")
    print(f"  cot-max:    {args.cot_max}")
    print(f"  workers:    {args.workers}")

    from imint.analyzers.cot_l1c import load_ensemble_l1c
    cot_models = load_ensemble_l1c()
    print(f"  COT ensemble: {len(cot_models)} models")

    stats = {"ok": 0, "deferred": 0, "cot_rejected": 0,
             "sen2cor_fail": 0, "scene_dl_fail": 0}
    stats_lock = threading.Lock()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(_process_scene, sc, data_dir, safe_cache,
                        args.cot_max, cot_models, stats, stats_lock)
            for sc in scenes
        ]
        for i, fut in enumerate(as_completed(futs)):
            try:
                fut.result()
            except Exception as e:
                _log(f"  scene worker error: {type(e).__name__}: {e}")
            if (i + 1) % 10 == 0:
                _log(f"  progress {i + 1}/{len(scenes)} scenes  "
                     f"ok={stats['ok']} deferred={stats['deferred']}  "
                     f"({time.time() - t0:.0f}s)")

    print(f"\n=== Done in {(time.time() - t0) / 60:.1f} min ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
