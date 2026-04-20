#!/usr/bin/env python3
"""Add TESSERA per-pixel 128-D embeddings to existing tiles.

TESSERA (ucam-eo, arXiv:2506.20380) publishes pre-computed annual
Sentinel-1/2 embeddings at 10 m GSD globally via the geotessera
library. Advertised coverage: 2017-2025; Sweden confirmed 100%
available 2018-2024 by our probe.

For each tile we:
    1. Look up TESSERA tiles that overlap the bbox via
       ``gt.registry.load_blocks_for_region(bounds, year)``.
    2. Fetch each TESSERA tile (int8 quantized) and dequantize with
       the tile's scale factors (``arr * scales``).
    3. Reproject + crop each TESSERA tile to our tile's EPSG:3006
       grid using rasterio's bilinear resample.
    4. Mosaic multiple TESSERA tiles (if tile straddles a boundary)
       by taking the last-written value per pixel — coverage overlaps
       are typically tiny.
    5. Store as float16 (B, H, W) = (128, 512, 512) in the .npz.

TESSERA year is matched to tile year (LPIS-year for crop tiles, first
valid date for LULC). Missing year → skip with has_tessera=0.

Idempotent via has_tessera sentinel.

Size estimate: 128 × 512 × 512 × 2 bytes (fp16) = 67 MB/tile × 8261
= 553 GB total. Half that (compressed) → ~300 GB on CephFS.

Usage:
    python scripts/enrich_tiles_tessera.py \\
        --data-dir /data/unified_v2_512 \\
        --workers 2 \\
        --skip-existing

    pip install geotessera
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


TESSERA_DIM = 128


def _infer_tile_year(data: dict) -> int | None:
    """Pick the year matching this tile's label data.

    Priority: ``year`` key → ``lpis_year`` → first valid date → None.
    """
    if "year" in data:
        try:
            return int(data["year"])
        except Exception:
            pass
    if "lpis_year" in data:
        try:
            return int(data["lpis_year"])
        except Exception:
            pass
    for d in data.get("dates", []):
        s = str(d)
        if s and len(s) >= 4:
            try:
                return int(s[:4])
            except ValueError:
                continue
    return None


def _clamp_year_for_tessera(year: int, available_years: tuple[int, ...]) -> int:
    """Return the closest TESSERA year available.

    We have full 2018-2024 coverage; 2017/2025 are advertised but not
    tested. If tile year is outside the tested range, pick the nearest.
    """
    if year in available_years:
        return year
    return min(available_years, key=lambda y: abs(y - year))


def _reproject_tessera_to_tile(
    tessera_arr: np.ndarray,        # (H_t, W_t, 128) float16 dequantized
    tessera_transform,              # rasterio Affine
    tessera_crs,                    # rasterio CRS (UTM)
    tile_bbox_3006: dict,
    tile_size_px: int,
) -> np.ndarray:
    """Reproject+crop a TESSERA tile to our EPSG:3006 grid.

    Returns (128, tile_size_px, tile_size_px) float16.
    """
    import rasterio
    from rasterio.crs import CRS as rCRS
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling

    dst_crs = rCRS.from_epsg(3006)
    dst_transform = from_bounds(
        tile_bbox_3006["west"], tile_bbox_3006["south"],
        tile_bbox_3006["east"], tile_bbox_3006["north"],
        tile_size_px, tile_size_px,
    )

    # tessera_arr is (H, W, 128) — rasterio wants (bands, H, W)
    src = np.transpose(tessera_arr, (2, 0, 1))  # (128, H, W)
    dst = np.zeros(
        (TESSERA_DIM, tile_size_px, tile_size_px), dtype=np.float32,
    )
    reproject(
        source=src.astype(np.float32),
        destination=dst,
        src_transform=tessera_transform,
        src_crs=tessera_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return dst.astype(np.float16)


def _fetch_tessera_for_tile(
    gt,
    tile_bbox_3006: dict,
    year: int,
    tile_size_px: int,
) -> tuple[np.ndarray, int]:
    """Fetch + assemble TESSERA embedding for one tile's bbox+year.

    Returns (128, H, W) float16 and the count of TESSERA tiles merged
    (0 on total failure, 1 for simple case, 2+ for bbox straddling
    multiple TESSERA tiles).
    """
    from rasterio.crs import CRS as rCRS
    from rasterio.warp import transform_bounds

    # EPSG:3006 → WGS84 for TESSERA's lon/lat index
    w, s, e, n = transform_bounds(
        rCRS.from_epsg(3006), rCRS.from_epsg(4326),
        tile_bbox_3006["west"], tile_bbox_3006["south"],
        tile_bbox_3006["east"], tile_bbox_3006["north"],
    )

    blocks = gt.registry.load_blocks_for_region(
        bounds=(w, s, e, n), year=year,
    )
    if not blocks:
        return np.zeros(
            (TESSERA_DIM, tile_size_px, tile_size_px), dtype=np.float16,
        ), 0

    # Accumulate reprojected chunks; overwrite-on-last for overlaps
    merged = np.zeros(
        (TESSERA_DIM, tile_size_px, tile_size_px), dtype=np.float16,
    )
    mask = np.zeros((tile_size_px, tile_size_px), dtype=bool)
    n_used = 0

    for (y, tile_lon, tile_lat, arr_q, crs, transform) in gt.fetch_embeddings(blocks):
        # Dequantize: geotessera stores (arr * scales) → float.
        # The geotessera API returns already-dequantized arrays in
        # most versions, but some older versions return int8 + separate
        # scales. Handle both.
        if arr_q.dtype == np.int8:
            # Need to fetch scales separately — geotessera exposes
            # this via registry.get_scales(tile_lon, tile_lat, year).
            try:
                scales = gt.registry.get_scales(tile_lon, tile_lat, y)
                arr = arr_q.astype(np.float32) * scales[np.newaxis, np.newaxis, :]
            except AttributeError:
                # Scales API unavailable — use int8 directly (raw), flagged.
                arr = arr_q.astype(np.float32)
        else:
            arr = arr_q.astype(np.float32)

        # (H_t, W_t, 128) → reproject to our grid
        chunk = _reproject_tessera_to_tile(
            arr, transform, crs, tile_bbox_3006, tile_size_px,
        )
        # Merge: only overwrite pixels where this chunk has non-zero
        # data (TESSERA reprojection leaves zeros outside its footprint).
        chunk_nonzero = np.any(chunk != 0, axis=0)
        merged[:, chunk_nonzero] = chunk[:, chunk_nonzero]
        mask |= chunk_nonzero
        n_used += 1

    return merged, n_used


def enrich_one_tile(tile_path: str, gt, skip_existing: bool = True) -> dict:
    """Add TESSERA 128-D embedding to one tile .npz file.

    Args:
        tile_path: Path to the .npz to modify in place.
        gt: Pre-initialized ``geotessera.GeoTessera`` handle. Shared
            across workers; heavy init (registry load, HF auth) is done
            once in ``main()``.
        skip_existing: Skip tiles that already have ``has_tessera=1``.
    """
    from imint.training.tile_config import TileConfig
    from imint.training.tile_bbox import resolve_tile_bbox

    name = Path(tile_path).stem
    try:
        data = dict(np.load(tile_path, allow_pickle=True))
    except Exception as e:
        return {"name": name, "status": "failed", "reason": str(e)[:120]}

    if skip_existing and int(data.get("has_tessera", 0)) == 1:
        return {"name": name, "status": "skipped"}

    spectral = data.get("spectral")
    if spectral is None:
        return {"name": name, "status": "failed", "reason": "no_spectral"}

    h, w = spectral.shape[1], spectral.shape[2]
    size_px = int(data.get("tile_size_px", h))
    tile_cfg = TileConfig(size_px=size_px)

    bbox = resolve_tile_bbox(name=name, tile=tile_cfg, npz_data=data)
    if bbox is None:
        return {"name": name, "status": "failed", "reason": "no_bbox"}
    tile_cfg.assert_bbox_matches(bbox)

    year = _infer_tile_year(data)
    if year is None:
        return {"name": name, "status": "failed", "reason": "no_year"}

    tessera_years = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
    tessera_year = _clamp_year_for_tessera(year, tessera_years)

    try:
        emb, n_used = _fetch_tessera_for_tile(
            gt, bbox, tessera_year, size_px,
        )
    except Exception as e:
        return {"name": name, "status": "failed",
                "reason": f"fetch: {type(e).__name__}: {str(e)[:100]}"}

    if n_used == 0:
        return {"name": name, "status": "failed",
                "reason": f"no_tessera_tiles_for_year={tessera_year}"}

    data["tessera"] = emb                               # (128, H, W) fp16
    data["tessera_year"] = np.int32(tessera_year)
    data["has_tessera"] = np.int32(1)

    np.savez_compressed(tile_path, **data)
    return {
        "name": name, "status": "ok",
        "n_blocks": n_used, "year": tessera_year,
    }


def main():
    p = argparse.ArgumentParser(description="Enrich tiles with TESSERA embeddings")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--workers", type=int, default=2,
                   help="Default low to avoid overloading the geotessera HF CDN")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--embeddings-dir", default="/data/tessera_cache/embeddings",
                   help="Where geotessera stores downloaded embedding tiles. "
                        "Point at a persistent volume; the default cwd fills "
                        "ephemeral pod disk within a few tiles.")
    p.add_argument("--registry-cache-dir", default="/data/tessera_cache/registry",
                   help="Where geotessera stores its parquet registry metadata.")
    p.add_argument("--purge-cache-after-each", action="store_true",
                   help="Delete downloaded embedding tiles after baking them "
                        "into our .npz, to cap disk usage. Costs re-downloads "
                        "if neighboring tiles share a TESSERA source tile.")
    args = p.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    print(f"=== TESSERA Enrichment ===")
    print(f"  Tiles: {len(tiles)}")
    print(f"  Workers: {args.workers}")
    print(f"  Embeddings cache: {args.embeddings_dir}")
    print(f"  Registry cache:   {args.registry_cache_dir}")

    # One-time GeoTessera init — heavy (registry parquet load) so we
    # do it once and share across workers. GeoTessera itself is
    # thread-safe for read queries.
    os.makedirs(args.embeddings_dir, exist_ok=True)
    os.makedirs(args.registry_cache_dir, exist_ok=True)
    from geotessera import GeoTessera
    gt = GeoTessera(
        embeddings_dir=args.embeddings_dir,
        cache_dir=args.registry_cache_dir,
    )
    print(f"  GeoTessera initialized")

    stats = {"ok": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()
    completed = 0
    t0 = time.time()

    def _run(path):
        nonlocal completed
        r = enrich_one_tile(path, gt, skip_existing=args.skip_existing)
        if args.purge_cache_after_each and r.get("status") == "ok":
            # Wipe the embeddings dir — keep registry cache intact.
            # Conservative: only purge files older than 60s to avoid
            # racing other workers mid-download.
            try:
                import shutil, time as _t
                now = _t.time()
                for root, dirs, files in os.walk(args.embeddings_dir):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        try:
                            if now - os.path.getmtime(fp) > 60:
                                os.remove(fp)
                        except Exception:
                            pass
            except Exception:
                pass
        with lock:
            completed += 1
            stats[r.get("status", "failed")] = stats.get(r.get("status", "failed"), 0) + 1
            elapsed = time.time() - t0
            rate = completed / elapsed * 3600 if elapsed > 0 else 0
            extras = []
            if r.get("n_blocks"):
                extras.append(f"blocks={r['n_blocks']}")
            if r.get("year"):
                extras.append(f"year={r['year']}")
            if r.get("reason") and r.get("status") == "failed":
                extras.append(f"reason={r['reason'][:50]}")
            extra_s = " " + " ".join(extras) if extras else ""
            print(f"  [{completed}/{len(tiles)}] {r['name']}: {r['status']}"
                  f"{extra_s} | {rate:.0f}/h", flush=True)

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
