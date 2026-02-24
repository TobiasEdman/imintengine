"""
imint/training/prepare_data.py — Fetch and cache LULC training data

Orchestrates bulk fetching of Sentinel-2 + NMD tile pairs from DES,
with progress tracking for resumability.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import TrainingConfig
from .class_schema import nmd_raster_to_lulc
from .sampler import (
    generate_grid, grid_to_wgs84, split_by_latitude,
    densify_grid, generate_densification_regions,
)


def prepare_training_data(config: TrainingConfig) -> None:
    """Main entry point for training data preparation.

    Steps:
        1. Generate uniform grid across Sweden
        2. Convert to WGS84 and split by latitude
        3. For each grid cell, fetch Sentinel-2 + NMD
        4. Convert NMD to LULC labels and save as .npz
        5. Write split files and class statistics
    """
    from ..fetch import (
        fetch_des_data, fetch_nmd_data,
        _connect, _stac_available_dates, FetchError,
    )

    data_dir = Path(config.data_dir)
    tiles_dir = data_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate grid ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  LULC Training Data Preparation")
    print(f"  Grid spacing: {config.grid_spacing_m}m")
    print(f"  Years: {config.years}")
    print(f"  Growing season: months {config.growing_season[0]}-{config.growing_season[1]}")
    print(f"{'='*60}")

    patch_size_m = config.fetch_pixels * 10  # 10m resolution
    cells = generate_grid(
        spacing_m=config.grid_spacing_m,
        patch_size_m=patch_size_m,
    )
    base_count = len(cells)

    # Optional: densify grid in rare-class areas
    if config.enable_grid_densification:
        regions = generate_densification_regions()
        cells = densify_grid(
            cells,
            densification_regions=regions,
            densify_spacing_m=config.densify_spacing_m,
            patch_size_m=patch_size_m,
        )
        print(f"\n  Grid: {base_count} base + "
              f"{len(cells) - base_count} densified = {len(cells)} locations")
    else:
        print(f"\n  Grid: {len(cells)} candidate locations")

    cells = grid_to_wgs84(cells)

    # ── Step 2: Split by latitude ─────────────────────────────────────
    train_cells, val_cells, test_cells = split_by_latitude(
        cells,
        val_lat_min=config.val_latitude_min,
        val_lat_max=config.val_latitude_max,
        test_lat_min=config.test_latitude_min,
    )
    print(f"  Split: train={len(train_cells)}, val={len(val_cells)}, test={len(test_cells)}")

    # ── Step 3: Load progress ─────────────────────────────────────────
    progress_path = data_dir / "progress.json"
    progress = _load_progress(progress_path)
    completed = set(progress.get("completed", []))
    failed = set(progress.get("failed", []))
    print(f"  Progress: {len(completed)} completed, {len(failed)} failed")

    # ── Step 4: Verify DES connection ─────────────────────────────────
    #   Preferred authentication: basic auth via environment variables
    #     export DES_USER=testuser DES_PASSWORD=secretpassword
    #   Alternative: DES_TOKEN (OIDC access token)
    print("\n  Verifying DES connection...")
    conn = _connect()
    print("  DES connection OK")

    # ── Step 4b: NMD pre-filter — cache NMD + remove lake cells ────
    #   The land mask (generate_grid) already removed ocean/sea cells.
    #   This step pre-fetches NMD for remaining cells to:
    #     a) Cache NMD on disk so the main loop fetch is instant
    #     b) Filter cells that are >95% inland water/background
    #   NMD is cached on disk (.nmd_cache/) so the second fetch in the
    #   main loop (with target_shape) hits cache and is free.
    print("\n  Pre-filtering cells via NMD (caching + lake filter)...")
    water_skipped = 0
    nmd_failed = 0
    nmd_land_cells = set()  # cell_keys that passed the land-fraction check

    all_cells_flat = train_cells + val_cells + test_cells
    total_to_filter = len(all_cells_flat)

    # NMD pre-filter log for dashboard
    nmd_log_path = data_dir / "nmd_prefilter_log.json"
    nmd_log = {
        "status": "running",
        "total_cells": total_to_filter,
        "processed": 0,
        "land_kept": 0,
        "water_skipped": 0,
        "failed": 0,
        "already_done": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": None,
    }
    _write_prepare_log(nmd_log_path, nmd_log)

    for ci, cell in enumerate(all_cells_flat):
        cell_key = f"{cell.easting}_{cell.northing}"
        if cell_key in completed or cell_key in failed:
            nmd_log["already_done"] = nmd_log.get("already_done", 0) + 1
            nmd_log["processed"] = ci + 1
            if (ci + 1) % 10 == 0:
                nmd_log["updated_at"] = datetime.now(timezone.utc).isoformat()
                _write_prepare_log(nmd_log_path, nmd_log)
            continue
        coords_nmd = {
            "west": cell.west_wgs84, "east": cell.east_wgs84,
            "south": cell.south_wgs84, "north": cell.north_wgs84,
        }
        try:
            nmd_result = fetch_nmd_data(coords=coords_nmd)
            labels = nmd_raster_to_lulc(
                nmd_result.nmd_raster, num_classes=config.num_classes,
            )
            land_frac = float(np.mean(
                (labels > 0) & (labels != 18) & (labels != 19)))
            if land_frac < 0.05:
                water_skipped += 1
                failed.add(cell_key)
                print(f"    {cell_key}: SKIP water — "
                      f"land_frac={land_frac:.1%}")
            else:
                nmd_land_cells.add(cell_key)
                print(f"    {cell_key}: land_frac={land_frac:.1%} ✓ "
                      f"(classes={np.unique(labels).tolist()})")
        except Exception as e:
            nmd_failed += 1
            print(f"    NMD fail {cell_key}: {type(e).__name__}: {e}")
            failed.add(cell_key)

        # Update NMD pre-filter log for dashboard
        nmd_log["processed"] = ci + 1
        nmd_log["land_kept"] = len(nmd_land_cells)
        nmd_log["water_skipped"] = water_skipped
        nmd_log["failed"] = nmd_failed
        nmd_log["updated_at"] = datetime.now(timezone.utc).isoformat()
        if (ci + 1) % 5 == 0 or ci + 1 == total_to_filter:
            _write_prepare_log(nmd_log_path, nmd_log)

        if (ci + 1) % 20 == 0:
            print(f"    NMD pre-filter: {ci+1}/{total_to_filter} "
                  f"(kept {len(nmd_land_cells)}, water={water_skipped}, "
                  f"fail={nmd_failed})")

    print(f"  NMD pre-filter done: {len(nmd_land_cells)} land cells, "
          f"{water_skipped} water, {nmd_failed} failed")

    # Final NMD log update
    nmd_log["status"] = "completed"
    nmd_log["processed"] = total_to_filter
    nmd_log["land_kept"] = len(nmd_land_cells)
    nmd_log["water_skipped"] = water_skipped
    nmd_log["failed"] = nmd_failed
    nmd_log["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_prepare_log(nmd_log_path, nmd_log)

    # Save progress after pre-filter (water cells are now in failed set)
    _save_progress(progress_path, completed, failed)

    # ── Prepare log for dashboard ───────────────────────────────────
    total_cells = len(train_cells) + len(val_cells) + len(test_cells)
    prep_log = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": None,
        "grid_cells": total_cells,
        "completed": len(completed),
        "failed": len(failed),
        "current_split": "",
        "tiles_saved": len(completed),
        "latest_tile": "",
        "latest_date": "",
        "latest_cloud": 0.0,
        "class_counts": {},
        "elapsed_s": 0.0,
    }
    prep_log_path = data_dir / "prepare_log.json"
    t_start = time.time()
    _write_prepare_log(prep_log_path, prep_log)

    # ── Step 5: Process all cells ─────────────────────────────────────
    all_splits = [
        ("train", train_cells),
        ("val", val_cells),
        ("test", test_cells),
    ]

    class_counts = {}
    tile_histograms = {}  # per-tile class pixel counts for rarity scoring
    tile_names_by_split = {"train": [], "val": [], "test": []}
    tile_idx = len(completed)

    total_cells = sum(len(sc) for _, sc in all_splits)
    processed_count = 0

    for split_name, split_cells in all_splits:
        print(f"\n  Processing {split_name} split ({len(split_cells)} cells)...")
        prep_log["current_split"] = split_name

        for i, cell in enumerate(split_cells):
            cell_key = f"{cell.easting}_{cell.northing}"
            processed_count += 1

            # Progress counter every 5 cells
            if processed_count % 5 == 0 or processed_count == total_cells:
                print(f"    Progress: {processed_count}/{total_cells} cells | "
                      f"{len(completed)} OK, {len(failed)} failed, "
                      f"{tile_idx} tiles saved")

            if cell_key in completed:
                # Find existing tile name
                tile_name = f"tile_{cell_key}.npz"
                if (tiles_dir / tile_name).exists():
                    tile_names_by_split[split_name].append(tile_name)
                continue

            if cell_key in failed:
                continue

            # Skip cells not in nmd_land_cells (filtered by pre-filter)
            if cell_key not in nmd_land_cells and cell_key not in completed:
                continue

            coords = {
                "west": cell.west_wgs84,
                "east": cell.east_wgs84,
                "south": cell.south_wgs84,
                "north": cell.north_wgs84,
            }

            tile_name = f"tile_{cell_key}.npz"
            success = False

            # Try each year
            for year in config.years:
                try:
                    # Find best cloud-free date in growing season
                    date_start = f"{year}-{config.growing_season[0]:02d}-01"
                    date_end = f"{year}-{config.growing_season[1]:02d}-30"

                    dates = _stac_available_dates(
                        coords, date_start, date_end,
                        scene_cloud_max=80,
                    )
                    if not dates:
                        print(f"    {cell_key} {year}: STAC 0 scenes")
                        continue
                    print(f"    {cell_key} {year}: {len(dates)} scenes, "
                          f"cloud [{dates[0][1]:.0f}%-{dates[-1][1]:.0f}%], "
                          f"best={dates[0][0]} ({dates[0][1]:.1f}%)")

                    best_date = dates[0][0]

                    result = fetch_des_data(
                        date=best_date,
                        coords=coords,
                        cloud_threshold=config.cloud_threshold,
                    )

                    # Save RGB preview
                    try:
                        from PIL import Image as _PILImage
                        rgb_u8 = (result.rgb * 255).clip(0, 255).astype(np.uint8)
                        _PILImage.fromarray(rgb_u8).save(
                            tiles_dir / f"preview_{cell_key}_{best_date}.png")
                    except Exception:
                        pass

                    # Check we have all Prithvi bands
                    missing = [b for b in config.prithvi_bands
                               if b not in result.bands]
                    if missing:
                        print(f"    {cell_key}: SKIP bands — missing {missing} "
                              f"(have: {sorted(result.bands.keys())})")
                        continue

                    # Stack Prithvi bands
                    image = np.stack(
                        [result.bands[b] for b in config.prithvi_bands],
                        axis=0,
                    ).astype(np.float32)  # (6, H, W) reflectance [0,1]

                    # Fetch NMD — hits disk cache from pre-filter,
                    # but now with target_shape to match spectral image
                    nmd_result = fetch_nmd_data(
                        coords=coords,
                        target_shape=image.shape[1:],
                    )

                    # Convert to LULC labels
                    labels = nmd_raster_to_lulc(
                        nmd_result.nmd_raster,
                        num_classes=config.num_classes,
                    )
                    print(f"    NMD (cached): {nmd_result.nmd_raster.shape} → "
                          f"labels {labels.shape}, "
                          f"classes={np.unique(labels).tolist()}")

                    # Save tile
                    np.savez_compressed(
                        tiles_dir / tile_name,
                        image=image,
                        label=labels,
                        easting=cell.easting,
                        northing=cell.northing,
                        date=best_date,
                        lat=cell.center_lat,
                        lon=cell.center_lon,
                    )

                    # Update class counts and per-tile histogram
                    tile_hist = {}
                    for cls_idx in range(config.num_classes + 1):
                        count = int((labels == cls_idx).sum())
                        class_counts[cls_idx] = class_counts.get(cls_idx, 0) + count
                        if count > 0:
                            tile_hist[cls_idx] = count
                    tile_histograms[tile_name] = tile_hist

                    tile_names_by_split[split_name].append(tile_name)
                    success = True
                    tile_idx += 1

                    # Update prepare log with latest tile info
                    prep_log["latest_tile"] = tile_name
                    prep_log["latest_date"] = best_date
                    prep_log["latest_cloud"] = round(result.cloud_fraction, 4)

                    print(f"    ✓ [{tile_idx}] {tile_name} ({best_date}, "
                          f"cloud={result.cloud_fraction:.1%})")

                    break  # Success, no need to try other years

                except (FetchError, Exception) as e:
                    print(f"    {cell_key} {year}: {type(e).__name__}: {e}")
                    continue

            # Track progress
            if success:
                completed.add(cell_key)
            else:
                failed.add(cell_key)

            # Update prepare log for dashboard
            prep_log["completed"] = len(completed)
            prep_log["failed"] = len(failed)
            prep_log["tiles_saved"] = tile_idx
            prep_log["class_counts"] = {str(k): v for k, v in class_counts.items()}
            prep_log["elapsed_s"] = round(time.time() - t_start, 1)
            prep_log["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_prepare_log(prep_log_path, prep_log)

            # Save progress periodically
            if (len(completed) + len(failed)) % 50 == 0:
                _save_progress(progress_path, completed, failed)

    # ── Step 6: Save progress, splits, and stats ──────────────────────
    _save_progress(progress_path, completed, failed)

    for split_name, names in tile_names_by_split.items():
        split_path = data_dir / f"split_{split_name}.txt"
        with open(split_path, "w") as f:
            f.write("\n".join(sorted(names)) + "\n")
        print(f"  {split_name}: {len(names)} tiles → {split_path}")

    # Compute tile weights for rare-class oversampling
    tile_weights = _compute_tile_weights(
        tile_histograms, class_counts,
        num_classes=config.num_classes,
        rare_threshold=config.rare_class_threshold,
        max_weight=config.max_tile_weight,
        ignore_index=config.ignore_index,
    )

    # Save class statistics
    stats = {
        "class_counts": class_counts,
        "tile_weights": tile_weights,
        "rare_classes": _find_rare_classes(
            class_counts, config.num_classes,
            config.rare_class_threshold, config.ignore_index,
        ),
        "total_tiles": tile_idx,
        "num_classes": config.num_classes,
    }
    stats_path = data_dir / "class_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Class stats → {stats_path}")

    # Report rare classes
    if stats["rare_classes"]:
        print(f"  Rare classes (<{config.rare_class_threshold:.0%} of pixels):")
        for cls_idx in stats["rare_classes"]:
            total_px = sum(class_counts.values())
            frac = class_counts.get(cls_idx, 0) / max(total_px, 1)
            print(f"    class {cls_idx:2d}: {frac:.4%}")
        n_boosted = sum(1 for w in tile_weights.values() if w > 1.01)
        print(f"  Tiles with boosted weight: {n_boosted}/{len(tile_weights)}")

    # Final prepare log update
    prep_log["status"] = "completed"
    prep_log["completed"] = len(completed)
    prep_log["failed"] = len(failed)
    prep_log["tiles_saved"] = tile_idx
    prep_log["class_counts"] = {str(k): v for k, v in class_counts.items()}
    prep_log["elapsed_s"] = round(time.time() - t_start, 1)
    prep_log["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_prepare_log(prep_log_path, prep_log)

    print(f"\n  Done! {tile_idx} tiles saved to {tiles_dir}")


def _write_prepare_log(path: Path, log: dict) -> None:
    """Atomic write of prepare_log.json for dashboard polling."""
    try:
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(log, f, indent=2)
        tmp.rename(path)
    except Exception:
        pass


def _load_progress(path: Path) -> dict:
    """Load progress checkpoint."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def _save_progress(path: Path, completed: set, failed: set) -> None:
    """Save progress checkpoint."""
    with open(path, "w") as f:
        json.dump({
            "completed": sorted(completed),
            "failed": sorted(failed),
        }, f)


def _find_rare_classes(
    class_counts: dict[int, int],
    num_classes: int,
    threshold: float,
    ignore_index: int = 0,
) -> list[int]:
    """Identify classes whose pixel fraction is below threshold.

    Args:
        class_counts: Dict mapping class index to total pixel count.
        num_classes: Number of classes.
        threshold: Fraction below which a class is considered rare.
        ignore_index: Background class to exclude.

    Returns:
        List of rare class indices.
    """
    total = sum(v for k, v in class_counts.items() if k != ignore_index)
    if total == 0:
        return []

    rare = []
    for cls_idx in range(num_classes + 1):
        if cls_idx == ignore_index:
            continue
        frac = class_counts.get(cls_idx, 0) / total
        if frac < threshold:
            rare.append(cls_idx)
    return rare


def _compute_tile_weights(
    tile_histograms: dict[str, dict[int, int]],
    class_counts: dict[int, int],
    num_classes: int,
    rare_threshold: float = 0.02,
    max_weight: float = 5.0,
    ignore_index: int = 0,
) -> dict[str, float]:
    """Compute per-tile sampling weights based on rare-class content.

    Tiles containing pixels from rare classes get higher weights so
    they are sampled more frequently during training.

    Algorithm:
        1. Find rare classes (global pixel fraction < threshold)
        2. For each tile, compute rarity_score = fraction of pixels
           belonging to rare classes
        3. weight = 1.0 + (max_weight - 1.0) * rarity_score

    Args:
        tile_histograms: Per-tile class pixel counts.
        class_counts: Global class pixel counts.
        num_classes: Number of classes.
        rare_threshold: Classes below this fraction are "rare".
        max_weight: Maximum tile weight.
        ignore_index: Background class to exclude.

    Returns:
        Dict mapping tile name to sampling weight.
    """
    rare_classes = set(_find_rare_classes(
        class_counts, num_classes, rare_threshold, ignore_index,
    ))

    if not rare_classes:
        # No rare classes → uniform weights
        return {name: 1.0 for name in tile_histograms}

    weights = {}
    for tile_name, hist in tile_histograms.items():
        total_px = sum(v for k, v in hist.items() if k != ignore_index)
        if total_px == 0:
            weights[tile_name] = 1.0
            continue

        rare_px = sum(v for k, v in hist.items() if k in rare_classes)
        rarity_score = rare_px / total_px

        w = 1.0 + (max_weight - 1.0) * rarity_score
        weights[tile_name] = round(w, 3)

    return weights
