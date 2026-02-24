"""
imint/training/prepare_data.py — Fetch and cache LULC training data

Orchestrates bulk fetching of Sentinel-2 + NMD tile pairs from DES,
with progress tracking for resumability.
"""
from __future__ import annotations

import json
import os
import traceback
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

    # ── Step 4: Connect to DES ────────────────────────────────────────
    print("\n  Connecting to DES...")
    conn = _connect()

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

    for split_name, split_cells in all_splits:
        print(f"\n  Processing {split_name} split ({len(split_cells)} cells)...")

        for i, cell in enumerate(split_cells):
            cell_key = f"{cell.easting}_{cell.northing}"

            if cell_key in completed:
                # Find existing tile name
                tile_name = f"tile_{cell_key}.npz"
                if (tiles_dir / tile_name).exists():
                    tile_names_by_split[split_name].append(tile_name)
                continue

            if cell_key in failed:
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
                        max_cloud_pct=30,
                    )
                    if not dates:
                        continue

                    # Try best (lowest cloud) date
                    best_date = dates[0][0]

                    result = fetch_des_data(
                        date=best_date,
                        coords=coords,
                        cloud_threshold=config.cloud_threshold,
                        token=conn.authenticate_oidc_device
                        if hasattr(conn, "authenticate_oidc_device") else None,
                    )

                    # Check we have all Prithvi bands
                    if not all(b in result.bands for b in config.prithvi_bands):
                        continue

                    # Stack Prithvi bands
                    image = np.stack(
                        [result.bands[b] for b in config.prithvi_bands],
                        axis=0,
                    ).astype(np.float32)  # (6, H, W) reflectance [0,1]

                    # Fetch NMD
                    nmd_result = fetch_nmd_data(
                        coords=coords,
                        target_shape=image.shape[1:],
                    )

                    # Convert to LULC labels
                    labels = nmd_raster_to_lulc(
                        nmd_result.nmd_raster,
                        num_classes=config.num_classes,
                    )

                    # Skip if >95% water/background
                    land_frac = np.mean((labels > 0) & (labels != 18) & (labels != 19))
                    if land_frac < 0.05:
                        continue

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

                    if tile_idx % 10 == 0:
                        print(f"    [{tile_idx}] {tile_name} ({best_date}, "
                              f"cloud={result.cloud_fraction:.1%})")

                    break  # Success, no need to try other years

                except (FetchError, Exception) as e:
                    if "cloud" not in str(e).lower():
                        print(f"    WARNING: {cell_key} {year}: {e}")
                    continue

            # Track progress
            if success:
                completed.add(cell_key)
            else:
                failed.add(cell_key)

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

    print(f"\n  Done! {tile_idx} tiles saved to {tiles_dir}")


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
