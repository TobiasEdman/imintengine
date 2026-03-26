#!/usr/bin/env python3
"""
scripts/build_crop_dataset_v3.py — Build year-balanced AND class-balanced crop dataset

Pipeline:
  1. Load LUCAS-SE 2018+2022 points (field-verified ground truth)
  2. Load LPIS shapefiles for 2022, 2023, 2024 (zipped shapefiles)
  3. Load LPIS 2025 from GML format
  4. Map SJV grödkoder → crop_schema 9 classes
  5. Cross-validate LUCAS 2022 vs LPIS 2022 (spatial join + confusion matrix)
  6. Build year-balanced AND class-balanced dataset:
     - Target: ~300 pts/year, ~40 pts/class/year
     - LUCAS = high confidence backbone
     - LPIS matching LUCAS = high confidence
     - Other LPIS = medium confidence
     - Cap majority classes, keep all minority points
  7. Export balanced_points_v3.json with validation metadata

LPIS source: Jordbruksverket open data (CC BY 4.0)
  Shapefiles: data/lpis/jordbruksskiften_YEAR.zip (EPSG:3006)
  GML 2025:   data/lpis/jordbruksskiften_2025.gml.zip

Usage:
    python scripts/build_crop_dataset_v3.py \\
        --lucas-csv data/lucas/LUCAS_2018_Copernicus_attributes.csv \\
                    data/lucas/EU_LUCAS_2022.csv \\
        --lpis-dir data/lpis \\
        --output data/lucas/balanced_points_v3.json

    # Validate only (no dataset export)
    python scripts/build_crop_dataset_v3.py \\
        --lucas-csv data/lucas/EU_LUCAS_2022.csv \\
        --lpis-dir data/lpis \\
        --validate-only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Project imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.crop_schema import (
    load_lucas_sweden,
    summarize_lucas_sweden,
    CLASS_NAMES,
    CLASS_NAMES_SV,
    NUM_CLASSES,
    lucas_code_to_class,
)
from scripts.build_crop_dataset import SJV_TO_CROP

# ── Constants ─────────────────────────────────────────────────────────────

# Years we expect to have data for
LPIS_SHAPEFILE_YEARS = [2022, 2023, 2024]
LPIS_GML_YEAR = 2025
LUCAS_YEARS = [2018, 2022]
ALL_YEARS = [2018, 2022, 2023, 2024, 2025]

# Target budget per year and per class
TARGET_PER_YEAR = 300
TARGET_PER_CLASS_PER_YEAR = 40

# LPIS column names (Jordbruksverket standard)
COL_GRODKOD = "grdkod_mar"
COL_BLOCKID = "blockid"
COL_SKIFTE = "skiftesbeteckning"
COL_AREAL = "faststalld_areal_ha"
COL_ARSLAGER = "arslager"


def sjv_grodkod_to_class(grodkod: int) -> int:
    """Map SJV grödkod to crop_schema class index (0 = unmapped)."""
    return SJV_TO_CROP.get(grodkod, 0)


# ── LPIS loading ──────────────────────────────────────────────────────────

def load_lpis_shapefile(zip_path: str, year: int) -> "gpd.GeoDataFrame":
    """Load an LPIS shapefile from a zip archive.

    Args:
        zip_path: Path to jordbruksskiften_YEAR.zip
        year: The crop year (used for tagging).

    Returns:
        GeoDataFrame with columns: geometry, crop_class, crop_name,
        grodkod, blockid, skifte, areal_ha, year.
    """
    import geopandas as gpd

    abs_path = str(Path(zip_path).resolve())
    print(f"  Loading shapefile: {abs_path}")

    # Try multiple read strategies (GDAL/fiona can be picky about paths)
    gdf = None
    for uri in [abs_path, f"zip://{abs_path}"]:
        try:
            gdf = gpd.read_file(uri)
            print(f"    Read via: {uri[:60]}...")
            break
        except Exception:
            continue

    if gdf is None:
        # Fallback: extract to temp dir and read shapefile directly
        import zipfile
        import tempfile

        print("    Fallback: extracting zip to temp directory...")
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(abs_path, "r") as zf:
                zf.extractall(tmpdir)
            shp_files = list(Path(tmpdir).rglob("*.shp"))
            if not shp_files:
                raise FileNotFoundError(
                    f"No .shp found inside {zip_path}"
                )
            gdf = gpd.read_file(str(shp_files[0]))

    print(f"    Raw features: {len(gdf)}")
    return _process_lpis_gdf(gdf, year)


def load_lpis_gml(zip_path: str, year: int) -> "gpd.GeoDataFrame":
    """Load LPIS GML from a zip archive (2025 format).

    Args:
        zip_path: Path to jordbruksskiften_2025.gml.zip
        year: The crop year.

    Returns:
        GeoDataFrame with same schema as load_lpis_shapefile.
    """
    import geopandas as gpd
    import zipfile
    import tempfile

    abs_path = str(Path(zip_path).resolve())
    print(f"  Loading GML: {abs_path}")

    # Extract GML from zip to a temp dir, then load
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(abs_path, "r") as zf:
            zf.extractall(tmpdir)
            # Find the GML file inside
            gml_files = list(Path(tmpdir).rglob("*.gml"))
            if not gml_files:
                print(f"    WARNING: No .gml file found inside {abs_path}")
                return gpd.GeoDataFrame()
            gml_path = gml_files[0]
            print(f"    Found GML: {gml_path.name}")
            gdf = gpd.read_file(str(gml_path))

    print(f"    Raw features: {len(gdf)}")
    return _process_lpis_gdf(gdf, year)


def _process_lpis_gdf(gdf: "gpd.GeoDataFrame", year: int) -> "gpd.GeoDataFrame":
    """Process raw LPIS GeoDataFrame: map grödkoder, compute centroids.

    Columns may vary slightly between years, so we normalize column names
    to lowercase and look up known column patterns.
    """
    import geopandas as gpd

    if gdf.empty:
        return gdf

    # Normalize column names to lowercase for robustness
    gdf.columns = [c.lower() for c in gdf.columns]

    # Find the grödkod column (may be grdkod_mar, grodkod, etc.)
    grodkod_col = None
    for candidate in [COL_GRODKOD, "grodkod", "grdkod", "grodkod_mar"]:
        if candidate in gdf.columns:
            grodkod_col = candidate
            break

    if grodkod_col is None:
        print(f"    WARNING: No grödkod column found. Columns: {list(gdf.columns)}")
        return gpd.GeoDataFrame()

    # Drop rows without grödkod
    gdf = gdf.dropna(subset=[grodkod_col])

    # Map grödkod to crop class
    gdf["grodkod_int"] = gdf[grodkod_col].astype(int)
    gdf["crop_class"] = gdf["grodkod_int"].apply(sjv_grodkod_to_class)

    # Drop unmapped classes (class 0 = not agricultural or not in schema)
    gdf = gdf[gdf["crop_class"] > 0].copy()
    print(f"    After mapping: {len(gdf)} parcels with valid crop class")

    # Add metadata columns
    gdf["crop_name"] = gdf["crop_class"].map(CLASS_NAMES)
    gdf["year"] = year

    # Normalize optional columns
    if COL_BLOCKID not in gdf.columns:
        gdf[COL_BLOCKID] = ""
    if COL_SKIFTE not in gdf.columns:
        gdf[COL_SKIFTE] = ""
    if COL_AREAL not in gdf.columns:
        gdf[COL_AREAL] = 0.0

    # Ensure CRS is EPSG:3006 (SWEREF99 TM)
    if gdf.crs is None:
        print("    WARNING: No CRS detected, assuming EPSG:3006")
        gdf = gdf.set_crs("EPSG:3006")
    elif gdf.crs.to_epsg() != 3006:
        print(f"    Reprojecting from {gdf.crs} to EPSG:3006")
        gdf = gdf.to_crs("EPSG:3006")

    return gdf


def lpis_gdf_to_points(gdf: "gpd.GeoDataFrame") -> list[dict]:
    """Convert LPIS GeoDataFrame to list of point dicts (centroid in WGS84).

    Each point dict is compatible with the balanced_points format.
    """
    import pyproj

    if gdf.empty:
        return []

    transformer = pyproj.Transformer.from_crs(
        "EPSG:3006", "EPSG:4326", always_xy=True
    )

    points = []
    for _, row in gdf.iterrows():
        centroid = row.geometry.centroid
        lon, lat = transformer.transform(centroid.x, centroid.y)

        grodkod = int(row["grodkod_int"])
        blockid = str(row.get(COL_BLOCKID, ""))
        skifte = str(row.get(COL_SKIFTE, ""))
        areal = float(row.get(COL_AREAL, 0)) if row.get(COL_AREAL) is not None else 0.0

        points.append({
            "point_id": f"LPIS_{row['year']}_{blockid}_{skifte}_{grodkod}",
            "lat": lat,
            "lon": lon,
            "lc_code": f"SJV_{grodkod}",
            "crop_class": int(row["crop_class"]),
            "crop_name": row["crop_name"],
            "year": int(row["year"]),
            "nuts0": "SE",
            "source": "lpis",
            "confidence": "medium",
            "sjv_grodkod": grodkod,
            "blockid": blockid,
            "areal_ha": areal,
        })

    return points


def load_all_lpis(lpis_dir: str) -> dict[int, list[dict]]:
    """Load all available LPIS data, organized by year.

    Checks for:
      - data/lpis/jordbruksskiften_YEAR.zip for 2022, 2023, 2024
      - data/lpis/jordbruksskiften_2025.gml.zip for 2025

    Skips missing files gracefully.

    Returns:
        Dict mapping year -> list of point dicts.
    """
    lpis_by_year: dict[int, list[dict]] = {}

    # Shapefile years (2022, 2023, 2024)
    for year in LPIS_SHAPEFILE_YEARS:
        zip_path = os.path.join(lpis_dir, f"jordbruksskiften_{year}.zip")
        if not os.path.exists(zip_path):
            print(f"  SKIP: {zip_path} not found")
            continue

        try:
            gdf = load_lpis_shapefile(zip_path, year)
            points = lpis_gdf_to_points(gdf)
            lpis_by_year[year] = points
            print(f"    Year {year}: {len(points)} points")
        except Exception as e:
            print(f"    ERROR loading {zip_path}: {e}")

    # GML year (2025)
    gml_path = os.path.join(lpis_dir, f"jordbruksskiften_{LPIS_GML_YEAR}.gml.zip")
    if os.path.exists(gml_path):
        try:
            gdf = load_lpis_gml(gml_path, LPIS_GML_YEAR)
            points = lpis_gdf_to_points(gdf)
            lpis_by_year[LPIS_GML_YEAR] = points
            print(f"    Year {LPIS_GML_YEAR}: {len(points)} points")
        except Exception as e:
            print(f"    ERROR loading {gml_path}: {e}")
    else:
        print(f"  SKIP: {gml_path} not found")

    return lpis_by_year


# ── Cross-validation (spatial join) ──────────────────────────────────────

def cross_validate_lucas_lpis(
    lucas_points: list[dict],
    lpis_gdf: "gpd.GeoDataFrame",
) -> dict:
    """Cross-validate LUCAS 2022 points against LPIS 2022 using spatial join.

    For each LUCAS point, find which LPIS polygon it falls within.
    Compare the crop classes. Output confusion matrix.

    Args:
        lucas_points: LUCAS points (year 2022) as list of dicts.
        lpis_gdf: LPIS 2022 GeoDataFrame (polygons in EPSG:3006).

    Returns:
        Dict with confusion matrix, accuracy, per-class metrics.
    """
    import geopandas as gpd
    import pyproj
    from shapely.geometry import Point

    if lpis_gdf.empty:
        return {"error": "No LPIS polygons for cross-validation"}

    # Filter LUCAS to 2022 only
    lucas_2022 = [p for p in lucas_points if p["year"] == 2022]
    if not lucas_2022:
        return {"error": "No LUCAS 2022 points for cross-validation"}

    print(f"    LUCAS 2022 points: {len(lucas_2022)}")
    print(f"    LPIS 2022 polygons: {len(lpis_gdf)}")

    # Create LUCAS GeoDataFrame in EPSG:3006
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3006", always_xy=True
    )

    geometries = []
    for p in lucas_2022:
        x, y = transformer.transform(p["lon"], p["lat"])
        geometries.append(Point(x, y))

    lucas_gdf = gpd.GeoDataFrame(
        lucas_2022,
        geometry=geometries,
        crs="EPSG:3006",
    )

    # Spatial join: find which LPIS polygon each LUCAS point falls within
    joined = gpd.sjoin(lucas_gdf, lpis_gdf, how="inner", predicate="within")

    print(f"    Spatial matches: {len(joined)} of {len(lucas_2022)} LUCAS points")

    if joined.empty:
        return {
            "matches": 0,
            "total_lucas_2022": len(lucas_2022),
            "total_lpis_2022": len(lpis_gdf),
            "error": "No spatial matches found",
        }

    # Compare crop classes
    # lucas crop_class is in crop_class_left, lpis is in crop_class_right
    confusion = defaultdict(lambda: defaultdict(int))
    correct = 0

    for _, row in joined.iterrows():
        lucas_cls = int(row["crop_class_left"])
        lpis_cls = int(row["crop_class_right"])
        confusion[lucas_cls][lpis_cls] += 1
        if lucas_cls == lpis_cls:
            correct += 1

    n_matches = len(joined)
    accuracy = correct / max(n_matches, 1)

    # Per-class precision and recall
    per_class = {}
    for cls_idx in range(1, NUM_CLASSES):
        # True positives
        tp = confusion.get(cls_idx, {}).get(cls_idx, 0)
        # All LUCAS points of this class that were matched
        lucas_total = sum(confusion.get(cls_idx, {}).values())
        # All LPIS predictions for this class
        lpis_total = sum(
            confusion.get(other, {}).get(cls_idx, 0)
            for other in confusion
        )

        recall = tp / max(lucas_total, 1)
        precision = tp / max(lpis_total, 1)

        per_class[CLASS_NAMES[cls_idx]] = {
            "tp": tp,
            "lucas_total": lucas_total,
            "lpis_predicted": lpis_total,
            "recall": round(recall, 3),
            "precision": round(precision, 3),
        }

    # Serialize confusion matrix (int keys -> str for JSON)
    confusion_serializable = {
        CLASS_NAMES.get(k, str(k)): {
            CLASS_NAMES.get(k2, str(k2)): v2
            for k2, v2 in sorted(v.items())
        }
        for k, v in sorted(confusion.items())
    }

    return {
        "matches": n_matches,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "total_lucas_2022": len(lucas_2022),
        "total_lpis_2022": len(lpis_gdf),
        "confusion": confusion_serializable,
        "per_class": per_class,
    }


def print_confusion_matrix(validation: dict) -> None:
    """Pretty-print the confusion matrix from cross-validation results."""
    confusion = validation.get("confusion", {})
    if not confusion:
        return

    print("\n    Confusion matrix (rows=LUCAS truth, cols=LPIS predicted):")
    # Get all class names that appear
    all_classes = sorted(
        set(list(confusion.keys()) + [
            c for row in confusion.values() for c in row.keys()
        ])
    )

    # Header
    header = f"    {'':20s}"
    for cls_name in all_classes:
        short = cls_name[:8]
        header += f" {short:>8s}"
    print(header)
    print("    " + "-" * (20 + 9 * len(all_classes)))

    # Rows
    for row_cls in all_classes:
        row_data = confusion.get(row_cls, {})
        line = f"    {row_cls:20s}"
        for col_cls in all_classes:
            count = row_data.get(col_cls, 0)
            if count > 0:
                line += f" {count:8d}"
            else:
                line += f" {'·':>8s}"
        print(line)


# ── Confidence tagging ────────────────────────────────────────────────────

def tag_lpis_confidence(
    lpis_points: list[dict],
    matched_lpis_ids: set[str],
) -> list[dict]:
    """Tag LPIS points with confidence based on LUCAS cross-validation.

    Points whose polygons matched a LUCAS point with the same crop class
    get confidence='high'. Others get confidence='medium'.
    """
    for p in lpis_points:
        if p["point_id"] in matched_lpis_ids:
            p["confidence"] = "high"
        else:
            p["confidence"] = "medium"
    return lpis_points


# ── Year + class balanced sampling ────────────────────────────────────────

def build_balanced_dataset(
    lucas_points: list[dict],
    lpis_by_year: dict[int, list[dict]],
    target_per_year: int = TARGET_PER_YEAR,
    target_per_class_per_year: int = TARGET_PER_CLASS_PER_YEAR,
    seed: int = 42,
) -> list[dict]:
    """Build year-balanced AND class-balanced dataset.

    Strategy:
      1. LUCAS points always included (confidence="high", source="lucas")
      2. For each year, balance across 8 crop classes (1-8)
      3. Cap majority classes at target_per_class_per_year
      4. Keep ALL minority class points (no dropping)
      5. Fill gaps from LPIS where LUCAS is insufficient

    Args:
        lucas_points: All LUCAS points (tagged with year, source, confidence).
        lpis_by_year: LPIS points organized by year.
        target_per_year: Target total points per year (~300).
        target_per_class_per_year: Target points per class per year (~40).
        seed: Random seed.

    Returns:
        Balanced list of point dicts.
    """
    rng = random.Random(seed)

    # Tag LUCAS points
    for p in lucas_points:
        p["source"] = "lucas"
        p["confidence"] = "high"

    # Organize LUCAS by year
    lucas_by_year: dict[int, list[dict]] = defaultdict(list)
    for p in lucas_points:
        lucas_by_year[p["year"]].append(p)

    balanced = []
    year_summaries = {}

    for year in ALL_YEARS:
        print(f"\n  Year {year}:")

        # LUCAS points for this year
        year_lucas = lucas_by_year.get(year, [])

        # LPIS points for this year
        year_lpis = lpis_by_year.get(year, [])

        # Organize by class
        lucas_by_class: dict[int, list[dict]] = defaultdict(list)
        for p in year_lucas:
            if p["crop_class"] > 0:
                lucas_by_class[p["crop_class"]].append(p)

        lpis_by_class: dict[int, list[dict]] = defaultdict(list)
        for p in year_lpis:
            if p["crop_class"] > 0:
                lpis_by_class[p["crop_class"]].append(p)

        year_points = []
        year_summary = {}

        for cls_idx in range(1, NUM_CLASSES):
            cls_lucas = lucas_by_class.get(cls_idx, [])
            cls_lpis = lpis_by_class.get(cls_idx, [])

            # Always include all LUCAS points
            cls_points = list(cls_lucas)
            n_lucas = len(cls_points)

            # Fill with LPIS to reach target
            n_need = max(0, target_per_class_per_year - n_lucas)
            if n_need > 0 and cls_lpis:
                supplement = rng.sample(cls_lpis, min(n_need, len(cls_lpis)))
                cls_points.extend(supplement)
                n_lpis_added = len(supplement)
            else:
                n_lpis_added = 0

            # Cap majority classes (but never drop LUCAS points)
            cap = target_per_class_per_year
            if len(cls_points) > cap:
                # Separate by source, prefer LUCAS
                lucas_in = [p for p in cls_points if p.get("source") == "lucas"]
                lpis_in = [p for p in cls_points if p.get("source") == "lpis"]

                if len(lucas_in) >= cap:
                    # Even LUCAS alone exceeds cap — sample LUCAS
                    cls_points = rng.sample(lucas_in, cap)
                else:
                    # Keep all LUCAS, sample LPIS to fill remaining
                    remaining = cap - len(lucas_in)
                    cls_points = lucas_in + rng.sample(
                        lpis_in, min(remaining, len(lpis_in))
                    )

            year_points.extend(cls_points)

            total = len(cls_points)
            year_summary[CLASS_NAMES[cls_idx]] = {
                "lucas": n_lucas,
                "lpis_added": n_lpis_added,
                "total": total,
            }

            status = ""
            if total < target_per_class_per_year:
                status = " (below target)"
            print(
                f"    {CLASS_NAMES[cls_idx]:20s}: "
                f"{n_lucas:3d} LUCAS + {n_lpis_added:3d} LPIS = {total:3d}{status}"
            )

        balanced.extend(year_points)
        year_summaries[year] = {
            "total": len(year_points),
            "classes": year_summary,
        }
        print(f"    Year {year} total: {len(year_points)} points")

    return balanced, year_summaries


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build year-balanced AND class-balanced crop dataset (v3)"
    )
    parser.add_argument(
        "--lucas-csv", nargs="+", required=True,
        help="LUCAS CSV files (2018 and/or 2022)",
    )
    parser.add_argument(
        "--lpis-dir", default="data/lpis",
        help="Directory containing LPIS zip files (default: data/lpis)",
    )
    parser.add_argument(
        "--output", default="data/lucas/balanced_points_v3.json",
        help="Output JSON path (default: data/lucas/balanced_points_v3.json)",
    )
    parser.add_argument(
        "--target-per-year", type=int, default=TARGET_PER_YEAR,
        help=f"Target points per year (default: {TARGET_PER_YEAR})",
    )
    parser.add_argument(
        "--target-per-class", type=int, default=TARGET_PER_CLASS_PER_YEAR,
        help=f"Target points per class per year (default: {TARGET_PER_CLASS_PER_YEAR})",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only cross-validate LUCAS vs LPIS, don't build dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("build_crop_dataset_v3 — Year + class balanced crop training dataset")
    print("=" * 70)

    # ── 1. Load LUCAS ─────────────────────────────────────────────────
    print("\n[1/6] Loading LUCAS-SE points...")
    csv_paths = args.lucas_csv if len(args.lucas_csv) > 1 else args.lucas_csv[0]
    lucas_points = load_lucas_sweden(csv_paths, crop_only=True)
    lucas_summary = summarize_lucas_sweden(lucas_points)
    print(f"  Total LUCAS: {lucas_summary['total']} points")

    # By year
    lucas_years = Counter(p["year"] for p in lucas_points)
    for yr, count in sorted(lucas_years.items()):
        print(f"    {yr}: {count} points")

    # By class
    for cls, count in sorted(lucas_summary["per_class"].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {cls:20s}: {count}")

    # ── 2. Load LPIS ──────────────────────────────────────────────────
    print("\n[2/6] Loading LPIS shapefiles...")
    lpis_dir = args.lpis_dir

    if not os.path.isdir(lpis_dir):
        print(f"  WARNING: LPIS directory not found: {lpis_dir}")
        print("  Continuing with LUCAS only.")
        lpis_by_year: dict[int, list[dict]] = {}
        lpis_gdfs: dict[int, object] = {}
    else:
        # We need GeoDataFrames for cross-validation and points for balancing
        import geopandas as gpd

        lpis_gdfs = {}
        lpis_by_year = {}

        for year in LPIS_SHAPEFILE_YEARS:
            zip_path = os.path.join(lpis_dir, f"jordbruksskiften_{year}.zip")
            if not os.path.exists(zip_path):
                print(f"  SKIP: {zip_path} not found")
                continue
            try:
                gdf = load_lpis_shapefile(zip_path, year)
                lpis_gdfs[year] = gdf
                points = lpis_gdf_to_points(gdf)
                lpis_by_year[year] = points
                print(f"    Year {year}: {len(points)} crop parcel centroids")
            except Exception as e:
                print(f"    ERROR loading year {year}: {e}")

        # GML year (2025)
        gml_path = os.path.join(lpis_dir, f"jordbruksskiften_{LPIS_GML_YEAR}.gml.zip")
        if os.path.exists(gml_path):
            try:
                gdf = load_lpis_gml(gml_path, LPIS_GML_YEAR)
                lpis_gdfs[LPIS_GML_YEAR] = gdf
                points = lpis_gdf_to_points(gdf)
                lpis_by_year[LPIS_GML_YEAR] = points
                print(f"    Year {LPIS_GML_YEAR}: {len(points)} crop parcel centroids")
            except Exception as e:
                print(f"    ERROR loading year {LPIS_GML_YEAR}: {e}")
        else:
            print(f"  SKIP: {gml_path} not found")

    # Summary
    total_lpis = sum(len(pts) for pts in lpis_by_year.values())
    print(f"\n  Total LPIS points: {total_lpis} across {len(lpis_by_year)} years")

    # ── 3. Cross-validate LUCAS 2022 vs LPIS 2022 ────────────────────
    validation_result = None
    matched_lpis_ids: set[str] = set()

    print("\n[3/6] Cross-validating LUCAS 2022 vs LPIS 2022...")

    if 2022 in lpis_gdfs and not lpis_gdfs[2022].empty:
        validation_result = cross_validate_lucas_lpis(lucas_points, lpis_gdfs[2022])

        if "error" not in validation_result:
            print(f"    Matches: {validation_result['matches']} / "
                  f"{validation_result['total_lucas_2022']} LUCAS points")
            print(f"    Overall accuracy: {validation_result['accuracy']:.1%}")

            # Per-class metrics
            print("\n    Per-class metrics:")
            for cls_name, metrics in sorted(validation_result.get("per_class", {}).items()):
                print(
                    f"      {cls_name:20s}: "
                    f"recall={metrics['recall']:.1%}  "
                    f"precision={metrics['precision']:.1%}  "
                    f"(n={metrics['lucas_total']})"
                )

            print_confusion_matrix(validation_result)

            # Tag LPIS 2022 points that spatially match LUCAS with same class
            # as high confidence
            if 2022 in lpis_by_year:
                _tag_matching_lpis(lucas_points, lpis_by_year[2022], lpis_gdfs[2022])
        else:
            print(f"    {validation_result['error']}")
    else:
        print("    SKIP: No LPIS 2022 data available for cross-validation")

    if args.validate_only:
        print("\n[DONE] Validation only — skipping dataset build.")
        return

    # ── 4. Confidence tagging ─────────────────────────────────────────
    print("\n[4/6] Tagging point confidence levels...")

    n_high_lucas = sum(1 for p in lucas_points if p.get("source") == "lucas")
    n_high_lpis = 0
    n_medium_lpis = 0
    for year_pts in lpis_by_year.values():
        for p in year_pts:
            if p.get("confidence") == "high":
                n_high_lpis += 1
            else:
                n_medium_lpis += 1

    print(f"    LUCAS (high):        {n_high_lucas}")
    print(f"    LPIS matched (high): {n_high_lpis}")
    print(f"    LPIS other (medium): {n_medium_lpis}")

    # ── 5. Build balanced dataset ─────────────────────────────────────
    print("\n[5/6] Building year + class balanced dataset...")
    print(f"  Target: ~{args.target_per_year} pts/year, "
          f"~{args.target_per_class} pts/class/year")

    balanced, year_summaries = build_balanced_dataset(
        lucas_points,
        lpis_by_year,
        target_per_year=args.target_per_year,
        target_per_class_per_year=args.target_per_class,
        seed=args.seed,
    )

    # Final summary
    print(f"\n  Final dataset: {len(balanced)} points")
    sources = Counter(p.get("source", "unknown") for p in balanced)
    confidences = Counter(p.get("confidence", "unknown") for p in balanced)
    years = Counter(p["year"] for p in balanced)
    classes = Counter(p["crop_name"] for p in balanced)

    print(f"  Sources: {dict(sources)}")
    print(f"  Confidence: {dict(confidences)}")
    print(f"  Per year:")
    for yr in sorted(years.keys()):
        print(f"    {yr}: {years[yr]}")
    print(f"  Per class:")
    for cls, count in sorted(classes.items(), key=lambda x: -x[1]):
        print(f"    {cls:20s}: {count}")

    # ── 6. Export ─────────────────────────────────────────────────────
    print(f"\n[6/6] Exporting to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    output_data = {
        "metadata": {
            "version": "v3",
            "created": datetime.utcnow().isoformat() + "Z",
            "total": len(balanced),
            "sources": dict(sources),
            "confidences": dict(confidences),
            "years": {str(k): v for k, v in sorted(years.items())},
            "per_year_summary": {
                str(k): v for k, v in year_summaries.items()
            },
            "lucas_total": len(lucas_points),
            "lpis_total": total_lpis,
            "lpis_years_loaded": sorted(lpis_by_year.keys()),
            "target_per_year": args.target_per_year,
            "target_per_class_per_year": args.target_per_class,
            "class_names": CLASS_NAMES,
            "class_names_sv": CLASS_NAMES_SV,
            "num_classes": NUM_CLASSES,
            "sjv_grodkod_mapping": {str(k): v for k, v in SJV_TO_CROP.items()},
            "validation": validation_result,
            "seed": args.seed,
        },
        "points": balanced,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nSaved: {args.output} ({len(balanced)} points)")
    print("Done.")


def _tag_matching_lpis(
    lucas_points: list[dict],
    lpis_points: list[dict],
    lpis_gdf: "gpd.GeoDataFrame",
) -> None:
    """Tag LPIS points as high confidence if their polygon contains a LUCAS
    point with the same crop class.

    Modifies lpis_points in place.
    """
    import geopandas as gpd
    import pyproj
    from shapely.geometry import Point

    # Filter LUCAS to 2022
    lucas_2022 = [p for p in lucas_points if p["year"] == 2022]
    if not lucas_2022:
        return

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3006", always_xy=True
    )

    geometries = []
    for p in lucas_2022:
        x, y = transformer.transform(p["lon"], p["lat"])
        geometries.append(Point(x, y))

    lucas_gdf = gpd.GeoDataFrame(
        lucas_2022,
        geometry=geometries,
        crs="EPSG:3006",
    )

    # Spatial join
    joined = gpd.sjoin(lucas_gdf, lpis_gdf, how="inner", predicate="within")

    # Find LPIS polygons where crop classes match
    matching_indices = set()
    for _, row in joined.iterrows():
        lucas_cls = int(row["crop_class_left"])
        lpis_cls = int(row["crop_class_right"])
        if lucas_cls == lpis_cls:
            matching_indices.add(row["index_right"])

    # Build a set of blockid+skifte combos that matched
    matched_keys = set()
    for idx in matching_indices:
        if idx < len(lpis_gdf):
            r = lpis_gdf.iloc[idx]
            blockid = str(r.get(COL_BLOCKID, ""))
            skifte = str(r.get(COL_SKIFTE, ""))
            grodkod = int(r.get("grodkod_int", 0))
            matched_keys.add((blockid, skifte, grodkod))

    # Tag matching LPIS points
    n_tagged = 0
    for p in lpis_points:
        key = (p.get("blockid", ""), str(p.get("skifte", "")), p.get("sjv_grodkod", 0))
        # Also try matching by point_id components
        blockid = p.get("blockid", "")
        grodkod = p.get("sjv_grodkod", 0)

        # Check if this parcel's blockid+skifte+grodkod was matched
        for mk in matched_keys:
            if mk[0] == blockid and mk[2] == grodkod:
                p["confidence"] = "high"
                n_tagged += 1
                break

    print(f"    Tagged {n_tagged} LPIS points as high confidence (LUCAS-validated)")


if __name__ == "__main__":
    main()
