#!/usr/bin/env python3
"""
scripts/build_crop_dataset.py — Build balanced crop training dataset from LPIS + LUCAS

Pipeline:
  1. Load LUCAS-SE crop points (2018+2022) — field-verified ground truth
  2. Fetch LPIS blocks from Jordbruksverket WFS — all Swedish agricultural parcels
  3. Map SJV grödkoder → crop_schema 9 classes
  4. Cross-validate LPIS vs LUCAS (spatial join, confusion matrix)
  5. Build balanced dataset: LUCAS as backbone + LPIS to fill gaps (trindsäd, potatis, korn)
  6. Export balanced_points_v2.json

LPIS source: Jordbruksverket INSPIRE WFS (CC BY 4.0, no auth required)
             http://epub.sjv.se/inspire/inspire/wfs
             Layer: inspire:senaste_arslager_block

Usage:
    python scripts/build_crop_dataset.py \\
        --lucas-csv data/lucas/LUCAS_2018_Copernicus_attributes.csv \\
                    data/lucas/EU_LUCAS_2022.csv \\
        --output data/lucas/balanced_points_v2.json \\
        --lpis-cache data/lpis_cache

    # Validation only (no LPIS fetch, uses cached data)
    python scripts/build_crop_dataset.py \\
        --lucas-csv data/lucas/EU_LUCAS_2022.csv \\
        --lpis-cache data/lpis_cache \\
        --validate-only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imint.training.crop_schema import (
    load_lucas_sweden,
    summarize_lucas_sweden,
    CLASS_NAMES,
    CLASS_NAMES_SV,
    NUM_CLASSES,
    lucas_code_to_class,
)

# ── SJV grödkod → crop_schema mapping ────────────────────────────────────
# Based on Jordbruksverket grödkodslista
# https://jordbruksverket.se/stod/jordbruk-tradgard-och-rennaring/sam-ansokan-och-allmant-om-jordbrukarstoden/grodkoder

SJV_TO_CROP = {
    # ── Vete (klass 1) ───────────────────────────────────────────
    1: 1,    # Höstvete
    2: 1,    # Vårvete

    # ── Korn (klass 2) ───────────────────────────────────────────
    3: 2,    # Höstkorn
    4: 2,    # Vårkorn

    # ── Havre (klass 3) ──────────────────────────────────────────
    5: 3,    # Havre

    # ── Oljeväxter (klass 4) ─────────────────────────────────────
    85: 4,   # Höstraps
    86: 4,   # Vårraps
    87: 4,   # Höstrybs
    88: 4,   # Vårrybs
    90: 4,   # Solros
    91: 4,   # Oljelin
    92: 4,   # Övriga oljeväxter

    # ── Vall/grönfoder (klass 5) ─────────────────────────────────
    49: 5,   # Slåtter-/betesvall (ej godkänd)
    50: 5,   # Slåtter-/betesvall
    51: 5,   # Frövall
    52: 5,   # Betesvall
    80: 5,   # Grönfoder

    # ── Potatis (klass 6) ────────────────────────────────────────
    70: 6,   # Matpotatis
    71: 6,   # Stärkelsepotatis
    72: 6,   # Industripotatis

    # ── Trindsäd (klass 7) ───────────────────────────────────────
    30: 7,   # Ärtor (konserv)
    31: 7,   # Ärtor (foder)
    32: 7,   # Åkerbönor
    33: 7,   # Konservbönor
    34: 7,   # Övriga bönor
    35: 7,   # Bruna bönor
    40: 7,   # Sötlupin
    43: 7,   # Övriga baljväxter

    # ── Övrig åkergrödor (klass 8) ───────────────────────────────
    11: 8,   # Höstråg
    12: 8,   # Vårråg
    13: 8,   # Rågvete
    14: 8,   # Blandsäd
    22: 8,   # Sockerbetor
    73: 8,   # Jordgubbar
    74: 8,   # Grönsaker
    75: 8,   # Blommor
    76: 8,   # Bärplantering
    77: 8,   # Fruktträd
    93: 8,   # Hampa
    95: 8,   # Energiskog
}


def sjv_grodkod_to_class(grodkod: int) -> int:
    """Map SJV grödkod to crop_schema class index."""
    return SJV_TO_CROP.get(grodkod, 0)


# ── LPIS fetching ────────────────────────────────────────────────────────

def fetch_lpis_sample(
    n_per_class: int = 200,
    target_classes: list[int] | None = None,
    cache_dir: str = "data/lpis_cache",
    seed: int = 42,
) -> list[dict]:
    """Fetch LPIS blocks from SJV WFS, sample centroids per crop class.

    Fetches blocks across Sweden in regional tiles, maps grödkoder to
    crop classes, and samples up to n_per_class centroids per class.

    Args:
        n_per_class: Max points per crop class to sample.
        target_classes: Only fetch these classes (None = all).
        cache_dir: Cache directory for WFS responses.
        seed: Random seed for sampling.

    Returns:
        List of point dicts compatible with balanced_points format.
    """
    import geopandas as gpd
    import json as _json
    import urllib.request
    import urllib.parse
    import hashlib
    from pathlib import Path

    rng = random.Random(seed)

    # WFS endpoint for jordbruksskiften (parcels with grödkod)
    WFS_URL = "http://epub.sjv.se/inspire/inspire/wfs"
    WFS_LAYER = "inspire:senaste_arslager_skifte"

    # Regional bboxes covering Swedish agricultural regions (WGS84)
    REGIONS = [
        {"name": "Skåne",         "west": 12.8, "south": 55.3, "east": 14.5, "north": 56.3},
        {"name": "Halland",       "west": 12.0, "south": 56.3, "east": 13.5, "north": 57.3},
        {"name": "Småland",       "west": 14.5, "south": 56.5, "east": 16.5, "north": 57.8},
        {"name": "Östergötland",  "west": 15.0, "south": 58.0, "east": 16.8, "north": 58.8},
        {"name": "Västergötland", "west": 12.0, "south": 57.5, "east": 14.0, "north": 58.8},
        {"name": "Mälardalen",    "west": 15.5, "south": 58.8, "east": 18.5, "north": 60.0},
        {"name": "Dalarna",       "west": 14.0, "south": 60.0, "east": 17.0, "north": 61.5},
        {"name": "Norrland_S",    "west": 14.0, "south": 62.0, "east": 19.0, "north": 64.0},
    ]

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Collect all LPIS parcel centroids with crop classes
    all_points = defaultdict(list)

    for region in REGIONS:
        bbox_str = (f"{region['west']},{region['south']},"
                    f"{region['east']},{region['north']},EPSG:4326")

        # Cache key
        cache_key = hashlib.md5(f"{bbox_str}|skifte".encode()).hexdigest()[:12]
        cache_path = Path(cache_dir) / f"skifte_{cache_key}.json"

        print(f"\n  {region['name']}...", end=" ")

        if cache_path.exists():
            print("(cached)")
            with open(cache_path) as f:
                data = _json.load(f)
        else:
            params = {
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetFeature",
                "typeName": WFS_LAYER,
                "outputFormat": "application/json",
                "srsName": "EPSG:3006",
                "bbox": bbox_str,
                "count": "10000",
            }
            url = f"{WFS_URL}?{urllib.parse.urlencode(params)}"
            try:
                resp = urllib.request.urlopen(url, timeout=90)
                data = _json.loads(resp.read())
                with open(cache_path, "w") as f:
                    _json.dump(data, f)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

        features = data.get("features", [])
        print(f"{len(features)} skiften")

        for feat in features:
            props = feat.get("properties", {})
            grodkod = props.get("grdkod_mar")
            if grodkod is None:
                continue
            try:
                grodkod = int(grodkod)
            except (ValueError, TypeError):
                continue

            crop_class = sjv_grodkod_to_class(grodkod)
            if crop_class == 0:
                continue
            if target_classes and crop_class not in target_classes:
                continue

            # Compute centroid from geometry
            geom = feat.get("geometry", {})
            coords_list = geom.get("coordinates", [])
            if not coords_list:
                continue

            try:
                # Polygon → first ring → mean of coords for centroid
                ring = coords_list[0] if geom["type"] == "Polygon" else coords_list[0][0]
                xs = [c[0] for c in ring]
                ys = [c[1] for c in ring]
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)

                # EPSG:3006 → WGS84
                import pyproj
                transformer = pyproj.Transformer.from_crs(
                    "EPSG:3006", "EPSG:4326", always_xy=True,
                )
                lon, lat = transformer.transform(cx, cy)
            except Exception:
                continue

            blockid = props.get("blockid", "")
            skifte = props.get("skiftesbeteckning", "")
            areal = props.get("faststalld_areal_ha") or props.get("ansokt_areal_ha", 0)

            all_points[crop_class].append({
                "point_id": f"LPIS_{blockid}_{skifte}_{grodkod}",
                "lat": lat,
                "lon": lon,
                "lc_code": f"SJV_{grodkod}",
                "crop_class": crop_class,
                "crop_name": CLASS_NAMES[crop_class],
                "year": int(props.get("arslager", 2025)),
                "nuts0": "SE",
                "source": "lpis",
                "sjv_grodkod": grodkod,
                "region": region["name"],
                "areal_ha": float(areal) if areal else 0,
            })

    # Sample n_per_class from each class
    sampled = []
    for cls_idx in range(NUM_CLASSES):
        pts = all_points.get(cls_idx, [])
        if not pts:
            continue
        n = min(n_per_class, len(pts))
        sampled.extend(rng.sample(pts, n))
        print(f"  {CLASS_NAMES[cls_idx]:20s}: {len(pts):5d} LPIS blocks → sampled {n}")

    return sampled


# ── Cross-validation ─────────────────────────────────────────────────────

def cross_validate_lpis_lucas(
    lucas_points: list[dict],
    lpis_points: list[dict],
    tolerance_m: float = 500,
) -> dict:
    """Cross-validate LPIS crop classes against LUCAS ground truth.

    Matches LUCAS points to nearest LPIS point within tolerance.

    Returns:
        Dict with confusion matrix, accuracy, per-class metrics.
    """
    from scipy.spatial import cKDTree

    # Convert to arrays for spatial matching
    lucas_coords = np.array([[p["lat"], p["lon"]] for p in lucas_points])
    lpis_coords = np.array([[p["lat"], p["lon"]] for p in lpis_points])

    if len(lpis_coords) == 0:
        return {"error": "No LPIS points to validate against"}

    # Build KD-tree on LPIS
    tree = cKDTree(lpis_coords)

    # Match each LUCAS point to nearest LPIS
    # ~111km per degree lat, ~60km per degree lon at 60°N
    tol_deg = tolerance_m / 111000.0

    matches = 0
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))

    for i, lp in enumerate(lucas_points):
        dist, idx = tree.query(lucas_coords[i])
        if dist > tol_deg:
            continue

        matches += 1
        lucas_cls = lp["crop_class"]
        lpis_cls = lpis_points[idx]["crop_class"]
        confusion[lucas_cls][lpis_cls] += 1
        if lucas_cls == lpis_cls:
            correct += 1

    accuracy = correct / max(matches, 1)

    return {
        "matches": matches,
        "correct": correct,
        "accuracy": accuracy,
        "confusion": dict(confusion),
        "total_lucas": len(lucas_points),
        "total_lpis": len(lpis_points),
    }


# ── Dataset building ─────────────────────────────────────────────────────

def build_balanced_dataset(
    lucas_points: list[dict],
    lpis_points: list[dict],
    cap_multiplier: float = 3.0,
    seed: int = 42,
) -> list[dict]:
    """Build balanced dataset: LUCAS backbone + LPIS gap-fill.

    Strategy:
      - LUCAS points are always included (ground truth)
      - LPIS fills gaps: classes with <min_threshold LUCAS points get LPIS supplement
      - Majority classes capped at cap_multiplier × median
      - Each point tagged with source ("lucas" / "lpis")
    """
    rng = random.Random(seed)

    # Tag LUCAS points
    for p in lucas_points:
        p["source"] = "lucas"

    # Group by class
    lucas_by_class = defaultdict(list)
    for p in lucas_points:
        lucas_by_class[p["crop_class"]].append(p)

    lpis_by_class = defaultdict(list)
    for p in lpis_points:
        lpis_by_class[p["crop_class"]].append(p)

    # Calculate cap
    lucas_sizes = [len(v) for v in lucas_by_class.values() if len(v) > 0]
    if lucas_sizes:
        median_size = statistics.median(lucas_sizes)
        cap = int(median_size * cap_multiplier)
    else:
        cap = 300

    print(f"\nBalancing: median LUCAS class = {median_size:.0f}, cap = {cap}")

    balanced = []
    for cls_idx in range(NUM_CLASSES):
        if cls_idx == 0:
            continue

        lucas_pts = lucas_by_class.get(cls_idx, [])
        lpis_pts = lpis_by_class.get(cls_idx, [])

        # Start with all LUCAS points
        cls_points = list(lucas_pts)
        n_lucas = len(cls_points)

        # Fill with LPIS if LUCAS is insufficient
        if n_lucas < 30 and lpis_pts:
            # Need LPIS supplement
            need = min(cap, max(30, n_lucas * 3)) - n_lucas
            supplement = rng.sample(lpis_pts, min(need, len(lpis_pts)))
            cls_points.extend(supplement)
            n_lpis = len(supplement)
        else:
            n_lpis = 0

        # Cap majority classes
        if len(cls_points) > cap:
            # Prefer LUCAS over LPIS when capping
            lucas_in = [p for p in cls_points if p.get("source") == "lucas"]
            lpis_in = [p for p in cls_points if p.get("source") == "lpis"]
            if len(lucas_in) > cap:
                cls_points = rng.sample(lucas_in, cap)
            else:
                remaining = cap - len(lucas_in)
                cls_points = lucas_in + rng.sample(lpis_in, min(remaining, len(lpis_in)))

        balanced.extend(cls_points)
        print(f"  {CLASS_NAMES[cls_idx]:20s}: {n_lucas:4d} LUCAS + {n_lpis:4d} LPIS = {len(cls_points):4d}")

    return balanced


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build balanced crop training dataset from LPIS + LUCAS"
    )
    parser.add_argument(
        "--lucas-csv", nargs="+", required=True,
        help="LUCAS CSV files (2018 and/or 2022)",
    )
    parser.add_argument(
        "--output", default="data/lucas/balanced_points_v2.json",
    )
    parser.add_argument(
        "--lpis-cache", default="data/lpis_cache",
    )
    parser.add_argument(
        "--lpis-per-class", type=int, default=200,
        help="Max LPIS points per class (default: 200)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate LPIS vs LUCAS, don't build dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Load LUCAS
    csv_paths = args.lucas_csv if len(args.lucas_csv) > 1 else args.lucas_csv[0]
    print("Loading LUCAS-SE points...")
    lucas_points = load_lucas_sweden(csv_paths, crop_only=True)
    lucas_summary = summarize_lucas_sweden(lucas_points)
    print(f"  LUCAS: {lucas_summary['total']} points")
    for cls, count in sorted(lucas_summary["per_class"].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {cls:20s}: {count}")

    # 2. Fetch LPIS — focus on gap classes
    gap_classes = []
    for cls_idx in range(1, NUM_CLASSES):
        count = lucas_summary["per_class"].get(CLASS_NAMES[cls_idx], 0)
        if count < 50:
            gap_classes.append(cls_idx)

    print(f"\nGap classes (< 50 LUCAS points): "
          f"{[CLASS_NAMES[c] for c in gap_classes]}")

    print("\nFetching LPIS from Jordbruksverket WFS...")
    lpis_points = fetch_lpis_sample(
        n_per_class=args.lpis_per_class,
        target_classes=gap_classes if gap_classes else None,
        cache_dir=args.lpis_cache,
        seed=args.seed,
    )
    print(f"\nTotal LPIS sample: {len(lpis_points)}")

    # 3. Cross-validate
    if lpis_points:
        print("\nCross-validating LPIS vs LUCAS...")
        validation = cross_validate_lpis_lucas(lucas_points, lpis_points)
        print(f"  Matched: {validation['matches']} / {validation['total_lucas']} LUCAS points")
        print(f"  Accuracy: {validation['accuracy']:.1%}")
        if validation.get("confusion"):
            print("  Confusion (LUCAS→LPIS):")
            for lucas_cls, lpis_map in sorted(validation["confusion"].items()):
                lpis_str = ", ".join(
                    f"{CLASS_NAMES.get(k, '?')}:{v}" for k, v in sorted(lpis_map.items())
                )
                print(f"    {CLASS_NAMES.get(lucas_cls, '?'):20s} → {lpis_str}")

    if args.validate_only:
        print("\nValidation only — done.")
        return

    # 4. Build balanced dataset
    print("\nBuilding balanced dataset...")
    balanced = build_balanced_dataset(lucas_points, lpis_points, seed=args.seed)

    # Summary
    balanced_summary = summarize_lucas_sweden(balanced)
    print(f"\nFinal dataset: {balanced_summary['total']} points")
    sources = Counter(p.get("source", "lucas") for p in balanced)
    print(f"  Sources: {dict(sources)}")
    for cls, count in sorted(balanced_summary["per_class"].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {cls:20s}: {count}")

    # 5. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "metadata": {
                "total": len(balanced),
                "sources": dict(sources),
                "lucas_total": len(lucas_points),
                "lpis_total": len(lpis_points),
                "class_names": CLASS_NAMES,
                "class_names_sv": CLASS_NAMES_SV,
                "num_classes": NUM_CLASSES,
                "sjv_grodkod_mapping": {str(k): v for k, v in SJV_TO_CROP.items()},
                "validation_accuracy": validation.get("accuracy") if lpis_points else None,
            },
            "points": balanced,
        }, f, indent=2, default=str)

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
