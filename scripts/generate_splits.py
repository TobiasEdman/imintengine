#!/usr/bin/env python3
"""scripts/generate_splits.py — Zone-stratified train/val/test split.

Assigns every tile to one of the 8 Swedish agricultural growing zones
(odlingszoner 1–8, Jordbruksverket) based on its SWEREF99TM northing
coordinate, then draws a stratified val and test set that covers ALL
zones — guaranteeing the model is evaluated on every climate/vegetation
regime from nemoral Skåne (zon 1) to alpine Lappland (zon 8).

Zone northing boundaries (approximate SWEREF99TM, meters):
  Zon 1   N < 6 200 000   Skåne coast / southernmost Sweden
  Zon 2   N < 6 400 000   Skåne interior, Blekinge, Öland south
  Zon 3   N < 6 650 000   Götaland, Kalmar, Gotland
  Zon 4   N < 6 850 000   Svealand, northern Götaland
  Zon 5   N < 7 100 000   Gästrikland, Hälsingland, northern Dalarna
  Zon 6   N < 7 300 000   Jämtland, Medelpad, southern Ångermanland
  Zon 7   N < 7 600 000   Northern Norrland, southern Norrbotten
  Zon 8   N ≥ 7 600 000   Lappland (most arctic/alpine areas)

Split strategy within each zone
  ─ Sort tiles by (northing, easting) — deterministic
  ─ Take a spatially contiguous strip from the middle of each zone for
    val (avoids both extremes which may share class distributions with
    adjacent zones)
  ─ Take a small block from the southern end of each zone for test
  ─ Everything else → train
  ─ A minimum of MIN_VAL_PER_ZONE tiles is guaranteed in val even for
    small/sparse zones (draws from train if needed)

Tile name conventions handled
  tile_E_N.npz            → parse E, N from filename
  crop_*_E_N.npz          → parse E, N from last two numeric tokens
  urban_*_E_N.npz         → same
  LPIS_*/lucas_* etc.     → read easting/northing from .npz fields;
                            fall back to bbox_3006 centre; fall back
                            to train assignment (safe default)

Usage::

    # Local (reads from local data dir)
    python scripts/generate_splits.py --data-dir /data/unified_v2

    # With custom fractions
    python scripts/generate_splits.py --data-dir /data/unified_v2 \\
        --val-frac 0.15 --test-frac 0.05 --min-val-per-zone 15

    # Dry-run (print zone table, do not write files)
    python scripts/generate_splits.py --data-dir /data/unified_v2 --dry-run

Output files (written to --data-dir):
    train.txt   one tile stem per line (no extension)
    val.txt
    test.txt
    splits_summary.json   zone breakdown statistics
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ── Zone boundaries (SWEREF99TM northing, metres) ─────────────────────────

ZONE_BOUNDS: list[tuple[str, int]] = [
    # (name, max_northing_exclusive)  — zone contains tiles with N < bound
    ("zon1", 6_200_000),
    ("zon2", 6_400_000),
    ("zon3", 6_650_000),
    ("zon4", 6_850_000),
    ("zon5", 7_100_000),
    ("zon6", 7_300_000),
    ("zon7", 7_600_000),
    ("zon8", 10_000_000),  # everything north of zon7
]

ZONE_NAMES = [z[0] for z in ZONE_BOUNDS]

# Fraction of tiles per zone used for val (middle of northing range)
# and test (southern edge of each zone).
DEFAULT_VAL_FRAC  = 0.15
DEFAULT_TEST_FRAC = 0.05
DEFAULT_MIN_VAL   = 10   # minimum val tiles per zone regardless of fraction

# ── Coordinate extraction ──────────────────────────────────────────────────

# Patterns for filename-parseable coordinates
_RE_TILE   = re.compile(r"tile_(\d+)_(\d+)\.npz$")
_RE_COORDS = re.compile(r".*_(\d{5,7})_(\d{7,8})\.npz$")  # last two numeric tokens


class TileCoord(NamedTuple):
    stem: str
    easting:  float
    northing: float


def _coords_from_name(name: str) -> tuple[float, float] | None:
    """Try to parse (easting, northing) from tile filename."""
    m = _RE_TILE.search(name)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = _RE_COORDS.search(name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def _coords_from_npz(path: Path) -> tuple[float, float] | None:
    """Read (easting, northing) from .npz file fields."""
    try:
        data = np.load(path, allow_pickle=False)
        if "easting" in data and "northing" in data:
            return float(data["easting"]), float(data["northing"])
        if "bbox_3006" in data:
            bbox = data["bbox_3006"]  # [west, south, east, north]
            cx = (float(bbox[0]) + float(bbox[2])) / 2
            cy = (float(bbox[1]) + float(bbox[3])) / 2
            return cx, cy
    except Exception:
        pass
    return None


# ── Zone assignment ────────────────────────────────────────────────────────

def assign_zone(northing: float) -> str:
    for name, bound in ZONE_BOUNDS:
        if northing < bound:
            return name
    return ZONE_NAMES[-1]


# ── Split logic ────────────────────────────────────────────────────────────

def make_splits(
    tiles: list[TileCoord],
    val_frac: float,
    test_frac: float,
    min_val_per_zone: int,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Return (train_stems, val_stems, test_stems) with guaranteed zone coverage.

    Within each zone tiles are sorted by (northing, easting) and then:
      - Test  = southern end of zone (lowest northings)
      - Val   = middle strip of zone (ensures separation from test)
      - Train = remainder
    """
    # Group by zone
    by_zone: dict[str, list[TileCoord]] = {z: [] for z in ZONE_NAMES}
    unzoned: list[TileCoord] = []

    for tc in tiles:
        if tc.northing is None or tc.northing == 0:
            unzoned.append(tc)
        else:
            by_zone[assign_zone(tc.northing)].append(tc)

    train_stems: list[str] = []
    val_stems:   list[str] = []
    test_stems:  list[str] = []

    rng = np.random.default_rng(seed)

    for zone, zone_tiles in by_zone.items():
        if not zone_tiles:
            continue

        # Sort by northing then easting for deterministic spatial ordering
        zone_tiles_sorted = sorted(zone_tiles, key=lambda t: (t.northing, t.easting))
        n = len(zone_tiles_sorted)

        n_test = max(1, int(n * test_frac)) if n >= 5 else 0
        n_val  = max(min_val_per_zone, int(n * val_frac))
        n_val  = min(n_val, n - n_test - 1)  # leave at least 1 for train
        n_val  = max(0, n_val)

        # Test = southernmost tiles (lowest northing)
        test_batch  = zone_tiles_sorted[:n_test]
        remaining   = zone_tiles_sorted[n_test:]

        # Val = middle of remaining range (take from the centre)
        mid = len(remaining) // 2
        half_v = n_val // 2
        val_start = max(0, mid - half_v)
        val_end   = val_start + n_val
        if val_end > len(remaining):
            val_end   = len(remaining)
            val_start = max(0, val_end - n_val)

        val_batch   = remaining[val_start:val_end]
        train_batch = remaining[:val_start] + remaining[val_end:]

        train_stems.extend(t.stem for t in train_batch)
        val_stems.extend(  t.stem for t in val_batch)
        test_stems.extend( t.stem for t in test_batch)

    # Tiles without parseable coordinates → train (safe default)
    train_stems.extend(t.stem for t in unzoned)

    return train_stems, val_stems, test_stems


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Zone-stratified train/val/test split for unified_v2")
    p.add_argument("--data-dir", required=True, help="Directory containing .npz tiles")
    p.add_argument("--val-frac",  type=float, default=DEFAULT_VAL_FRAC,
                   help=f"Target val fraction per zone (default: {DEFAULT_VAL_FRAC})")
    p.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC,
                   help=f"Target test fraction per zone (default: {DEFAULT_TEST_FRAC})")
    p.add_argument("--min-val-per-zone", type=int, default=DEFAULT_MIN_VAL,
                   help=f"Minimum val tiles per zone (default: {DEFAULT_MIN_VAL})")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Print zone table and exit without writing files")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: {data_dir} not found", file=sys.stderr)
        sys.exit(1)

    # ── Discover tiles ──────────────────────────────────────────────────
    all_paths = sorted(
        p for p in data_dir.glob("*.npz")
        if ".tmp" not in p.name
    )
    print(f"\n  Found {len(all_paths):,} tiles in {data_dir}")

    # ── Extract coordinates ─────────────────────────────────────────────
    print("  Extracting coordinates …")
    tiles: list[TileCoord] = []
    n_from_file = n_from_npz = n_fallback = 0

    for path in all_paths:
        stem = path.stem
        coords = _coords_from_name(path.name)
        if coords is not None:
            n_from_file += 1
        else:
            coords = _coords_from_npz(path)
            if coords is not None:
                n_from_npz += 1
            else:
                n_fallback += 1
                coords = (0.0, 0.0)  # will land in unzoned → train

        e, n_coord = coords
        tiles.append(TileCoord(stem=stem, easting=e, northing=n_coord))

    print(f"  Coords: {n_from_file:,} from filename, "
          f"{n_from_npz:,} from .npz, "
          f"{n_fallback:,} fallback→train")

    # ── Zone distribution ───────────────────────────────────────────────
    by_zone: dict[str, int] = {z: 0 for z in ZONE_NAMES}
    by_zone["unzoned"] = 0
    for tc in tiles:
        if tc.northing == 0.0 and tc.easting == 0.0:
            by_zone["unzoned"] += 1
        else:
            by_zone[assign_zone(tc.northing)] += 1

    print(f"\n  {'Zone':<10} {'N tiles':>8}  {'Approx latitude':}")
    zone_lat = {
        "zon1": "< 55.5°N  (nemoral — Skåne coast)",
        "zon2": "55.5–57.7°N (Skåne interior, Blekinge)",
        "zon3": "57.7–60.0°N (Götaland, Gotland)",
        "zon4": "60.0–61.8°N (Svealand)",
        "zon5": "61.8–63.7°N (Gästrikland, Hälsingland)",
        "zon6": "63.7–65.5°N (Jämtland, Medelpad)",
        "zon7": "65.5–68.2°N (Norrland, S Norrbotten)",
        "zon8": "> 68.2°N   (Lappland, alpine)",
        "unzoned": "no coords  → train",
    }
    for z in list(ZONE_NAMES) + ["unzoned"]:
        n = by_zone[z]
        flag = " ⚠ EMPTY" if n == 0 and z != "unzoned" else ""
        print(f"  {z:<10} {n:>8}   {zone_lat.get(z,'')}{flag}")

    if args.dry_run:
        print("\n  (dry-run — no files written)")
        return

    # ── Generate splits ─────────────────────────────────────────────────
    train_stems, val_stems, test_stems = make_splits(
        tiles,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        min_val_per_zone=args.min_val_per_zone,
        seed=args.seed,
    )

    # ── Zone coverage report ────────────────────────────────────────────
    val_set  = set(val_stems)
    test_set = set(test_stems)
    stem_to_tc = {tc.stem: tc for tc in tiles}

    print(f"\n  Split summary (val_frac={args.val_frac}, "
          f"test_frac={args.test_frac}, "
          f"min_val_per_zone={args.min_val_per_zone}):")
    print(f"  {'Zone':<10}  {'train':>6}  {'val':>6}  {'test':>6}")

    zone_stats: dict[str, dict] = {}
    for zone in ZONE_NAMES:
        zone_tile_stems = {
            tc.stem for tc in tiles
            if tc.northing > 0 and assign_zone(tc.northing) == zone
        }
        n_tr = len(zone_tile_stems - val_set - test_set)
        n_vl = len(zone_tile_stems & val_set)
        n_te = len(zone_tile_stems & test_set)
        flag = " ⚠ NO VAL" if n_vl == 0 else ""
        print(f"  {zone:<10}  {n_tr:>6}  {n_vl:>6}  {n_te:>6}{flag}")
        zone_stats[zone] = {"train": n_tr, "val": n_vl, "test": n_te}

    print(f"  {'TOTAL':<10}  {len(train_stems):>6}  {len(val_stems):>6}  {len(test_stems):>6}")
    print(f"  Unzoned → train: {n_fallback}")

    # ── Write files ─────────────────────────────────────────────────────
    for fname, stems in [("train.txt", train_stems),
                         ("val.txt",   val_stems),
                         ("test.txt",  test_stems)]:
        out = data_dir / fname
        out.write_text("\n".join(sorted(stems)) + "\n")
        print(f"  ✓ {out}  ({len(stems):,} tiles)")

    summary = {
        "total": len(tiles),
        "train": len(train_stems),
        "val":   len(val_stems),
        "test":  len(test_stems),
        "val_frac":  args.val_frac,
        "test_frac": args.test_frac,
        "min_val_per_zone": args.min_val_per_zone,
        "zones": zone_stats,
        "coord_source": {
            "from_filename": n_from_file,
            "from_npz":      n_from_npz,
            "fallback_train": n_fallback,
        },
    }
    out_json = data_dir / "splits_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  ✓ {out_json}")

    print("\n  Done. Pass --split-dir to train_pixel.py to use these splits.")


if __name__ == "__main__":
    main()
