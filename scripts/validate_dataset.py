"""
scripts/validate_dataset.py — Dataset quality validation for unified_v2

Verifies 8,135 tiles for consistent, correct labels. Not a filter — a diagnostic.

Checks:
  1. Label range        — values >= NUM_UNIFIED_CLASSES
  2. SJV/LPIS consistency (crop tiles)
  3. NMD plausibility   (lulc tiles)
  4. Cross-tile statistics — per-class tile/pixel counts, rare-class flags
  5. Missing label key

Output:
  - Summary to stdout
  - validation_report.json in data_dir for any tile with failures
  - Exit 0 if no critical failures (label range violations), 1 if any critical failures
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Import unified schema — try installed package first, then fallback to repo
# ---------------------------------------------------------------------------
try:
    from imint.training.unified_schema import (
        NUM_UNIFIED_CLASSES,
        SJV_TO_UNIFIED,
        UNIFIED_CLASSES,
    )
except ModuleNotFoundError:
    # Running from repo root without install
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from imint.training.unified_schema import (
        NUM_UNIFIED_CLASSES,
        SJV_TO_UNIFIED,
        UNIFIED_CLASSES,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tile-name prefix → expected unified class (dominant LPIS class)
_CROP_PREFIX_TO_CLASS: dict[str, int] = {
    "vete":         11,
    "korn":         12,
    "havre":        13,
    "oljevaxter":   14,
    "slattervall":  15,
    "bete":         16,
    "potatis":      17,
    "sockerbetor":  18,
    "trindsad":     19,
    "rag":          20,
    "majs":         21,
    "hygge":        22,
}

_WATER_CLASS = 10
_WATER_DOMINANT_THRESH = 0.90
_UNIFORMITY_THRESH = 0.95  # fraction of pixels dominated by a single class
_RARE_CLASS_MIN_TILES = 10

_SJV_KNOWN_CODES: frozenset[int] = frozenset(SJV_TO_UNIFIED.keys())
_SJV_DEFAULT_CLASS = 0   # background — destination for unmapped SJV codes


# ---------------------------------------------------------------------------
# Per-tile validation
# ---------------------------------------------------------------------------

def _crop_expected_class(tile_name: str) -> int | None:
    """Return the expected dominant unified class from a crop tile name, or None."""
    # e.g. "crop_oljevaxter_2022_12345.npz" → "oljevaxter"
    m = re.match(r"crop_([^_]+)", tile_name)
    if not m:
        return None
    key = m.group(1).lower()
    return _CROP_PREFIX_TO_CLASS.get(key)


def _year_from_dates(dates: Any) -> int | None:
    """Extract year (int) from dates array/list. Returns None on failure."""
    try:
        first = str(dates[0])
        return int(first[:4])
    except (IndexError, ValueError, TypeError):
        return None


def validate_tile(
    tile_name: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Validate a single tile. Returns a result dict with issue lists."""
    issues: list[str] = []
    critical: list[str] = []  # label-range violations → trigger exit 1

    # ── Check 5: missing label key ──────────────────────────────────────────
    if "label" not in data:
        issues.append("missing_label_key")
        return {"tile": tile_name, "critical": [], "issues": issues}

    label: np.ndarray = data["label"]
    if label.dtype != np.uint8:
        label = label.astype(np.uint8)

    total_pixels: int = label.size
    source: str = str(data.get("source", "unknown")) if "source" in data else "unknown"

    # ── Check 1: label range ────────────────────────────────────────────────
    max_val = int(label.max())
    if max_val >= NUM_UNIFIED_CLASSES:
        bad_vals = sorted(int(v) for v in np.unique(label) if v >= NUM_UNIFIED_CLASSES)
        bad_px = int(np.sum(label >= NUM_UNIFIED_CLASSES))
        critical.append(
            f"label_out_of_range: max={max_val}, "
            f"bad_values={bad_vals}, bad_pixels={bad_px}/{total_pixels}"
        )

    # ── Check 2: SJV/LPIS consistency (crop tiles only) ────────────────────
    is_crop = tile_name.startswith("crop_")
    if is_crop or source == "crop":
        # 2a: label_mask SJV codes
        if "label_mask" in data:
            mask: np.ndarray = data["label_mask"]
            active_codes = np.unique(mask[mask > 0]).tolist()
            unknown_codes = [
                int(c) for c in active_codes
                if int(c) not in _SJV_KNOWN_CODES
            ]
            if unknown_codes:
                issues.append(
                    f"sjv_unknown_codes: {unknown_codes[:20]}"
                    + ("..." if len(unknown_codes) > 20 else "")
                )
        else:
            issues.append("crop_tile_missing_label_mask")

        # 2b: expected dominant class present
        expected_cls = _crop_expected_class(tile_name)
        if expected_cls is not None:
            counts = np.bincount(label.ravel(), minlength=NUM_UNIFIED_CLASSES)
            # exclude background (0) when looking for dominant
            counts_no_bg = counts.copy()
            counts_no_bg[0] = 0
            dominant_cls = int(counts_no_bg.argmax())
            if counts_no_bg[expected_cls] == 0:
                issues.append(
                    f"expected_class_absent: expected={expected_cls}"
                    f"({UNIFIED_CLASSES[expected_cls]}), dominant={dominant_cls}"
                    f"({UNIFIED_CLASSES.get(dominant_cls, '?')})"
                )
            elif dominant_cls != expected_cls:
                # warn but not an error — mixed tiles are normal
                frac = counts_no_bg[expected_cls] / max(counts_no_bg.sum(), 1)
                if frac < 0.05:
                    issues.append(
                        f"expected_class_rare: expected={expected_cls}"
                        f"({UNIFIED_CLASSES[expected_cls]}) covers only "
                        f"{frac:.1%}, dominant={dominant_cls}"
                        f"({UNIFIED_CLASSES.get(dominant_cls, '?')})"
                    )

        # 2c: year consistency
        if "lpis_year" in data and "dates" in data:
            lpis_year = int(data["lpis_year"])
            dates_year = _year_from_dates(data["dates"])
            if dates_year is not None and lpis_year != dates_year:
                issues.append(
                    f"year_mismatch: lpis_year={lpis_year}, dates_year={dates_year}"
                )

    # ── Check 3: NMD plausibility (lulc tiles) ─────────────────────────────
    is_lulc = tile_name.startswith("urban_") or source == "lulc"
    if is_lulc:
        counts = np.bincount(label.ravel(), minlength=NUM_UNIFIED_CLASSES)

        # All background → labeling failure
        nonbg_pixels = total_pixels - int(counts[0])
        if nonbg_pixels == 0:
            issues.append("all_background: entire tile is class 0")
        else:
            # Single-class dominance (excluding water — expected)
            for cls_idx in range(NUM_UNIFIED_CLASSES):
                frac = counts[cls_idx] / total_pixels
                if frac >= _UNIFORMITY_THRESH:
                    if cls_idx == _WATER_CLASS:
                        pass  # water tiles >90% are expected — only flag at 100%
                    else:
                        issues.append(
                            f"suspicious_uniformity: class={cls_idx}"
                            f"({UNIFIED_CLASSES.get(cls_idx, '?')}) covers {frac:.1%}"
                        )
                    break  # one dominant class per tile — no need to continue

    return {"tile": tile_name, "critical": critical, "issues": issues}


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def run_validation(data_dir: str) -> int:
    """Run full validation. Returns exit code (0=ok, 1=critical failures)."""

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
    total = len(files)
    print(f"Dataset dir:  {data_dir}")
    print(f"Tiles found:  {total:,}")
    print()

    # Accumulators
    failing_tiles: list[dict[str, Any]] = []
    load_errors: list[str] = []
    no_label: list[str] = []

    # Cross-tile statistics
    per_class_tile_count: np.ndarray = np.zeros(NUM_UNIFIED_CLASSES, dtype=np.int64)
    per_class_pixel_count: np.ndarray = np.zeros(NUM_UNIFIED_CLASSES, dtype=np.int64)

    # Detailed counters
    n_critical = 0
    n_issues = 0
    n_crop = 0
    n_lulc = 0

    for idx, fname in enumerate(files):
        if idx % 500 == 0 and idx > 0:
            print(f"  ... {idx:,}/{total:,} tiles scanned", flush=True)

        fpath = os.path.join(data_dir, fname)
        try:
            npz = np.load(fpath, allow_pickle=False)
            data: dict[str, Any] = dict(npz)
        except Exception as exc:
            load_errors.append(f"{fname}: {exc}")
            continue

        # Source accounting
        source = str(data.get("source", "")) if "source" in data else ""
        if source == "crop" or fname.startswith("crop_"):
            n_crop += 1
        elif source == "lulc" or fname.startswith("urban_"):
            n_lulc += 1

        # Per-tile validation
        result = validate_tile(fname, data)

        # Accumulate class stats (only when label present)
        if "label" in data:
            lbl: np.ndarray = data["label"]
            counts = np.bincount(lbl.ravel().astype(np.int64), minlength=NUM_UNIFIED_CLASSES)
            # Clip to schema size — out-of-range bins are ignored for stats
            counts = counts[:NUM_UNIFIED_CLASSES]
            per_class_pixel_count += counts
            for cls_idx in range(NUM_UNIFIED_CLASSES):
                if counts[cls_idx] > 0:
                    per_class_tile_count[cls_idx] += 1
        else:
            no_label.append(fname)

        if result["critical"] or result["issues"]:
            if result["critical"]:
                n_critical += 1
            if result["issues"]:
                n_issues += 1
            failing_tiles.append(result)

    # ── Check 4: Cross-tile rare class detection ────────────────────────────
    rare_classes: list[dict[str, Any]] = []
    for cls_idx in range(1, NUM_UNIFIED_CLASSES):  # skip background
        tc = int(per_class_tile_count[cls_idx])
        if tc < _RARE_CLASS_MIN_TILES:
            rare_classes.append({
                "class": cls_idx,
                "name": UNIFIED_CLASSES.get(cls_idx, "?"),
                "tile_count": tc,
                "pixel_count": int(per_class_pixel_count[cls_idx]),
            })

    # ── Print summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total tiles:            {total:>8,}")
    print(f"  Crop tiles:           {n_crop:>8,}")
    print(f"  LULC/urban tiles:     {n_lulc:>8,}")
    print(f"Load errors:            {len(load_errors):>8,}")
    print(f"Missing label key:      {len(no_label):>8,}")
    print(f"Tiles with issues:      {n_issues:>8,}")
    print(f"Tiles with CRITICAL:    {n_critical:>8,}  (label range violations)")
    print()

    if load_errors:
        print(f"Load errors (first 10):")
        for e in load_errors[:10]:
            print(f"  {e}")
        print()

    if no_label:
        print(f"Missing label key (first 10):")
        for f in no_label[:10]:
            print(f"  {f}")
        print()

    # Per-class statistics
    print("Per-class statistics (class | name | tiles | pixels):")
    print(f"  {'cls':>3}  {'name':<22}  {'tiles':>7}  {'pixels':>14}")
    print(f"  {'-'*3}  {'-'*22}  {'-'*7}  {'-'*14}")
    for cls_idx in range(NUM_UNIFIED_CLASSES):
        name = UNIFIED_CLASSES.get(cls_idx, "?")
        tc = int(per_class_tile_count[cls_idx])
        pc = int(per_class_pixel_count[cls_idx])
        flag = " *** RARE" if cls_idx > 0 and tc < _RARE_CLASS_MIN_TILES else ""
        print(f"  {cls_idx:>3}  {name:<22}  {tc:>7,}  {pc:>14,}{flag}")
    print()

    if rare_classes:
        print(f"RARE CLASSES (<{_RARE_CLASS_MIN_TILES} tiles):")
        for rc in rare_classes:
            print(f"  class {rc['class']:>2} {rc['name']}: {rc['tile_count']} tiles, {rc['pixel_count']:,} pixels")
        print()

    if n_critical > 0:
        print(f"CRITICAL FAILURES: {n_critical} tile(s) have label values >= {NUM_UNIFIED_CLASSES}")
        print("First 20 critical tiles:")
        shown = 0
        for t in failing_tiles:
            if t["critical"] and shown < 20:
                print(f"  {t['tile']}")
                for c in t["critical"]:
                    print(f"    CRITICAL: {c}")
                shown += 1
        print()

    # ── Write report JSON ───────────────────────────────────────────────────
    report = {
        "summary": {
            "total_tiles": total,
            "crop_tiles": n_crop,
            "lulc_tiles": n_lulc,
            "load_errors": len(load_errors),
            "no_label_tiles": len(no_label),
            "tiles_with_issues": n_issues,
            "tiles_with_critical": n_critical,
            "num_unified_classes": NUM_UNIFIED_CLASSES,
            "rare_class_threshold": _RARE_CLASS_MIN_TILES,
        },
        "per_class_stats": {
            cls_idx: {
                "name": UNIFIED_CLASSES.get(cls_idx, "?"),
                "tile_count": int(per_class_tile_count[cls_idx]),
                "pixel_count": int(per_class_pixel_count[cls_idx]),
                "rare": bool(cls_idx > 0 and per_class_tile_count[cls_idx] < _RARE_CLASS_MIN_TILES),
            }
            for cls_idx in range(NUM_UNIFIED_CLASSES)
        },
        "rare_classes": rare_classes,
        "load_errors": load_errors,
        "no_label_tiles": no_label,
        "failing_tiles": failing_tiles,
    }

    report_path = os.path.join(data_dir, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"Report written: {report_path}")

    exit_code = 1 if n_critical > 0 else 0
    status = "CRITICAL FAILURES FOUND" if exit_code else "OK — no critical failures"
    print(f"Exit status:    {exit_code} ({status})")
    return exit_code


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/unified_v2"
    sys.exit(run_validation(data_dir))
