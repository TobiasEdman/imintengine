#!/usr/bin/env python3
"""Audit tile .npz files for partial/corrupt enrichment writes.

During the disk-quota crunch, ``np.savez_compressed`` may have written
truncated files while the ``has_*`` sentinel was already set to 1.
This audit:

    1. Walks every .npz in the data dir.
    2. For each enrichment type (rededge, tessera, b08, s1, label),
       when the sentinel says the data exists, verifies the shape and
       that the array isn't mostly zeros (indicating failed write).
    3. By default, prints a report. With ``--fix`` it removes the bad
       keys + sentinel so ``--skip-existing`` runs redo them.

Usage:
    # Report only
    python scripts/audit_enrichment_integrity.py --data-dir /data/unified_v2_512

    # Report + reset bad sentinels so enrichment re-runs them
    python scripts/audit_enrichment_integrity.py --data-dir /data/unified_v2_512 --fix
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _n_frames_from_spectral(spectral: np.ndarray, n_bands: int = 6) -> int:
    return spectral.shape[0] // n_bands


def audit_tile(path: str) -> dict:
    """Return a dict of integrity findings for one tile.

    Keys:
        name: tile basename
        issues: list of (key, reason) for bad sentinels
        clean: bool — True if no issues
    """
    name = Path(path).stem
    result = {"name": name, "issues": [], "keys": []}
    try:
        data = dict(np.load(path, allow_pickle=True))
    except Exception as e:
        result["issues"].append(("LOAD", f"cannot load: {e}"))
        return result

    spectral = data.get("spectral", data.get("image"))
    if spectral is None or not isinstance(spectral, np.ndarray) or spectral.ndim != 3:
        result["issues"].append(("spectral", "missing or malformed"))
        return result

    h, w = spectral.shape[1], spectral.shape[2]
    n_frames = _n_frames_from_spectral(spectral)

    # --- rededge (has_rededge=1 → expect (3*T, H, W) with non-zero content) ---
    if int(data.get("has_rededge", 0)) == 1:
        arr = data.get("rededge")
        if arr is None:
            result["issues"].append(("rededge", "has_rededge=1 but key missing"))
        elif arr.shape != (3 * n_frames, h, w):
            result["issues"].append(
                ("rededge", f"shape {arr.shape} != ({3*n_frames}, {h}, {w})")
            )
        elif float(np.abs(arr).mean()) < 1e-6:
            result["issues"].append(("rededge", "all-zero content"))
        else:
            result["keys"].append("rededge")

    # --- b08 (has_b08=1 → (T, H, W) non-zero) ---
    if int(data.get("has_b08", 0)) == 1:
        arr = data.get("b08")
        if arr is None:
            result["issues"].append(("b08", "has_b08=1 but key missing"))
        elif arr.shape != (n_frames, h, w):
            result["issues"].append(
                ("b08", f"shape {arr.shape} != ({n_frames}, {h}, {w})")
            )
        elif float(np.abs(arr).mean()) < 1e-6:
            result["issues"].append(("b08", "all-zero content"))
        else:
            result["keys"].append("b08")

    # --- s1 (has_s1=1 → (2*T, H, W) non-zero) ---
    if int(data.get("has_s1", 0)) == 1:
        arr = data.get("s1_vv_vh")
        if arr is None:
            result["issues"].append(("s1", "has_s1=1 but s1_vv_vh missing"))
        elif arr.shape != (2 * n_frames, h, w):
            result["issues"].append(
                ("s1", f"shape {arr.shape} != ({2*n_frames}, {h}, {w})")
            )
        elif float(np.abs(arr).mean()) < 1e-6:
            result["issues"].append(("s1", "all-zero content"))
        else:
            result["keys"].append("s1")

    # --- tessera (has_tessera=1 → (128, H, W) non-zero) ---
    if int(data.get("has_tessera", 0)) == 1:
        arr = data.get("tessera")
        if arr is None:
            result["issues"].append(("tessera", "has_tessera=1 but key missing"))
        elif arr.shape != (128, h, w):
            result["issues"].append(
                ("tessera", f"shape {arr.shape} != (128, {h}, {w})")
            )
        elif float(np.abs(arr).mean()) < 1e-6:
            result["issues"].append(("tessera", "all-zero content"))
        else:
            result["keys"].append("tessera")

    # --- label (present → (H, W) uint8 with values) ---
    if "label" in data:
        arr = data["label"]
        if arr.shape != (h, w):
            result["issues"].append(
                ("label", f"shape {arr.shape} != ({h}, {w})")
            )
        else:
            result["keys"].append("label")

    return result


def fix_tile(path: str, bad_keys: list[str]) -> None:
    """Remove bad enrichment keys + their sentinels so skip_existing redoes them.

    Leaves good keys intact.
    """
    sentinel_map = {
        "rededge": "has_rededge",
        "b08": "has_b08",
        "s1": "has_s1",
        "tessera": "has_tessera",
    }
    data = dict(np.load(path, allow_pickle=True))
    for key in bad_keys:
        if key == "rededge":
            data.pop("rededge", None)
            data.pop("has_rededge", None)
        elif key == "b08":
            data.pop("b08", None)
            data.pop("has_b08", None)
        elif key == "s1":
            data.pop("s1_vv_vh", None)
            data.pop("s1_temporal_mask", None)
            data.pop("s1_dates", None)
            data.pop("has_s1", None)
        elif key == "tessera":
            data.pop("tessera", None)
            data.pop("tessera_year", None)
            data.pop("has_tessera", None)
        elif key == "label":
            data.pop("label", None)
            data.pop("nmd_label_raw", None)
            data.pop("label_mask", None)
            data.pop("parcel_area_ha", None)
            data.pop("n_parcels", None)
            data.pop("harvest_mask", None)
            data.pop("n_harvest_polygons", None)
            data.pop("n_mature_polygons", None)
            data.pop("harvest_probability", None)
            data.pop("nmd_area_ha", None)
    np.savez_compressed(path, **data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--fix", action="store_true",
                   help="Remove bad keys + sentinels so enrichment redoes them")
    p.add_argument("--glob", default="*.npz",
                   help="Tile glob pattern (default *.npz)")
    p.add_argument("--max-tiles", type=int, default=None)
    args = p.parse_args()

    tiles = sorted(glob.glob(os.path.join(args.data_dir, args.glob)))
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    print(f"=== Enrichment integrity audit ===")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Tiles:    {len(tiles)}")
    print(f"  Mode:     {'FIX' if args.fix else 'report only'}")
    print()

    issue_counts: dict[str, int] = {}
    bad_tiles: list[tuple[str, list[str]]] = []  # (path, [bad_keys])
    clean = 0

    for i, path in enumerate(tiles):
        r = audit_tile(path)
        if r["issues"]:
            bad_keys = []
            for key, reason in r["issues"]:
                issue_counts[key] = issue_counts.get(key, 0) + 1
                bad_keys.append(key)
            bad_tiles.append((path, bad_keys))
        else:
            clean += 1
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(tiles)}] clean={clean} bad={len(bad_tiles)}",
                  flush=True)

    print()
    print(f"=== Results ===")
    print(f"  Clean tiles:      {clean:5}  ({100*clean/len(tiles):.1f}%)")
    print(f"  Tiles with issues:{len(bad_tiles):5}  ({100*len(bad_tiles)/len(tiles):.1f}%)")
    print()
    print(f"  Issues by key:")
    for k, n in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:10} {n:5}")
    print()

    # Print first 20 bad tiles as examples
    if bad_tiles:
        print(f"  First 20 bad tiles:")
        for path, bad_keys in bad_tiles[:20]:
            print(f"    {Path(path).stem}: {', '.join(bad_keys)}")
        print()

    if args.fix and bad_tiles:
        print(f"=== FIX: resetting {len(bad_tiles)} bad tiles ===")
        for i, (path, bad_keys) in enumerate(bad_tiles):
            try:
                fix_tile(path, bad_keys)
            except Exception as e:
                print(f"    {Path(path).stem}: FIX FAILED — {e}")
            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(bad_tiles)}] fixed", flush=True)
        print(f"Done. Enrichment jobs can now be relaunched with "
              f"--skip-existing and will redo the {len(bad_tiles)} bad tiles.")


if __name__ == "__main__":
    main()
