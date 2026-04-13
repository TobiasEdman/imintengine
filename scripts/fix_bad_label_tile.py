"""
Fix the one tile with raw SJV codes leaked into the label array.
Uses label_mask (raw SJV codes) to remap bad pixels back to unified classes.

Usage:
    python3 scripts/fix_bad_label_tile.py /data/unified_v2
"""
import sys
import os
import numpy as np

SJV_TO_UNIFIED = {
    4: 11, 5: 11, 307: 11, 316: 11,                          # vete
    1: 12, 2: 12, 12: 12, 13: 12, 315: 12,                   # korn
    3: 13, 10: 13, 15: 13,                                    # havre
    20: 14, 21: 14, 22: 14, 23: 14, 24: 14, 25: 14,
    26: 14, 27: 14, 28: 14, 38: 14, 40: 14, 41: 14, 42: 14,  # oljeväxter
    49: 15, 50: 15, 57: 15, 58: 15, 59: 15, 62: 15,
    63: 15, 302: 15, 308: 15,                                 # slåttervall
    52: 16, 53: 16, 54: 16, 55: 16, 56: 16,
    61: 16, 89: 16, 90: 16, 95: 16,                          # bete
    45: 17, 46: 17, 311: 17,                                  # potatis
    47: 18, 48: 18,                                           # sockerbetor
    30: 19, 31: 19, 32: 19, 33: 19, 34: 19,
    35: 19, 36: 19, 37: 19, 39: 19, 43: 19,                  # trindsäd
    7: 20, 8: 20, 317: 20,                                    # råg
    9: 21, 60: 21, 74: 21, 77: 21, 80: 21,
    81: 21, 85: 21, 87: 21, 88: 21,                          # övrig åker
    # Additional codes seen in this tile
    111: 14, 114: 14, 115: 14, 116: 14,
    117: 14, 118: 14, 125: 14,
}

data_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/unified_v2"
files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))

fixed = 0
deleted = 0

for fname in files:
    path = os.path.join(data_dir, fname)
    try:
        d = dict(np.load(path, allow_pickle=False))
        if "label" not in d:
            continue
        lbl = d["label"].astype(np.int32)
        if lbl.max() < 23:
            continue

        print(f"Bad tile: {fname} — values {np.unique(lbl[lbl >= 23]).tolist()}")

        if "label_mask" in d:
            sjv = d["label_mask"].astype(np.int32)
            new_lbl = lbl.copy()
            bad = new_lbl >= 23
            for code, cls in SJV_TO_UNIFIED.items():
                new_lbl[bad & (sjv == code)] = cls
            # Any still-bad pixels → övrig åker (21)
            remaining = new_lbl >= 23
            if remaining.any():
                print(f"  {remaining.sum()} pixels unmapped → övrig åker (21)")
                new_lbl[remaining] = 21
            d["label"] = new_lbl.astype(np.uint8)
            np.savez_compressed(path, **d)
            print(f"  Fixed → {np.unique(new_lbl).tolist()}")
            fixed += 1
        else:
            # No label_mask to remap from — zero out label so tile is excluded
            d["label"] = np.zeros(lbl.shape, dtype=np.uint8)
            np.savez_compressed(path, **d)
            print(f"  No label_mask — zeroed out (excluded from training)")
            deleted += 1

    except Exception as e:
        print(f"Error on {fname}: {e}")

print(f"\nDone: {fixed} fixed, {deleted} zeroed out")
