"""Scan all tiles in unified_v2 and report out-of-range label values."""
import os
import sys
import numpy as np
import collections

data_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/unified_v2"
files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))

bad = []
vals = collections.Counter()
no_label = 0

for f in files:
    try:
        d = np.load(os.path.join(data_dir, f), allow_pickle=False)
        if "label" not in d:
            no_label += 1
            continue
        lbl = d["label"].astype(np.int32)
        u = np.unique(lbl)
        vals.update(u.tolist())
        if int(u.max()) >= 23:
            bad.append((f, u[u >= 23].tolist()))
    except Exception as e:
        pass

print(f"Scanned:    {len(files):,} tiles")
print(f"No label:   {no_label:,}")
print(f"Bad tiles (label >= 23): {len(bad):,}")
if bad:
    print("\nFirst 40 bad tiles:")
    for t, v in bad[:40]:
        print(f"  {t}: {v}")

print("\nAll label values seen:")
for k in sorted(vals):
    print(f"  {k:3d}: {vals[k]:>10,}")
