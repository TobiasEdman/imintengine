#!/usr/bin/env python3
"""Probe enrich_one_tile() on a handful of tiles with full tracebacks.

Use this when enrich_tiles_s1.py reports mass failures but the
aggregate log doesn't surface the root cause. Run it inside the
same environment the job used (fetch pod has `imintengine` cloned
at /workspace and deps installed).

Usage (local):
    python scripts/debug_enrich_s1.py --data-dir /data/unified_v2_512 --n 3

Usage (in k8s pod):
    kubectl exec -n prithvi-training-default <pod> -- python3 \\
        /workspace/imintengine/scripts/debug_enrich_s1.py \\
        --data-dir /data/unified_v2_512 --n 3
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback
from pathlib import Path

# Make the imint package importable when run outside `pip install -e`.
# Priority: IMINT_REPO env var > script's own parent > /workspace/imintengine
_repo_candidates = [
    os.environ.get("IMINT_REPO"),
    str(Path(__file__).resolve().parents[1]),
    "/workspace/imintengine",
]
for _candidate in _repo_candidates:
    if _candidate and (Path(_candidate) / "imint").is_dir():
        sys.path.insert(0, _candidate)
        sys.path.insert(0, str(Path(_candidate) / "scripts"))
        print(f"  Repo root:   {_candidate}")
        break
else:
    print("ERROR: could not find imintengine repo root", file=sys.stderr)
    sys.exit(4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/data/unified_v2_512")
    parser.add_argument("--n", type=int, default=3, help="Tiles to probe")
    parser.add_argument(
        "--tile", default=None,
        help="Specific tile .npz path (overrides --n sampling)",
    )
    args = parser.parse_args()

    if args.tile:
        tiles = [args.tile]
    else:
        tiles = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))[:args.n]

    if not tiles:
        print(f"No .npz tiles found in {args.data_dir}")
        sys.exit(2)

    print(f"Environment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  CWD:    {os.getcwd()}")
    for k in ("CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET"):
        v = os.environ.get(k, "")
        masked = (v[:4] + "…") if v else "<unset>"
        print(f"  {k}: {masked}")
    print(f"  Tiles to probe: {len(tiles)}\n")

    # Import the actual enrich function — the error may surface at import
    try:
        from enrich_tiles_s1 import enrich_one_tile  # scripts/ on path
        from imint.training.cdse_s1 import fetch_s1_scene  # noqa: F401
    except Exception:
        print("IMPORT ERROR:")
        traceback.print_exc()
        sys.exit(3)

    for path in tiles:
        name = Path(path).stem
        print(f"=== {name} ===")
        try:
            result = enrich_one_tile(path, skip_existing=False)
            print(f"  result: {result}")
        except Exception:
            print("  EXCEPTION in enrich_one_tile:")
            traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
