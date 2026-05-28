#!/usr/bin/env python3
"""Build every train/val/test split needed by the eval suite.

Reads the tiles directory once, generates all axis-specific splits, and
writes them to ``/checkpoints/eval_splits/<axis>_<variant>/`` so the
eval orchestrator can pick them up later.

Run once per dataset version; subsequent eval runs reuse the existing
splits to keep numbers comparable across model iterations.

Usage::

    python scripts/prepare_eval_splits.py \
        --tiles-dir /cephfs/unified_v2_512 \
        --output-root /checkpoints/eval_splits \
        --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiles-dir", required=True,
                   help="Source tile .npz directory.")
    p.add_argument("--output-root", required=True,
                   help="Where to write split definitions.")
    p.add_argument("--seed", type=int, default=42,
                   help="Deterministic seed for stratified sampling.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Don't overwrite existing split dirs.")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # from imint.eval import splits as S

    # TODO:
    #   metadata = S.load_tile_metadata(args.tiles_dir)
    #   S.make_in_distribution_split(...) → write
    #   for variant in chronological_holdout, next_year, year_loo:
    #       S.make_temporal_split(...) → write
    #   for axis in north_south, coast_inland, skog_slatt:
    #       S.make_geographic_split(...) → write
    #   etc.
    raise NotImplementedError(
        "Skeleton — fill once load_tile_metadata is implemented."
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
