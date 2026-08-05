#!/usr/bin/env python3
"""Score NMD against SLU NFI field truth on the same plots as the model.

External reference point for the model's NFI validation
(``validate_against_nfi.py``): sample the NMD base raster at each NFI plot's
Easting/Northing, map it through the repo's own NMD→unified forest classes,
and score forest-type agreement against the same species-derived NFI truth the
model is scored on. This makes ``forest_type_accuracy`` directly comparable
between the model and NMD — the label source NMD is, itself, measured against
the independent field truth.

NMD is a hard classification (no softmax), so this reports accuracy +
per-class recall + confusion, but no AUROC (that needs probabilities).

Usage (ICE, NMD raster + NFI index on the PVC):
    python scripts/benchmark_nmd_vs_nfi.py \
        --nmd-dir /data/nmd \
        --plot-index /data/nfi/nfi_index_unified_v2_512.parquet \
        --out /data/nfi_eval/benchmark_nmd_vs_nfi.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_against_nfi import (  # noqa: E402
    derive_nfi_forest_class, FOREST_CLASSES, FOREST_NAMES, accuracy_suite,
)
from imint.training.class_schema import nmd_raster_to_lulc  # noqa: E402
from imint.training.unified_schema import nmd19_to_unified  # noqa: E402


def score_nmd(nmd_path: str, index_df: pd.DataFrame,
              dominant_frac: float = 0.7) -> dict:
    """Sample NMD at plot Eastings/Northings and score vs NFI forest type."""
    import rasterio

    with rasterio.open(nmd_path) as src:
        coords = list(zip(index_df["Easting"].astype(float),
                          index_df["Northing"].astype(float)))
        raw = np.array([v[0] for v in src.sample(coords)], dtype=np.int64)

    seq = nmd_raster_to_lulc(np.clip(raw, 0, 255).astype(np.uint8))
    pred = nmd19_to_unified(seq.astype(np.uint8)).astype(int)
    truth = np.array([
        (c if (c := derive_nfi_forest_class(r, dominant_frac=dominant_frac))
         is not None else -1)
        for _, r in index_df.iterrows()
    ])

    forest = truth >= 1
    n_forest = int(forest.sum())
    acc = float((pred[forest] == truth[forest]).mean()) if n_forest else float("nan")
    recall = {
        FOREST_NAMES[t]: round(float((pred[truth == t] == t).mean()), 4)
        for t in FOREST_CLASSES if (truth == t).any()
    }
    confusion = {
        FOREST_NAMES[t]: {
            FOREST_NAMES.get(int(p), f"class_{int(p)}"):
                int(((truth == t) & (pred == p)).sum())
            for p in np.unique(pred[truth == t])
        }
        for t in FOREST_CLASSES if (truth == t).any()
    }
    return {
        "source": f"NMD ({Path(nmd_path).stem})",
        "n_plots": int(len(index_df)),
        "n_forest": n_forest,
        "forest_type_accuracy": round(acc, 4),
        "per_class_recall": recall,
        "confusion_nfi_x_nmd": confusion,
        # Standard confusion-matrix suite over ALL plots — same as the model
        # harness (validate_against_nfi.accuracy_suite), so v8/v8b and NMD are
        # directly comparable on user's/producer's/F1/overall/kappa.
        "accuracy_suite": accuracy_suite(list(truth), list(pred)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nmd-dir", default="/data/nmd")
    ap.add_argument("--plot-index", required=True)
    ap.add_argument("--out", default="docs/data/benchmark-nmd-vs-nfi.json")
    a = ap.parse_args()

    nmd_path = sorted(glob.glob(str(Path(a.nmd_dir) / "*.tif")))[0]
    index_df = pd.read_parquet(a.plot_index)
    print(f"NMD: {nmd_path}  plots: {len(index_df)}")

    result = score_nmd(nmd_path, index_df)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
