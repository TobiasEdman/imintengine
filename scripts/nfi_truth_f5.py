#!/usr/bin/env python3
"""Five-forest-type NFI truth for the distillation head (IM-023 f5).

Tobias 2026-11-17: barrblandskog (NMD 113, class 28) is refined against the
NFI plots like the four forest types the head already knows, and lövblandad
barrskog (NMD 114) stays blandskog: the two mixed types are handled alike.
The rule is the one the NFI evaluation pre-registered for class 28, so the
head learns the truth it will be scored on:

* truth v1 (``derive_nfi_forest_class``): conifer or deciduous dominance at
  0.7 of the standing volume, pine (incl. contorta) vs spruce by volume,
  else blandskog, treeless -1;
* then ``derive_barrblandskog``: a v1 conifer plot where neither pine nor
  spruce reaches 0.7 of the conifer volume is barrblandskog (28).

Contorta counts as pine, as in the evaluation; its sensitivity is a later
evaluation item.

Subcommands:

``derive``      the plot index with the truth in column ``nfi_forest_f5``
                (-1 treeless, 1-4, 28) and a JSON report of v1 → f5 moves.
                ``extract_plot_features.py --truth-col nfi_forest_f5`` carries
                it into the features parquet.
``same-plots``  refuse unless a features parquet holds exactly the plots of a
                reference one. The frozen NFI holdout excludes every plot-year
                of the R4 head's features parquet; a head trained on the same
                plots stays disjoint from it, one trained on others need not.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from imint.eval.nfi_amendment import AMENDMENT_PROTOCOL, BARRBLANDSKOG, derive_barrblandskog  # noqa: E402
from validate_against_nfi import derive_nfi_forest_class  # noqa: E402

TRUTH_COL = "nfi_forest_f5"
TREELESS = -1
#: The identity of a plot row in a features parquet (extract_plot_features.py).
PLOT_KEYS = ("TractID", "PlotID", "point_id", "tile_name")


def truths(index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Truth v1 per plot row, and f5: v1 with its mixed-conifer plots moved to barrblandskog."""
    dominant = AMENDMENT_PROTOCOL["truth_dominant_fraction"]
    classes = (derive_nfi_forest_class(row, dominant_frac=dominant) for _, row in index.iterrows())
    v1 = np.array([TREELESS if c is None else c for c in classes], dtype=np.int64)
    mixed = derive_barrblandskog(index.assign(nfi_forest=v1),
                                 species_fraction=AMENDMENT_PROTOCOL["barrblandskog_species_fraction"])
    return v1, np.where(mixed, BARRBLANDSKOG, v1)


def derive(plot_index: Path, out: Path, report: Path) -> dict:
    if out.exists():
        raise SystemExit(f"{out} exists: write the truth to a new file")
    index = pd.read_parquet(plot_index)
    if TRUTH_COL in index.columns:
        raise SystemExit(f"{plot_index} already has a {TRUTH_COL} column")
    v1, truth = truths(index)
    out.parent.mkdir(parents=True, exist_ok=True)
    index.assign(**{TRUTH_COL: truth}).to_parquet(out, index=False)
    summary = {
        "plot_index": str(plot_index), "out": str(out), "truth_col": TRUTH_COL,
        "rule": {"truth_dominant_fraction": AMENDMENT_PROTOCOL["truth_dominant_fraction"],
                 "barrblandskog_species_fraction": AMENDMENT_PROTOCOL["barrblandskog_species_fraction"],
                 "barrblandskog_truth": AMENDMENT_PROTOCOL["barrblandskog_truth"],
                 "contorta": "counted as pine"},
        "rows": len(index),
        "f5_counts": {str(k): v for k, v in sorted(Counter(truth.tolist()).items())},
        "moves_v1_to_f5": {f"{a}->{b}": n for (a, b), n in sorted(Counter(zip(v1.tolist(), truth.tolist())).items())
                           if a != b},
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def plot_keys(path: Path) -> Counter:
    frame = pd.read_parquet(path, columns=list(PLOT_KEYS))
    return Counter(map(tuple, frame.astype(str).itertuples(index=False, name=None)))


def same_plots(features: Path, reference: Path) -> None:
    ours, theirs = plot_keys(features), plot_keys(reference)
    if ours != theirs:
        extra, missing = sorted((ours - theirs).elements()), sorted((theirs - ours).elements())
        raise SystemExit(f"{features} differs from {reference}: {len(extra)} plot rows not in the "
                         f"reference, e.g. {extra[:3]}; {len(missing)} missing, e.g. {missing[:3]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("derive", help="add the five-type truth to a plot index")
    d.add_argument("--plot-index", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True, help="a new parquet")
    d.add_argument("--report", type=Path, required=True)
    s = sub.add_parser("same-plots", help="refuse a features parquet with other plots than the reference")
    s.add_argument("--features", type=Path, required=True)
    s.add_argument("--reference", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.command == "derive":
        print(json.dumps(derive(args.plot_index, args.out, args.report), indent=2))
    else:
        same_plots(args.features, args.reference)
        print(f"same plots as {args.reference}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
