#!/usr/bin/env python3
"""Rank the ladder on field truth — NFI and LUCAS — on identical ground.

Each cell's validator wrote a per-plot (NFI) and per-point (LUCAS) dump. This
scores them against each other under two fairness rules the per-cell reports
cannot enforce on their own:

**Same points.** Backbones run at different ``img_size`` (504 or 496), so each
centre-crops a different window and keeps a different subset of plots. Columns
are therefore scored on the INTERSECTION of points every cell kept, never on
each cell's own subset.

**Same vocabulary.** Rung 1 is 23-class; rungs 2-4 are 28-class. The extra
classes 23-27 are not new concepts — they are NMD2023's finer subdivisions of
open land. The cross-rung LUCAS table therefore folds 23-27 onto class 8 in
BOTH truth and prediction, so "shrub-dominated open land" scores against
"open land" as the agreement it is. Restricting only the truth, as this first
did, quietly favours rung 1: a 28-class model can answer 24 where the truth is
8 and be marked wrong, while a 23-class model cannot make that mistake at all.
The full 28-class table is reported separately for rungs 2-4.

NFI needs no vocabulary correction: its 5-class forest collapse maps classes
1-4, which both vocabularies contain.

    python scripts/ladder_fieldtruth_standings.py \
        --dump-dir /cephfs/ladder_eval/fieldtruth
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_against_nfi import accuracy_suite  # noqa: E402
from imint.eval.fieldtruth import (  # noqa: E402
    shared_observations, load_frozen_holdout, restrict_to_frozen, verify_prediction_dump,
    verify_evaluation_source,
)

# The 23-class unified vocabulary is 0..22; 28-class adds 23..27, which are
# NMD2023-only FINE SUBDIVISIONS of open land (unified_schema: 24 shrub-,
# 25 dwarf-shrub-, 26 grass-dominated, 27 bare ground; 23 peat extraction).
# They are not new concepts, so the shared vocabulary is reached by collapsing
# them to their parent rather than by discarding the points.
SHARED_MAX_CLASS = 22
OPEN_LAND = 8
FINE_TO_PARENT = {23: OPEN_LAND, 24: OPEN_LAND, 25: OPEN_LAND,
                  26: OPEN_LAND, 27: OPEN_LAND}


def to_shared_vocab(s: "pd.Series") -> "pd.Series":
    """Fold the 28-class-only fine classes onto their 23-class parent.

    Applied to BOTH truth and prediction. Restricting only the truth — the
    first attempt here — silently favours rung 1: a 28-class model can answer
    24 on a point whose truth is 8 and be marked wrong, while a 23-class model
    cannot make that mistake at all. Collapsing both sides scores "shrub-
    dominated open land" against "open land" as the agreement it is, and keeps
    every point instead of dropping the 7.1% whose truth is 24 or 27.
    """
    return s.replace(FINE_TO_PARENT)


NFI_KEY = ["TractID", "PlotID", "Year"]
CELL_RE = re.compile(r"-(?P<cell>[a-z0-9]+_r[1-4])\.parquet$")


def load_dumps(dump_dir: Path, prefix: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for p in sorted(dump_dir.glob(f"{prefix}-*.parquet")):
        m = CELL_RE.search(p.name)
        if m:
            out[m.group("cell")] = pd.read_parquet(p)
    return out


def score_nfi(dumps: dict[str, pd.DataFrame]) -> list[dict]:
    dumps = shared_observations(dumps, NFI_KEY, "nfi_forest", "model_pred")
    rows = []
    for cell, df in dumps.items():
        d = df
        # nfi_forest == -1 is the producer's TREELESS plot, not a missing
        # observation, and the five-class scoring maps it to class 0. Dropping
        # these would forgive every false forest positive: a model that calls
        # a treeless plot forest would simply not be asked about it. The
        # producer scores one correct forest plot and one wrongly-forested
        # treeless plot as 0.5 over n=2; filtering gives 1.0 over n=1.
        truth = d["nfi_forest"].to_numpy()
        truth = np.where(truth < 0, 0, truth)
        s = accuracy_suite(truth, d["model_pred"].to_numpy())
        rows.append({"cell": cell, "n": int(len(d)),
                     "overall": s["overall_accuracy_5class"],
                     "kappa": s["cohen_kappa"], "per_class": s["per_class"]})
    rows.sort(key=lambda r: -(r["overall"] or 0))
    return rows


def score_lucas(dumps: dict[str, pd.DataFrame], *, shared_only: bool,
                cells: list[str] | None = None) -> list[dict]:
    sel = {c: d for c, d in dumps.items() if cells is None or c in cells}
    if not sel:
        return []
    sel = shared_observations(sel, ["point_id", "Year"],
                              "unified_class", "pred_class")
    rows = []
    for cell, df in sel.items():
        d = df.copy()
        if shared_only:
            d["unified_class"] = to_shared_vocab(d["unified_class"])
            d["pred_class"] = to_shared_vocab(d["pred_class"])
        n = int(len(d))
        if not n:
            continue
        correct = int((d["unified_class"] == d["pred_class"]).sum())
        per_class = {}
        for c, g in d.groupby("unified_class"):
            recall = float((g["pred_class"] == c).mean())
            pred_c = int((d["pred_class"] == c).sum())
            prec = (float(((d["pred_class"] == c) & (d["unified_class"] == c)).sum())
                    / pred_c) if pred_c else None
            per_class[int(c)] = {"support": int(len(g)),
                                 "recall": round(recall, 4),
                                 "precision": round(prec, 4) if prec is not None else None}
        rows.append({"cell": cell, "n": n,
                     "overall": round(correct / n, 4),
                     "classes": len(per_class), "per_class": per_class})
    rows.sort(key=lambda r: -(r["overall"] or 0))
    return rows


def table(title: str, rows: list[dict], note: str = "") -> None:
    print(f"\n=== {title} ===")
    if note:
        print(f"    {note}")
    if not rows:
        print("    (inga dumpar)")
        return
    has_k = "kappa" in rows[0]
    header = f"    {'cell':<18}{'n':>7}{'overall':>10}"
    print(header + (f"{'kappa':>9}" if has_k else ""))
    for r in rows:
        line = f"    {r['cell']:<18}{r['n']:>7}{r['overall']:>10.4f}"
        if has_k:
            line += f"{r['kappa']:>9.4f}"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump-dir", type=Path,
                    default=Path("/cephfs/ladder_eval/fieldtruth"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--holdout-manifest", type=Path,
                    help="verified NFI freeze; otherwise results are diagnostics")
    ap.add_argument("--expected-manifest-sha256", help="SHA256 approved at go/no-go")
    args = ap.parse_args()

    nfi = load_dumps(args.dump_dir, "nfi-per-plot")
    lucas = load_dumps(args.dump_dir, "lucas-per-point")
    print(f"NFI dumps: {len(nfi)} cells   LUCAS dumps: {len(lucas)} cells")

    freeze = None
    if args.holdout_manifest:
        holdout, freeze = load_frozen_holdout(args.holdout_manifest, args.expected_manifest_sha256)
        verify_evaluation_source(freeze, environment="scoring")
        expected = set(freeze["cells"])
        if set(nfi) != expected:
            raise ValueError("NFI dump cells do not match the frozen evaluation")
        for cell in nfi:
            nfi[cell] = verify_prediction_dump(
                args.dump_dir / f"nfi-per-plot-{cell}.parquet",
                args.holdout_manifest, cell, manifest=freeze)
        nfi = {cell: restrict_to_frozen(d, holdout) for cell, d in nfi.items()}
    nfi_rows = score_nfi(nfi) if nfi else []
    lucas_shared = score_lucas(lucas, shared_only=True) if lucas else []
    r234 = [c for c in lucas if not c.endswith("_r1")]
    lucas_full = score_lucas(lucas, shared_only=False, cells=r234) if lucas else []

    table("NFI — forest type, 5-class collapse", nfi_rows,
          "all 28 cells comparable: classes 1-4 exist in both vocabularies")
    table("LUCAS — shared vocabulary (23-27 folded onto open land)", lucas_shared,
          "all cells on identical points; both truth AND prediction collapsed, "
          "so a 28-class model is not charged for finer-but-correct answers")
    table("LUCAS — full 28-class (rungs 2-4 only)", lucas_full,
          "rung 1 omitted: 23-class output cannot address classes 24/27")

    payload = {"schema": "ladder-fieldtruth-standings-v2",
               "holdout_manifest": str(args.holdout_manifest) if freeze else None,
               "manifest_sha256": freeze["_manifest_sha256"] if freeze else None,
               "interpretation": {"nfi": "held-out" if freeze else "diagnostic",
                                  "lucas": "diagnostic"},
               "nfi": nfi_rows, "lucas_shared_vocab": lucas_shared,
               "lucas_full_28class": lucas_full}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
