#!/usr/bin/env python3
"""Rank the ladder on field truth — NFI and LUCAS — on identical ground.

Each cell's validator wrote a per-plot (NFI) and per-point (LUCAS) dump. This
scores them against each other under two fairness rules the per-cell reports
cannot enforce on their own:

**Same points.** Backbones run at different ``img_size`` (504 or 496), so each
centre-crops a different window and keeps a different subset of plots. Columns
are therefore scored on the INTERSECTION of points every cell kept, never on
each cell's own subset.

**Same vocabulary.** Rung 1 is 23-class; rungs 2-4 are 28-class. LUCAS carries
truth for classes 24 and 27, which a 23-class model cannot emit — 7.1% of the
points. Scoring rung 1 against those counts structural impossibility as model
error, so the cross-rung LUCAS table is restricted to the shared vocabulary
(classes <= 22), and the full 28-class table is reported separately for rungs
2-4 only.

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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_against_nfi import accuracy_suite  # noqa: E402

# The 23-class unified vocabulary is 0..22; 28-class adds 23..27.
SHARED_MAX_CLASS = 22
NFI_KEY = ["TractID", "PlotID", "Year"]
CELL_RE = re.compile(r"-(?P<cell>[a-z0-9]+_r[1-4])\.parquet$")


def load_dumps(dump_dir: Path, prefix: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for p in sorted(dump_dir.glob(f"{prefix}-*.parquet")):
        m = CELL_RE.search(p.name)
        if m:
            out[m.group("cell")] = pd.read_parquet(p)
    return out


def common_keys(dumps: dict[str, pd.DataFrame], key: list[str]) -> pd.DataFrame:
    """Rows present in EVERY cell's dump, as a key-only frame."""
    shared = None
    for df in dumps.values():
        k = df[key].drop_duplicates()
        shared = k if shared is None else shared.join(
            k.set_index(key), on=key, how="inner")
    return shared if shared is not None else pd.DataFrame(columns=key)


def _restrict(df: pd.DataFrame, keys: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """Keep only rows whose key appears in *keys* (no column widening)."""
    idx = pd.MultiIndex.from_frame(keys[key]) if len(key) > 1 else pd.Index(keys[key[0]])
    cur = pd.MultiIndex.from_frame(df[key]) if len(key) > 1 else pd.Index(df[key[0]])
    return df[cur.isin(idx)]


def score_nfi(dumps: dict[str, pd.DataFrame]) -> list[dict]:
    keys = common_keys(dumps, NFI_KEY)
    rows = []
    for cell, df in dumps.items():
        d = _restrict(df, keys, NFI_KEY)
        d = d[d["nfi_forest"] >= 0]
        s = accuracy_suite(d["nfi_forest"].to_numpy(), d["model_pred"].to_numpy())
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
    keys = common_keys(sel, ["point_id"])
    rows = []
    for cell, df in sel.items():
        d = _restrict(df, keys, ["point_id"])
        if shared_only:
            d = d[d["unified_class"] <= SHARED_MAX_CLASS]
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
    args = ap.parse_args()

    nfi = load_dumps(args.dump_dir, "nfi-per-plot")
    lucas = load_dumps(args.dump_dir, "lucas-per-point")
    print(f"NFI dumps: {len(nfi)} cells   LUCAS dumps: {len(lucas)} cells")

    nfi_rows = score_nfi(nfi) if nfi else []
    lucas_shared = score_lucas(lucas, shared_only=True) if lucas else []
    r234 = [c for c in lucas if not c.endswith("_r1")]
    lucas_full = score_lucas(lucas, shared_only=False, cells=r234) if lucas else []

    table("NFI — held-out forest type, 5-class collapse", nfi_rows,
          "all 28 cells comparable: classes 1-4 exist in both vocabularies")
    table("LUCAS — shared vocabulary (classes <= 22)", lucas_shared,
          "all 28 cells on identical points; 24/27 excluded so rung 1 is not "
          "charged for classes it cannot emit")
    table("LUCAS — full 28-class (rungs 2-4 only)", lucas_full,
          "rung 1 omitted: 23-class output cannot address classes 24/27")

    payload = {"schema": "ladder-fieldtruth-standings-v1",
               "nfi": nfi_rows, "lucas_shared_vocab": lucas_shared,
               "lucas_full_28class": lucas_full}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
