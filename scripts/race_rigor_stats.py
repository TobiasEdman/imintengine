#!/usr/bin/env python3
"""Rigor statistics for the FM race table (publication_plan.md §6).

Turns per-plot prediction dumps (``validate_against_nfi.py --dump-per-plot``)
into the statistics the adversarial review mandated (2026-08-14): every "tie"
must be earned via equivalence testing, not inferred from an underpowered
null.

Scoring is the SAME canonical path as ``model_race_standings.py`` — 5-class
suite space {0 non-forest, 1-4 forest type}, treeless truth (-1) → non-forest,
fraction-head models collapsed with the calibrated Trädslag rule (floor 0.05 /
dom 0.6), hard models by 28→5 argmax. Scoring raw ``model_pred`` (as an earlier
version did) understates every fraction model, so per-model OA here reproduces
the standings' fraction numbers (Tessera 0.5885, Prithvi-600M 0.5789).

For each model pair this script reports

- exact McNemar (mid-p) on the plot intersection,
- paired OA difference with a spatially-blocked bootstrap 95% CI
  (blocks = ``--block-km`` grid cells in EPSG:3006, resampled with
  replacement — plots in a tract/block are not independent),
- a TOST verdict: *equivalent* iff the 90% blocked-bootstrap CI of the
  difference lies inside ±SESOI (two one-sided tests at α=0.05),
- Holm-corrected McNemar p-values across the whole comparison family.

Verdicts per pair: DIFFERENT (Holm-significant), EQUIVALENT (TOST pass),
INCONCLUSIVE (neither — the underpowered-null case the review flagged).

Usage:
    python scripts/race_rigor_stats.py \
        --nfi-glob 'data/nfi/*_per_plot.parquet' \
        --sesoi 0.02 --block-km 50 \
        --out-json data/distill/race_rigor_stats.json \
        --out-md data/distill/race_rigor_stats.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble_stack import (  # noqa: E402
    suite_space_truth, tradslag_reference_correct,
)
from model_race_standings import MODELS, collapse_hard_28_to_5  # noqa: E402

_KEY = ["TractID", "PlotID", "Year"]
_TOST_ALPHA = 0.05  # TOST at α=0.05 ⇔ 90% CI inside ±SESOI

# dump stem → collapse rule, from the standings registry (single source).
_COLLAPSE = {stem: collapse for _, stem, _, collapse in MODELS}


def _model_name(path: str) -> str:
    return Path(path).name.replace("_per_plot.parquet", "")


def _canonical_correct(df: pd.DataFrame, collapse: str) -> np.ndarray:
    """Per-plot correctness in 5-class suite space, matching the standings.

    ``collapse='fraction'`` applies the calibrated Trädslag collapse on the
    p1–p4 fraction channels; ``'hard'`` collapses the 28-class argmax to the
    forest 5-space. Truth is ``suite_space_truth`` (treeless → non-forest).
    """
    y = suite_space_truth(df["nfi_forest"].to_numpy())
    if collapse == "fraction":
        tr = df.rename(columns={c: f"tradslag__{c}" for c in ("p1", "p2", "p3", "p4")})
        correct, _ = tradslag_reference_correct(tr, np.ones(len(tr), bool), y)
        return correct
    pred = collapse_hard_28_to_5(df["model_pred"].to_numpy())
    return (pred == y).astype(np.int64)


def load_dumps(pattern: str,
               keep_tiles: set[str] | None = None,
               collapse_map: dict[str, str] | None = None,
               ) -> dict[str, pd.DataFrame]:
    """Load per-plot dumps, score them canonically, index by plot key.

    Every plot is kept (treeless plots score as non-forest, class 0) so the
    denominator matches the standings. ``keep_tiles`` restricts to the
    held-out test tiles — applied BEFORE dedupe, so a plot that also falls on
    a train tile is scored from its test-tile dump, not silently dropped to
    the train-tile copy. A stem absent from the standings registry defaults
    to hard collapse (28→5 argmax).
    """
    cmap = _COLLAPSE if collapse_map is None else collapse_map
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob(pattern)):
        stem = _model_name(path)
        df = pd.read_parquet(path)
        if keep_tiles is not None:
            df = df[df["tile_name"].astype(str).isin(keep_tiles)].copy()
        # Plots in tile-overlap zones are dumped once per tile — keep one
        # row per plot, chosen deterministically so pairing is consistent
        # across models (all dumps share the same tile index). Dedupe for
        # McNemar independence (the standings keeps duplicates for its OA).
        df = (df.sort_values("tile_name")
                .drop_duplicates(subset=_KEY, keep="first")
                .reset_index(drop=True))
        collapse = cmap.get(stem, "hard")
        df["correct"] = _canonical_correct(df, collapse)
        out[stem] = df.set_index(_KEY)
    if not out:
        raise SystemExit(f"no per-plot dumps match {pattern!r}")
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def mcnemar_midp(b: int, c: int) -> float:
    """Exact McNemar mid-p on the discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k)) / 2 ** n
    point = math.comb(n, k) / 2 ** n
    return min(1.0, 2 * (tail + 0.5 * point))


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, order-preserving."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def block_ids(df: pd.DataFrame, block_km: float) -> np.ndarray:
    """Spatial block id per plot: EPSG:3006 grid cell of side block_km."""
    size = block_km * 1000.0
    bx = np.floor(df["Easting"].to_numpy() / size).astype(np.int64)
    by = np.floor(df["Northing"].to_numpy() / size).astype(np.int64)
    return bx * 100_000 + by


def block_bootstrap_diff(
    corr_a: np.ndarray,
    corr_b: np.ndarray,
    blocks: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap OA(a)−OA(b) by resampling spatial blocks with replacement."""
    uniq = np.unique(blocks)
    # Per-block sufficient statistics — resampling then only needs sums.
    n_by = np.array([(blocks == u).sum() for u in uniq], dtype=np.int64)
    ka_by = np.array([corr_a[blocks == u].sum() for u in uniq], dtype=np.int64)
    kb_by = np.array([corr_b[blocks == u].sum() for u in uniq], dtype=np.int64)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    n_s = n_by[idx].sum(axis=1)
    return (ka_by[idx].sum(axis=1) - kb_by[idx].sum(axis=1)) / n_s


def compare_pair(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    sesoi: float,
    block_km: float,
    n_boot: int,
    rng: np.random.Generator,
) -> dict | None:
    shared = a.index.intersection(b.index)
    if len(shared) == 0:
        return None
    a, b = a.loc[shared], b.loc[shared]
    ca, cb = a["correct"].to_numpy(), b["correct"].to_numpy()
    n01 = int(((ca == 0) & (cb == 1)).sum())  # a wrong, b right
    n10 = int(((ca == 1) & (cb == 0)).sum())  # a right, b wrong
    boots = block_bootstrap_diff(ca, cb, block_ids(a, block_km), n_boot, rng)
    lo95, hi95 = np.quantile(boots, [0.025, 0.975])
    lo90, hi90 = np.quantile(boots, [_TOST_ALPHA, 1 - _TOST_ALPHA])
    return {
        "n_shared": int(len(shared)),
        "oa_diff": float(ca.mean() - cb.mean()),
        "discordant": {"a_only_wrong": n01, "b_only_wrong": n10},
        "mcnemar_midp": mcnemar_midp(n01, n10),
        "diff_ci95": [float(lo95), float(hi95)],
        "diff_ci90": [float(lo90), float(hi90)],
        "tost_equivalent": bool(-sesoi < lo90 and hi90 < sesoi),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nfi-glob", default="data/nfi/*_per_plot.parquet")
    ap.add_argument("--split-json", default="data/distill/distill_split.json",
                    help="tile split definition (train_tiles/test_tiles)")
    ap.add_argument("--split", choices=["test", "all"], default="test",
                    help="'test' = held-out tiles only (publication-valid); "
                         "'all' includes training-tile plots (contaminated — "
                         "machinery checks only)")
    ap.add_argument("--sesoi", type=float, default=0.02,
                    help="smallest OA difference that matters (fraction)")
    ap.add_argument("--block-km", type=float, default=50.0)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--out-json", default="data/distill/race_rigor_stats.json")
    ap.add_argument("--out-md", default="data/distill/race_rigor_stats.md")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    keep_tiles = None
    if args.split == "test":
        split = json.loads(Path(args.split_json).read_text())
        keep_tiles = {str(t) for t in split["test_tiles"]}
    dumps = load_dumps(args.nfi_glob, keep_tiles=keep_tiles)

    per_model = {}
    for name, df in dumps.items():
        k, n = int(df["correct"].sum()), len(df)
        lo, hi = wilson_ci(k, n)
        per_model[name] = {"n": n, "oa": k / n, "oa_ci95": [lo, hi]}

    names = sorted(per_model, key=lambda m: -per_model[m]["oa"])
    pairs: dict[str, dict] = {}
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            res = compare_pair(
                dumps[na], dumps[nb], sesoi=args.sesoi,
                block_km=args.block_km, n_boot=args.n_boot, rng=rng)
            if res is not None:
                pairs[f"{na} vs {nb}"] = res

    keys = list(pairs)
    for key, p_adj in zip(keys, holm([pairs[k]["mcnemar_midp"] for k in keys])):
        pairs[key]["mcnemar_holm"] = p_adj
        if p_adj < 0.05:
            verdict = "DIFFERENT"
        elif pairs[key]["tost_equivalent"]:
            verdict = "EQUIVALENT"
        else:
            verdict = "INCONCLUSIVE"
        pairs[key]["verdict"] = verdict

    result = {
        "config": {"sesoi": args.sesoi, "block_km": args.block_km,
                   "n_boot": args.n_boot, "seed": args.seed,
                   "nfi_glob": args.nfi_glob, "split": args.split,
                   "split_json": args.split_json if keep_tiles else None,
                   "n_test_tiles": len(keep_tiles) if keep_tiles else None},
        "per_model": per_model,
        "pairs": pairs,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2))

    split_note = (f"held-out test tiles only ({len(keep_tiles)} tiles)"
                  if keep_tiles else
                  "ALL tiles incl. training — contaminated, not for publication")
    lines = [
        "# Race rigor statistics (NFI per-plot)",
        "",
        f"Split: {split_note} · SESOI ±{args.sesoi:.3f} · "
        f"{args.block_km:.0f} km spatial blocks · "
        f"{args.n_boot} bootstrap draws · McNemar mid-p, Holm-corrected",
        "",
        "| Model | n | OA | 95% CI |",
        "|---|---|---|---|",
    ]
    for m in names:
        s = per_model[m]
        lines.append(f"| {m} | {s['n']} | {s['oa']:.4f} | "
                     f"[{s['oa_ci95'][0]:.4f}, {s['oa_ci95'][1]:.4f}] |")
    lines += [
        "",
        "| Pair | n | ΔOA | 95% CI | McNemar (Holm) | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for key in keys:
        p = pairs[key]
        lines.append(
            f"| {key} | {p['n_shared']} | {p['oa_diff']:+.4f} | "
            f"[{p['diff_ci95'][0]:+.4f}, {p['diff_ci95'][1]:+.4f}] | "
            f"{p['mcnemar_holm']:.3f} | {p['verdict']} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_json} + {args.out_md} "
          f"({len(per_model)} models, {len(pairs)} pairs)")


if __name__ == "__main__":
    main()
