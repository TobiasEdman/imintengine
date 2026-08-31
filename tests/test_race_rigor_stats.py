"""Tests for scripts/race_rigor_stats.py — the TOST/McNemar/bootstrap core."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from race_rigor_stats import (  # noqa: E402
    block_bootstrap_diff,
    block_ids,
    compare_pair,
    holm,
    mcnemar_midp,
    wilson_ci,
)


def _dump(preds: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    n = len(preds)
    df = pd.DataFrame({
        "TractID": np.arange(n) // 4,
        "PlotID": np.arange(n) % 4,
        "Year": 2022,
        "Easting": 300_000 + (np.arange(n) % 30) * 60_000,
        "Northing": 6_200_000 + (np.arange(n) // 30) * 60_000,
        "nfi_forest": truth,
        "model_pred": preds,
    })
    df["correct"] = (df["model_pred"] == df["nfi_forest"]).astype(int)
    return df.set_index(["TractID", "PlotID", "Year"])


def test_wilson_ci_contains_point_estimate():
    lo, hi = wilson_ci(80, 100)
    assert lo < 0.8 < hi
    assert 0.70 < lo and hi < 0.88


def test_mcnemar_midp_symmetric_and_null():
    assert mcnemar_midp(0, 0) == 1.0
    assert mcnemar_midp(5, 5) == pytest.approx(mcnemar_midp(5, 5))
    # Strong asymmetry → small p (25 vs 2 discordant).
    assert mcnemar_midp(25, 2) < 0.001
    # Symmetry in arguments.
    assert mcnemar_midp(3, 9) == pytest.approx(mcnemar_midp(9, 3))


def test_mcnemar_midp_matches_exact_binomial():
    # b=1, c=7, n=8: mid-p = 2*(P(X<1) + 0.5*P(X=1)), X~Bin(8, .5)
    expect = 2 * ((1 / 256) + 0.5 * (8 / 256))
    assert mcnemar_midp(1, 7) == pytest.approx(expect)


def test_holm_adjusts_and_preserves_order():
    adj = holm([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03)   # 3 * 0.01
    assert all(a <= 1.0 for a in adj)
    # Monotone in the sorted sense: smallest raw p gets smallest adjusted.
    assert adj[0] == min(adj)


def test_identical_models_are_equivalent():
    rng = np.random.default_rng(0)
    truth = rng.integers(1, 5, 400)
    preds = truth.copy()
    preds[:100] = (preds[:100] % 4) + 1  # 75% OA, same errors both models
    a, b = _dump(preds, truth), _dump(preds.copy(), truth)
    res = compare_pair(a, b, sesoi=0.02, block_km=50, n_boot=2000,
                       rng=np.random.default_rng(1))
    assert res["oa_diff"] == 0.0
    assert res["mcnemar_midp"] == 1.0
    assert res["tost_equivalent"] is True


def test_clearly_different_models_are_not_equivalent():
    rng = np.random.default_rng(0)
    truth = rng.integers(1, 5, 400)
    good = truth.copy()
    bad = truth.copy()
    bad[:120] = (bad[:120] % 4) + 1  # 30% of plots flipped wrong
    res = compare_pair(_dump(good, truth), _dump(bad, truth),
                       sesoi=0.02, block_km=50, n_boot=2000,
                       rng=np.random.default_rng(1))
    assert res["oa_diff"] == pytest.approx(0.30)
    assert res["mcnemar_midp"] < 1e-6
    assert res["tost_equivalent"] is False
    assert res["diff_ci95"][0] > 0  # CI excludes zero


def test_pair_alignment_uses_intersection():
    rng = np.random.default_rng(2)
    truth = rng.integers(1, 5, 200)
    a = _dump(truth.copy(), truth)
    b = _dump(truth.copy(), truth).iloc[:150]  # 50 plots missing (à la 300M)
    res = compare_pair(a, b, sesoi=0.02, block_km=50, n_boot=500,
                       rng=np.random.default_rng(3))
    assert res["n_shared"] == 150


def test_block_bootstrap_centers_on_true_diff():
    rng = np.random.default_rng(4)
    n = 1000
    corr_a = (rng.random(n) < 0.60).astype(int)
    corr_b = (rng.random(n) < 0.50).astype(int)
    df = _dump(np.ones(n, dtype=int), np.ones(n, dtype=int))
    boots = block_bootstrap_diff(corr_a, corr_b, block_ids(df, 50), 4000,
                                 np.random.default_rng(5))
    true_diff = corr_a.mean() - corr_b.mean()
    assert abs(np.median(boots) - true_diff) < 0.01
    assert np.std(boots) > 0  # blocks actually resampled


def test_block_ids_group_by_grid_cell():
    df = _dump(np.ones(4, dtype=int), np.ones(4, dtype=int))
    df = df.iloc[:4].copy()
    df["Easting"] = [10_000, 20_000, 60_000, 60_000]
    df["Northing"] = [10_000, 20_000, 10_000, 10_000]
    ids = block_ids(df, 50)
    assert ids[0] == ids[1]      # same 50 km cell
    assert ids[0] != ids[2]      # different cell
    assert ids[2] == ids[3]


def test_load_dumps_dedupes_tile_overlap_plots(tmp_path):
    from race_rigor_stats import load_dumps
    df = pd.DataFrame({
        "TractID": [1, 1, 2], "PlotID": [1, 1, 1], "Year": [2022] * 3,
        "Easting": [300_000] * 3, "Northing": [6_200_000] * 3,
        "tile_name": ["b", "a", "a"],  # plot (1,1) dumped from two tiles
        "nfi_forest": [1, 1, 2], "model_pred": [1, 2, 2],
        "p1": [0.5] * 3, "p2": [0.2] * 3, "p3": [0.2] * 3, "p4": [0.1] * 3,
    })
    df.to_parquet(tmp_path / "m1_per_plot.parquet")
    dumps = load_dumps(str(tmp_path / "*_per_plot.parquet"))
    d = dumps["m1"]
    assert len(d) == 2  # deduped to one row per plot
    # Deterministic choice: lowest tile_name ("a") wins → pred 2, wrong.
    assert d.loc[(1, 1, 2022), "correct"] == 0


def test_load_dumps_test_tile_filter_applied_before_dedupe(tmp_path):
    from race_rigor_stats import load_dumps
    # Plot (1,1) falls on train tile "a" (pred right) AND test tile "b"
    # (pred wrong). Unfiltered dedupe picks "a"; the holdout filter must
    # keep the "b" copy instead of dropping the plot entirely.
    df = pd.DataFrame({
        "TractID": [1, 1, 2], "PlotID": [1, 1, 1], "Year": [2022] * 3,
        "Easting": [300_000] * 3, "Northing": [6_200_000] * 3,
        "tile_name": ["a", "b", "a"],
        "nfi_forest": [1, 1, 2], "model_pred": [1, 2, 2],
        "p1": [0.5] * 3, "p2": [0.2] * 3, "p3": [0.2] * 3, "p4": [0.1] * 3,
    })
    df.to_parquet(tmp_path / "m1_per_plot.parquet")
    d = load_dumps(str(tmp_path / "*_per_plot.parquet"),
                   keep_tiles={"b"})["m1"]
    assert len(d) == 1                      # only the test-tile plot
    assert d.iloc[0]["tile_name"] == "b"    # scored from the test-tile dump
    assert d.iloc[0]["correct"] == 0


def test_load_dumps_fraction_collapse_and_treeless(tmp_path):
    """Fraction-head dumps use the calibrated collapse (NOT model_pred), and
    treeless truth (-1) scores as non-forest — the canonical-scoring fix."""
    from race_rigor_stats import load_dumps
    # Plot A: pine by fractions (tall dominant) but model_pred deliberately
    # WRONG (3) — a fraction model must ignore model_pred.
    # Plot B: treeless truth (-1) with zero fractions → predicts non-forest.
    df = pd.DataFrame({
        "TractID": [1, 2], "PlotID": [1, 1], "Year": [2022, 2022],
        "Easting": [300_000, 360_000], "Northing": [6_200_000, 6_200_000],
        "tile_name": ["a", "a"],
        "nfi_forest": [1, -1],            # pine, treeless
        "model_pred": [3, 3],             # wrong on purpose
        "p1": [0.8, 0.0], "p2": [0.1, 0.0], "p3": [0.1, 0.0], "p4": [0.0, 0.0],
    })
    # stem "tessera_frac" is a fraction model in the standings registry.
    df.to_parquet(tmp_path / "tessera_frac_per_plot.parquet")
    d = load_dumps(str(tmp_path / "*_per_plot.parquet"))["tessera_frac"]
    assert len(d) == 2
    assert d.loc[(1, 1, 2022), "correct"] == 1   # pine via fractions, not pred=3
    assert d.loc[(2, 1, 2022), "correct"] == 1   # treeless → non-forest, correct


def test_load_dumps_reproduces_standings_fraction_oa():
    """The fixed harness must reproduce the published fraction OA exactly."""
    import json
    from race_rigor_stats import load_dumps
    for req in ("data/distill/distill_split.json",
                "data/nfi/tessera_frac_per_plot.parquet"):
        if not Path(req).exists():
            pytest.skip(f"local-data test: {req} absent (gitignored, not in CI)")
    split = json.loads(Path("data/distill/distill_split.json").read_text())
    keep = {str(t) for t in split["test_tiles"]}
    d = load_dumps("data/nfi/tessera_frac_per_plot.parquet", keep_tiles=keep)
    oa = d["tessera_frac"]["correct"].mean()
    # Standings/deck cite 0.5885 (n=209 rows); dedupe here → within 1 pp.
    assert abs(oa - 0.5885) < 0.01, f"got {oa:.4f}"
