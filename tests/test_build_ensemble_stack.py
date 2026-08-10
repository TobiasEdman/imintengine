"""tests/test_build_ensemble_stack.py — unit tests for the ensemble combiner.

Covers the leakage-free / statistical-honesty helpers that carry the load:
  * McNemar exact test (discordant counting + degenerate cases).
  * paired bootstrap CI on the accuracy delta (sign, zero-span).
  * G1 verdict language (within-noise is never a win — §6 discipline).
  * design-matrix R6 tagging (fraction channels only under +fractions).
  * npz port forward reproduces sklearn exactly (logreg + MLP).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_ensemble_stack as B  # noqa: E402


def test_mcnemar_all_agree_is_p1() -> None:
    a = np.array([1, 1, 0, 0, 1])
    r = B.mcnemar_exact(a, a.copy())
    assert r["discordant"] == 0
    assert r["p_value"] == 1.0


def test_mcnemar_counts_discordant_directionally() -> None:
    # combiner (A) right where ref (B) wrong on 3 plots; ref right where
    # combiner wrong on 1 plot.
    a = np.array([1, 1, 1, 0, 1, 1])  # combiner correctness
    b = np.array([0, 0, 0, 1, 1, 1])  # reference correctness
    r = B.mcnemar_exact(a, b)
    assert r["c_combiner_only_right"] == 3
    assert r["b_ref_only_right"] == 1
    assert r["discordant"] == 4


def test_bootstrap_ci_positive_delta_excludes_zero() -> None:
    # A correct on 90/100, B on 50/100, disjoint errors → large positive delta.
    a = np.ones(100, dtype=int)
    a[:10] = 0
    b = np.ones(100, dtype=int)
    b[:50] = 0
    ci = B.bootstrap_ci_delta(a, b, n_resamples=2000)
    assert ci["delta_point"] == pytest.approx(0.40, abs=1e-9)
    assert ci["ci95_low"] > 0.0
    assert not ci["spans_zero"]


def test_bootstrap_ci_equal_accuracy_spans_zero() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, size=200)
    b = a.copy()  # identical → delta exactly 0 everywhere
    ci = B.bootstrap_ci_delta(a, b, n_resamples=2000)
    assert ci["delta_point"] == 0.0
    assert ci["spans_zero"]


def test_verdict_within_noise_when_ci_spans_zero() -> None:
    ci = {"spans_zero": True, "delta_point": 0.05}
    mcn = {"p_value": 0.01}
    assert B.verdict(ci, mcn) == "within_noise"


def test_verdict_within_noise_when_mcnemar_nonsig() -> None:
    ci = {"spans_zero": False, "delta_point": 0.05}
    mcn = {"p_value": 0.2}
    assert B.verdict(ci, mcn) == "within_noise"


def test_verdict_significant_win_needs_both() -> None:
    ci = {"spans_zero": False, "delta_point": 0.08}
    mcn = {"p_value": 0.02}
    assert B.verdict(ci, mcn) == "beats_gate_significant"


def test_verdict_significant_loss() -> None:
    ci = {"spans_zero": False, "delta_point": -0.08}
    mcn = {"p_value": 0.02}
    assert B.verdict(ci, mcn) == "below_gate_significant"


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame({
        "distill__p1": [0.1, 0.2], "distill__p2": [0.3, 0.1],
        "distill__p3": [0.2, 0.4], "distill__p4": [0.4, 0.3],
        "tradslag__p1": [0.5, 0.6], "tradslag__p2": [0.2, 0.1],
        "tradslag__p3": [0.3, 0.3], "tradslag__p4": [0.2, 0.1],
    })


def test_design_matrix_r6_fraction_only_under_plus_fractions() -> None:
    df = _toy_df()
    mem = [B.Member("distill", "hard", 28),
           B.Member("tradslag", "fraction", 28)]
    X_i, cols_i = B.build_design_matrix(df, mem, "hard_p")
    # variant (i): fraction member contributes NO channels (R6 — never mixed
    # untagged with softmax probs).
    assert all(not c.startswith("tradslag__") for c in cols_i)
    assert X_i.shape[1] == 4
    X_ii, cols_ii = B.build_design_matrix(df, mem, "plus_fractions")
    assert any(c.startswith("tradslag__") for c in cols_ii)
    assert X_ii.shape[1] == 8


def test_suite_space_truth_maps_treeless_to_zero() -> None:
    y = B.suite_space_truth(np.array([-1, 1, 2, 3, 4]))
    assert y.tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("head", ["logreg", "mlp"])
def test_npz_port_reproduces_sklearn(tmp_path: Path, head: str) -> None:
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 8)).astype(np.float32)
    y = rng.integers(0, 5, size=120)
    scaler = StandardScaler().fit(X)
    clf = B.build_head(head)
    clf.fit(scaler.transform(X), y)

    out = tmp_path / "combiner.npz"
    B.export_combiner_npz(out, scaler, clf, [f"c{i}" for i in range(8)],
                          "toy", "hard_p", head)
    npz = np.load(out, allow_pickle=True)
    port = B.head_forward_npz(X, npz)
    sk = clf.predict(scaler.transform(X))
    assert np.array_equal(port, sk)


def test_fit_and_eval_scaler_ignores_holdout() -> None:
    """Leakage guard: the scaler in ``fit_and_eval`` is fit on X_tr ONLY.

    Its ``mean_``/``scale_`` must be bit-identical to a ``StandardScaler``
    fit on X_tr alone — i.e. the held-out X_te cannot leak into standardization.
    A regression here (fit on the concatenation, or on X_te) would shift the
    scaler and silently contaminate the honest hold-out number.
    """
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(7)
    X_tr = rng.normal(loc=5.0, scale=3.0, size=(200, 6)).astype(np.float32)
    y_tr = rng.integers(0, 5, size=200)
    # X_te drawn from a deliberately different distribution — if it leaked into
    # the fit, the scaler stats would move.
    X_te = rng.normal(loc=-20.0, scale=50.0, size=(60, 6)).astype(np.float32)
    y_te = rng.integers(0, 5, size=60)

    _, _, scaler, _ = B.fit_and_eval(X_tr, y_tr, X_te, y_te, "logreg")
    ref = StandardScaler().fit(X_tr)
    assert np.array_equal(scaler.mean_, ref.mean_)
    assert np.array_equal(scaler.scale_, ref.scale_)


def _write_dump(path: Path, rows: dict) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _synthetic_dump_rows(nfi_forest: list[int]) -> dict:
    """A minimal per-plot dump with a TractID+PlotID collision built in.

    Rows 0 and 2 share TractID+PlotID (5009/203) but differ in Year/tile_name
    (a re-visit / co-location) — so a TractID+PlotID-only join self-crosses,
    while the 6-tuple stays 1:1.
    """
    return {
        "TractID": [5009, 5009, 5009],
        "PlotID": [203, 206, 203],
        "Year": [2022, 2022, 2019],           # row 0 vs row 2 differ here
        "Easting": [298335, 298319, 298335],
        "Northing": [6489793, 6489487, 6489793],
        "tile_name": ["44123932", "44123932", "38550012"],  # and here
        "nfi_forest": nfi_forest,
        "model_pred": [1, 2, 3],
        "p1": [0.1, 0.2, 0.3], "p2": [0.3, 0.1, 0.2],
        "p3": [0.2, 0.4, 0.1], "p4": [0.4, 0.3, 0.4],
    }


def test_two_tuple_join_explodes_six_tuple_stays_exact(tmp_path: Path) -> None:
    """TractID+PlotID-only join self-crosses the collision; the 6-tuple is 1:1.

    Demonstrates *why* JOIN_KEYS is the 6-tuple: joining two copies of the same
    3-row frame on TractID+PlotID alone produces 3 + 2×2 = 5 rows (the 5009/203
    pair matches both its own copies), whereas the 6-tuple join stays at 3.
    """
    rows = _synthetic_dump_rows([1, 2, 3])
    a = pd.DataFrame(rows)
    b = pd.DataFrame(rows)
    two = a.merge(b, on=["TractID", "PlotID"], how="inner")
    six = a.merge(b, on=list(B.JOIN_KEYS), how="inner")
    assert len(two) == 5          # collision self-crosses (3 + 2 extra)
    assert len(six) == 3          # 6-tuple is unique → exact row count


def test_load_members_six_tuple_join_exact_row_count(tmp_path: Path) -> None:
    """load_members joins two synthetic members on the 6-tuple → exact rows.

    The sibling test proves a TractID+PlotID-only join would explode to 5 on
    this fixture; here we drive the real ``load_members`` (with its row sentinel
    relaxed to the synthetic size) and confirm the 6-tuple join stays at 3,
    namespaces both members' channels, and keeps one shared truth column.
    """
    rows = _synthetic_dump_rows([1, 2, 3])
    _write_dump(tmp_path / "distill_per_plot.parquet", rows)
    _write_dump(tmp_path / "tradslag_per_plot.parquet", rows)
    members = (B.Member("distill", "hard", 28),
               B.Member("tradslag", "fraction", 28))

    base, present = B.load_members(tmp_path, members, expected_rows=3)
    assert len(base) == 3
    assert [m.name for m in present] == ["distill", "tradslag"]
    assert "distill__p1" in base.columns and "tradslag__p1" in base.columns
    assert list(base["nfi_forest"]) == [1, 2, 3]
    # exactly one truth column survives the join (no nfi_forest__*_dup leak)
    assert sum(c.startswith("nfi_forest") for c in base.columns) == 1


def test_load_members_nfi_forest_mismatch_raises(tmp_path: Path) -> None:
    """A per-plot nfi_forest disagreement across dumps → SystemExit.

    This fires inside the merge loop (before the 944 guard), so a 3-row
    synthetic set is enough: the second member reports a different truth on one
    plot → dumps sampled at different pixels → hard fail.
    """
    a_rows = _synthetic_dump_rows([1, 2, 3])
    b_rows = _synthetic_dump_rows([1, 2, 4])   # row 2 disagrees (3 → 4)
    _write_dump(tmp_path / "distill_per_plot.parquet", a_rows)
    _write_dump(tmp_path / "tradslag_per_plot.parquet", b_rows)
    members = (B.Member("distill", "hard", 28),
               B.Member("tradslag", "fraction", 28))
    with pytest.raises(SystemExit, match="nfi_forest disagrees"):
        B.load_members(tmp_path, members)


def test_tradslag_reference_reproduces_gate_on_real_dumps() -> None:
    """Regression: the recomputed Trädslag reference hits the 0.579 gate on the
    real held-out 209. Skips gracefully if the dumps aren't present."""
    import json

    dump_dir = Path(__file__).resolve().parents[1] / "data" / "nfi"
    split_p = (Path(__file__).resolve().parents[1] / "data" / "distill"
               / "distill_split.json")
    if not (dump_dir / "tradslag_per_plot.parquet").exists() or \
            not split_p.exists():
        pytest.skip("real dumps / split not present")
    stack_df, present = B.load_members(dump_dir, B.ALL_MEMBERS)
    if "tradslag" not in {m.name for m in present}:
        pytest.skip("tradslag dump not present")
    split = json.loads(split_p.read_text())
    tt = {str(t) for t in split["test_tiles"]}
    te = stack_df["tile_name"].astype(str).isin(tt).to_numpy()
    y = B.suite_space_truth(stack_df["nfi_forest"].to_numpy())
    _, suite = B.tradslag_reference_correct(stack_df, te, y[te])
    assert suite["overall_accuracy_5class"] == pytest.approx(0.579, abs=0.002)
