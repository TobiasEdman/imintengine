#!/usr/bin/env python3
"""scripts/build_ensemble_stack.py — P2: NFI-stacked ensemble combiner.

Builds a per-plot stack matrix from the member per-plot dumps, trains a small
combiner head (logreg / MLP) on the *train-split* plots ONLY, and scores it on
the 209 held-out plots against the G1 gate (0.579 = the Trädslag fraction
member, kappa-calibrated, on the same held-out set).

Two truths that must not blur (campaign plan §"Mål"):
  * Track A (this script): 5-class forest type on the 209 held-out plots.
  * The reported combiner accuracy is ALWAYS on the 209 — the same set the
    0.579 gate was measured on. Train-OOF is reported separately, never mixed.

Honest hold-out discipline (§Regler): the StandardScaler, the head, and every
selection decision are fit / made on the 735 train plots ONLY. Concretely, ALL
of the following are decided on train-OOF accuracy, never on the held-out set:
  * the best (set, variant) per head (argmax over the 8-config sweep);
  * the R3 logreg-vs-MLP choice (paired bootstrap CI + McNemar on the OOF
    correctness vectors over the 735, not on the 209).
Only after the single (set, variant, head) configuration is locked from
train-OOF is the held-out 209 touched — exactly once — to report the final
number and its McNemar/bootstrap comparison against the 0.579 Trädslag gate.
The full 8-config sweep is still written to the JSON under ``exploratory`` for
transparency, explicitly flagged selection-biased so it is never headlined.

Channel semantics (R6): the member dumps carry ``p1..p4`` whose meaning differs
by member. For the softmax members (v8b*, distill) ``p1..p4`` are the softmax
probabilities of the 4 forest classes sampled at the plot pixel. For the
Trädslag member ``p1..p4`` are the FRACTION-head channels as written by
``validate_against_nfi.make_fraction_predict_fn``: p1=tall crown-cover,
p2=gran crown-cover, p3=deciduous total (trivial+ädel), p4=min(conifer,decid)
mixedness proxy — NOT softmax. Mixing them untagged is forbidden; the member
manifest records ``mode`` (hard|fraction) so a downstream consumer knows which
is which, and the fraction channels are only added in feature variant (ii)+.

Reuses the leakage-free patterns from ``nfi_head_cv`` (StandardScaler fit on
train only) and ``train_distill_head`` (dependency-light npz export + port
check). ``accuracy_suite`` (from ``validate_against_nfi``) is the single scorer,
so combiner numbers are directly comparable to the member baselines.

    python scripts/build_ensemble_stack.py \
        --dump-dir data/nfi --split data/distill/distill_split.json \
        --out-stack data/distill/ensemble_stack.parquet \
        --out-head data/distill/ensemble_combiner.npz \
        --out-json data/distill/ensemble_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_against_nfi import accuracy_suite  # noqa: E402

SEED = 42

# The full identifier tuple that uniquely keys an NFI plot in the dumps. A plot
# can recur under TractID+PlotID alone (same plot re-visited across NFI Years,
# or co-located on two overlapping tiles), so joining on TractID+PlotID ONLY
# would explode 944 → 1470 rows. The dumps are already row-aligned in the same
# order, and this 6-tuple is unique (944/944) — it is the honest join key.
JOIN_KEYS: tuple[str, ...] = (
    "TractID", "PlotID", "Year", "Easting", "Northing", "tile_name",
)

# Forest-class softmax/fraction channels present in every dump.
P_COLS: tuple[str, ...] = ("p1", "p2", "p3", "p4")

# The G1 gate: the Trädslag fraction member (kappa-calibrated floor=0.05/
# dom=0.6) on the 209 held-out plots (docs/data/tradslag_fraction_finding.md).
G1_GATE = 0.579


class BestRun(NamedTuple):
    """One candidate combiner config, kept per head during the train-OOF sweep.

    Selection is on ``oof_oa`` (train out-of-fold accuracy) — ``ho_oa`` and
    ``pred_te`` are carried for the SINGLE post-lock held-out evaluation only,
    never as a selection key. ``oof_correct`` is the per-train-plot OOF
    correctness vector (735-long), the honest basis for the R3 logreg-vs-MLP
    paired test.
    """

    oof_oa: float                 # train out-of-fold accuracy (SELECTION key)
    ho_oa: float                  # held-out accuracy (report only, not selection)
    run: dict                     # the run record (set/variant/head/suites)
    scaler: object                # StandardScaler fit on the 735 train plots
    clf: object                   # fitted sklearn head
    cols: list[str]               # design-matrix feature columns
    pred_te: np.ndarray           # held-out predictions (report only)
    oof_correct: np.ndarray       # per-train-plot OOF correctness (R3 key)


@dataclass(frozen=True)
class Member:
    """One ensemble member and how to read its per-plot dump.

    Attributes:
        name: member id, also the dump stem (``<name>_per_plot.parquet``).
        mode: ``"hard"`` → p1..p4 are softmax probs of the 4 forest classes;
            ``"fraction"`` → p1..p4 are the fraction-head channels
            (tall, gran, decid_total, mixedness) — semantically NOT probs.
        num_classes: the member model's native output width (provenance only).
    """

    name: str
    mode: str  # "hard" | "fraction"
    num_classes: int


# Member registry. Designed as data so Tessera (and later CROMA/TerraMind) are
# one row each — no code change. ``set_all`` / ``set_top2`` are the ratified
# member SETS (decision 1); Tessera appends to whichever set once its dump lands.
ALL_MEMBERS: tuple[Member, ...] = (
    Member("v8b", "hard", 23),
    Member("v8b_markfukt", "hard", 23),
    Member("v8b_nmd2023_long", "hard", 28),
    Member("distill", "hard", 28),
    Member("tradslag", "fraction", 28),
    # Encoder-diversity member (distill labels + frac head on frozen Tessera
    # embeddings). Absent until its dump lands — load_members skips it, so
    # the *_tessera sets silently equal their base until then.
    Member("tessera", "hard", 28),
)

MEMBER_SETS: dict[str, tuple[str, ...]] = {
    # Ratified sets (decision 1) — the tessera-OUT ablation baselines.
    "all5": ("v8b", "v8b_markfukt", "v8b_nmd2023_long", "distill", "tradslag"),
    "top2": ("distill", "tradslag"),
    # tessera-IN sets — G2 ablation is *_tessera vs its base.
    "all5_tessera": ("v8b", "v8b_markfukt", "v8b_nmd2023_long", "distill",
                     "tradslag", "tessera"),
    "top2_tessera": ("distill", "tradslag", "tessera"),
}

# Feature variants (decision 2). Each maps to a predicate over member.mode that
# decides whether that member's p-channels enter the design matrix:
#   (i)   hard-p     : only the softmax members' 4 forest probs.
#   (ii)  +fractions : also the Trädslag fraction channels (R6-tagged).
#   (iii) +tessera   : same as (ii) but includes a tessera member if present —
#                      expressed as a member-set choice, not a channel filter,
#                      so it is a no-op until the tessera dump is added.
FEATURE_VARIANTS: tuple[str, ...] = ("hard_p", "plus_fractions")


def load_members(
    dump_dir: Path, members: tuple[Member, ...], *, expected_rows: int = 944,
) -> tuple[pd.DataFrame, list[Member]]:
    """Join all member dumps on the 6-tuple identifier → one wide frame.

    Verifies the join produces ``expected_rows`` rows (944 for the real NFI
    dumps; overridable for synthetic test fixtures), no NaN in the used
    channels, and that every member reports the identical NFI truth per plot (a
    per-plot ``nfi_forest`` mismatch would mean the dumps were sampled at
    different pixels).
    """
    present: list[Member] = []
    base: pd.DataFrame | None = None
    for m in members:
        path = dump_dir / f"{m.name}_per_plot.parquet"
        if not path.exists():
            print(f"  member {m.name!r}: dump absent ({path}) — skipped")
            continue
        df = pd.read_parquet(path)
        for c in JOIN_KEYS + P_COLS + ("nfi_forest",):
            if c not in df.columns:
                raise SystemExit(f"{path}: missing column {c!r}")
        # Namespace the member channels; keep join keys + one shared truth col.
        ren = {c: f"{m.name}__{c}" for c in P_COLS + ("model_pred",)
               if c in df.columns}
        df = df.rename(columns=ren)[list(JOIN_KEYS) + list(ren.values())
                                    + ["nfi_forest"]]
        if base is None:
            base = df
        else:
            merged = base.merge(
                df, on=list(JOIN_KEYS), how="inner",
                suffixes=("", f"__{m.name}_dup"),
            )
            dup_truth = f"nfi_forest__{m.name}_dup"
            if dup_truth in merged.columns:
                mism = int((merged["nfi_forest"] != merged[dup_truth]).sum())
                if mism:
                    raise SystemExit(
                        f"{m.name}: nfi_forest disagrees with base on {mism} "
                        "plots — dumps not sampled at the same pixels")
                merged = merged.drop(columns=[dup_truth])
            base = merged
        present.append(m)

    if base is None:
        raise SystemExit("no member dumps found")
    if len(base) != expected_rows:
        raise SystemExit(f"join produced {len(base)} rows, expected "
                         f"{expected_rows} — dumps are not row-aligned on the "
                         "6-tuple key")

    used = [f"{m.name}__{c}" for m in present for c in P_COLS]
    nan = base[used].isna().sum().sum()
    if nan:
        raise SystemExit(f"{nan} NaN in member channels after join")
    return base, present


def suite_space_truth(nfi_forest: np.ndarray) -> np.ndarray:
    """NFI truth → 5-class suite space {0 non-forest, 1..4}; -1 (treeless)→0."""
    y = nfi_forest.astype(np.int64)
    return np.where(y == -1, 0, y)


def build_design_matrix(
    df: pd.DataFrame, members: list[Member], variant: str,
) -> tuple[np.ndarray, list[str]]:
    """Assemble X for a feature variant, R6-tagging fraction vs prob channels.

    Variant ``hard_p``: only the softmax members' 4 forest probs.
    Variant ``plus_fractions``: also every fraction member's 4 channels.
    Fraction channels are only ever admitted under ``plus_fractions`` so a
    prob column and a fraction column never sit in the same matrix untagged.
    """
    cols: list[str] = []
    for m in members:
        if m.mode == "hard":
            cols += [f"{m.name}__{c}" for c in P_COLS]
        elif m.mode == "fraction":
            if variant == "plus_fractions":
                cols += [f"{m.name}__{c}" for c in P_COLS]
        else:
            raise ValueError(f"unknown member mode {m.mode!r}")
    if not cols:
        raise ValueError(f"variant {variant!r} selected no channels for this "
                         "member set")
    X = df[cols].to_numpy(dtype=np.float32)
    return X, cols


def build_head(name: str):
    """Fresh unfitted sklearn head — same config as ``nfi_head_cv.build_head``."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    if name == "logreg":
        return LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128,), max_iter=500, early_stopping=True,
            random_state=SEED)
    raise ValueError(f"unknown head {name!r}")


def train_oof(X_tr: np.ndarray, y_tr: np.ndarray, head: str,
              folds: int = 5) -> np.ndarray:
    """Stratified out-of-fold predictions on the TRAIN split only (reporting).

    Mirrors ``nfi_head_cv.oof_predict``: scaler fit per fold on that fold's
    train side. This never touches the 209 — it is the train-side honesty
    number reported alongside (not instead of) the held-out result.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.full(len(y_tr), -99, dtype=np.int64)
    for tr, te in skf.split(X_tr, y_tr):
        scaler = StandardScaler().fit(X_tr[tr])
        clf = build_head(head)
        clf.fit(scaler.transform(X_tr[tr]), y_tr[tr])
        oof[te] = clf.predict(scaler.transform(X_tr[te]))
    assert (oof != -99).all()
    return oof


def fit_and_eval(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
    head: str,
):
    """Fit scaler+head on train (735), predict held-out (209). Honest hold-out.

    Returns (holdout_pred, holdout_suite, fitted_scaler, fitted_clf).
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)
    clf = build_head(head)
    clf.fit(scaler.transform(X_tr), y_tr)
    pred_te = clf.predict(scaler.transform(X_te))
    return pred_te, accuracy_suite(y_te, pred_te), scaler, clf


def softmax_mean_baseline(
    df: pd.DataFrame, members: list[Member], te_mask: np.ndarray,
    y_te: np.ndarray,
) -> dict:
    """Baseline A: average the forest-class probabilities across the softmax
    members and argmax → a {1..4} forest class (floor to non-forest never fires
    since a probability vector always has a max). Fraction members are excluded
    (R6): their p-channels are not on the same simplex, so averaging them into a
    prob mean is meaningless. This is the ensemble FLOOR — no training.
    """
    hard = [m for m in members if m.mode == "hard"]
    if not hard:
        return {"note": "no hard members — softmax-mean undefined"}
    stack = np.stack(
        [df[[f"{m.name}__{c}" for c in P_COLS]].to_numpy(dtype=np.float64)
         for m in hard], axis=0)          # (n_members, n_plots, 4)
    mean_p = stack.mean(axis=0)           # (n_plots, 4)
    pred_forest = mean_p.argmax(axis=1) + 1  # channels p1..p4 → classes 1..4
    pred_te = pred_forest[te_mask]
    return accuracy_suite(y_te, pred_te)


# ── Statistical honesty (P3 / P7) ──────────────────────────────────────────

def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired exact McNemar test on two per-plot correctness vectors.

    b = A-wrong & B-right, c = A-right & B-wrong. Exact binomial two-sided
    p-value on min(b,c) ~ Binom(b+c, 0.5). ``correct_a`` is the combiner,
    ``correct_b`` is the reference (Trädslag member) — same 209 plots, aligned.
    """
    from scipy.stats import binomtest

    a_right = correct_a.astype(bool)
    b_right = correct_b.astype(bool)
    b = int((~a_right & b_right).sum())   # combiner wrong, ref right
    c = int((a_right & ~b_right).sum())   # combiner right, ref wrong
    n = b + c
    p = 1.0 if n == 0 else binomtest(min(b, c), n, 0.5,
                                     alternative="two-sided").pvalue
    return {"b_ref_only_right": b, "c_combiner_only_right": c,
            "discordant": n, "p_value": round(float(p), 4)}


def bootstrap_ci_delta(
    correct_a: np.ndarray, correct_b: np.ndarray,
    n_resamples: int = 10_000, seed: int = SEED,
) -> dict:
    """Bootstrap 95% CI on the accuracy difference (A − B) over the same plots.

    Paired: each resample draws plot indices with replacement and recomputes
    (mean(correct_a) − mean(correct_b)) so the two accuracies share the sample —
    the honest CI on the *difference*, not two independent CIs.
    """
    rng = np.random.default_rng(seed)
    a = correct_a.astype(np.float64)
    b = correct_b.astype(np.float64)
    n = len(a)
    idx = rng.integers(0, n, size=(n_resamples, n))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_point": round(float(a.mean() - b.mean()), 4),
            "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
            "n_resamples": n_resamples,
            "spans_zero": bool(lo <= 0.0 <= hi)}


def verdict(delta_ci: dict, mcnemar: dict) -> str:
    """G1 verdict language (§6 discipline): a within-noise result is NEVER a
    win. Significant iff the bootstrap CI on the delta excludes 0 AND McNemar
    p < 0.05."""
    sig = (not delta_ci["spans_zero"]) and mcnemar["p_value"] < 0.05
    if sig and delta_ci["delta_point"] > 0:
        return "beats_gate_significant"
    if sig and delta_ci["delta_point"] < 0:
        return "below_gate_significant"
    return "within_noise"


# ── Trädslag reference (the 0.579 gate), recomputed on the same 209 ─────────

def tradslag_reference_correct(
    df: pd.DataFrame, te_mask: np.ndarray, y_te: np.ndarray,
    *, floor: float = 0.05, dominant_frac: float = 0.6,
) -> tuple[np.ndarray, dict]:
    """Reproduce the G1 reference member on the held-out plots.

    The Trädslag member's p-channels are (tall, gran, decid_total, mixedness);
    the published 0.579 gate collapses them with floor=0.05, dom=0.6 (calibrated
    on the 735 train plots — docs/data/tradslag_fraction_finding.md). We
    recompute the collapse here so the McNemar/bootstrap comparison is against
    the EXACT same per-plot predictions the gate was measured on, on the SAME
    209 plots and the SAME truth vector. Returns (per_plot_correct, suite).
    """
    tall = df["tradslag__p1"].to_numpy(dtype=np.float64)
    gran = df["tradslag__p2"].to_numpy(dtype=np.float64)
    decid = df["tradslag__p3"].to_numpy(dtype=np.float64)
    conifer = tall + gran
    total = conifer + decid
    with np.errstate(divide="ignore", invalid="ignore"):
        conif_share = np.where(total > 0, conifer / total, 0.0)
        decid_share = np.where(total > 0, decid / total, 0.0)
    pred = np.zeros(len(df), dtype=np.int64)            # non-forest default
    is_forest = total >= floor
    conif_dom = is_forest & (conif_share >= dominant_frac)
    decid_dom = is_forest & (decid_share >= dominant_frac)
    bland = is_forest & ~conif_dom & ~decid_dom
    pred[conif_dom & (tall >= gran)] = 1
    pred[conif_dom & (tall < gran)] = 2
    pred[decid_dom] = 3
    pred[bland] = 4
    pred_te = pred[te_mask]
    suite = accuracy_suite(y_te, pred_te)
    correct = (pred_te == y_te).astype(np.int64)
    return correct, suite


def export_combiner_npz(out_head: Path, scaler, clf, cols: list[str],
                        member_set: str, variant: str, head: str) -> None:
    """Export the best combiner to a dependency-light npz with a port check.

    Mirrors ``train_distill_head``: standardize → head-forward in pure numpy,
    verified bit-exact against sklearn ``.predict`` on the full 944 design
    matrix before writing. Only the MLP path (128-hidden, single layer) is
    port-exported here — if the best head is logreg we export its coefficients
    with a linear forward and check that too.
    """
    mean_ = scaler.mean_.astype(np.float32)
    scale_ = scaler.scale_.astype(np.float32)
    classes_ = clf.classes_.astype(np.int64)

    payload: dict = {
        "mean_": mean_, "scale_": scale_, "classes_": classes_,
        "feature_cols": np.array(cols),
        "member_set": np.array(member_set), "variant": np.array(variant),
        "head": np.array(head), "seed": np.int64(SEED),
    }
    if head == "mlp":
        W1, W2 = clf.coefs_
        b1, b2 = clf.intercepts_
        payload.update(kind=np.array("mlp"),
                       W1=W1.astype(np.float32), b1=b1.astype(np.float32),
                       W2=W2.astype(np.float32), b2=b2.astype(np.float32))
    else:  # logreg
        payload.update(kind=np.array("logreg"),
                       coef_=clf.coef_.astype(np.float32),
                       intercept_=clf.intercept_.astype(np.float32))
    out_head.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_head, **payload)
    print(f"  wrote combiner → {out_head} "
          f"({out_head.stat().st_size / 1024:.1f} KB)")


def head_forward_npz(X: np.ndarray, npz) -> np.ndarray:
    """Pure-numpy forward reproducing the exported head's hard prediction."""
    Xs = (X - npz["mean_"]) / npz["scale_"]
    kind = str(npz["kind"])
    if kind == "mlp":
        h = np.maximum(Xs @ npz["W1"] + npz["b1"], 0.0)
        logits = h @ npz["W2"] + npz["b2"]
    else:
        logits = Xs @ npz["coef_"].T + npz["intercept_"]
        if logits.shape[1] == 1:            # binary logreg edge case
            logits = np.hstack([-logits, logits])
    return npz["classes_"][logits.argmax(axis=1)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", default="data/nfi",
                    help="dir with <member>_per_plot.parquet dumps")
    ap.add_argument("--split", default="data/distill/distill_split.json")
    ap.add_argument("--out-stack", default="data/distill/ensemble_stack.parquet")
    ap.add_argument("--out-head", default="data/distill/ensemble_combiner.npz")
    ap.add_argument("--out-json", default="data/distill/ensemble_results.json")
    ap.add_argument("--heads", default="logreg,mlp")
    ap.add_argument("--bootstrap", type=int, default=10_000)
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]

    # ── Join every available member on the 6-tuple key ──────────────────────
    stack_df, present = load_members(dump_dir, ALL_MEMBERS)
    print(f"joined {len(present)} members → {len(stack_df)} plots "
          f"({[m.name for m in present]})")

    # Persist the stack matrix for P8 disagreement analysis (§Regler).
    out_stack = Path(args.out_stack)
    out_stack.parent.mkdir(parents=True, exist_ok=True)
    stack_df.to_parquet(out_stack, index=False)
    print(f"wrote stack matrix → {out_stack}")

    # ── Split membership from distill_split.json (grouped-by-tile) ──────────
    split = json.loads(Path(args.split).read_text())
    test_tiles = {str(t) for t in split["test_tiles"]}
    tn = stack_df["tile_name"].astype(str)
    te_mask = tn.isin(test_tiles).to_numpy()
    tr_mask = ~te_mask
    if int(te_mask.sum()) != split["n_test_plots"]:
        raise SystemExit(
            f"held-out size {int(te_mask.sum())} != split n_test_plots "
            f"{split['n_test_plots']}")
    y = suite_space_truth(stack_df["nfi_forest"].to_numpy())
    y_tr, y_te = y[tr_mask], y[te_mask]
    print(f"split: {int(tr_mask.sum())} train / {int(te_mask.sum())} held-out")

    present_names = {m.name for m in present}
    member_sets = {k: [n for n in v if n in present_names]
                   for k, v in MEMBER_SETS.items()}

    # ── Trädslag G1 reference on the same 209 ───────────────────────────────
    trad_correct = None
    trad_suite = None
    if "tradslag" in present_names:
        trad_correct, trad_suite = tradslag_reference_correct(
            stack_df, te_mask, y_te)
        print(f"\nG1 reference (Trädslag frac, floor=0.05/dom=0.6) held-out "
              f"OA = {trad_suite['overall_accuracy_5class']:.4f} "
              f"(gate {G1_GATE}, kappa {trad_suite['cohen_kappa']:.4f})")

    results: dict = {
        "_meta": {
            "join_keys": list(JOIN_KEYS), "n_plots": len(stack_df),
            "n_train": int(tr_mask.sum()), "n_holdout": int(te_mask.sum()),
            "members_present": [m.name for m in present],
            "member_sets": member_sets, "feature_variants": list(FEATURE_VARIANTS),
            "g1_gate": G1_GATE, "seed": SEED,
        },
        "g1_reference_tradslag": trad_suite,
        # The single headline result: the OOF-locked config on the 209, filled
        # after the sweep. `reported` is the ONLY block that may be headlined.
        "reported": None,
        # The full 8-config sweep, kept for transparency. Its held-out column is
        # selection-biased (max-of-8) — DO NOT headline any number in here.
        "exploratory": {
            "note": ("selection-biased (max-of-8 on held-out) — kept for "
                     "transparency; never headline. Config choice is on "
                     "train-OOF only; see `reported` for the honest number."),
            "runs": [],
        },
    }

    # ── Sweep: 2 sets × 2 variants × 2 heads, SELECTED on train-OOF only ────
    # best_by_head keeps the OOF-argmax config per head. The held-out suite is
    # computed per config for the exploratory table but is NEVER a selection
    # key — selection is oof_oa. The 209 are thus never used to CHOOSE anything;
    # the sweep's held-out column is transparency-only (selection-biased).
    best_by_head: dict[str, BestRun] = {}
    print(f"\n{'set':>6} {'variant':>15} {'baselineA':>10} {'head':>7} "
          f"{'trainOOF':>9} {'holdout':>8} {'ΔG1':>7} {'McN-p':>7} "
          f"{'CIΔ_lo':>7} {'CIΔ_hi':>7}  verdict")
    print("-" * 108)

    for set_name, names in member_sets.items():
        mem = [m for m in present if m.name in names]
        if not mem:
            continue
        for variant in FEATURE_VARIANTS:
            try:
                X, cols = build_design_matrix(stack_df, mem, variant)
            except ValueError:
                continue  # variant selects no channels for this set
            X_tr, X_te = X[tr_mask], X[te_mask]

            baseA = softmax_mean_baseline(stack_df, mem, te_mask, y_te)
            baseA_oa = baseA.get("overall_accuracy_5class")

            for head in heads:
                oof = train_oof(X_tr, y_tr, head)
                oof_oa = accuracy_suite(y_tr, oof)["overall_accuracy_5class"]
                oof_correct = (oof == y_tr).astype(np.int64)
                pred_te, suite_te, scaler, clf = fit_and_eval(
                    X_tr, y_tr, X_te, y_te, head)
                ho_oa = suite_te["overall_accuracy_5class"]

                # Exploratory held-out comparison (selection-biased; the JSON
                # flags it so, and NEITHER config selection NOR R3 reads it).
                comb_correct = (pred_te == y_te).astype(np.int64)
                mcn = ci = None
                vdt = "no_reference"
                if trad_correct is not None:
                    mcn = mcnemar_exact(comb_correct, trad_correct)
                    ci = bootstrap_ci_delta(comb_correct, trad_correct,
                                            n_resamples=args.bootstrap)
                    vdt = verdict(ci, mcn)

                run = {
                    "member_set": set_name, "variant": variant, "head": head,
                    "n_features": X.shape[1], "feature_cols": cols,
                    "baseline_A_softmax_mean": baseA,
                    "train_oof_accuracy_5class": oof_oa,
                    "holdout_suite": suite_te,
                    "mcnemar_vs_tradslag": mcn,
                    "bootstrap_delta_vs_tradslag": ci,
                    "g1_verdict": vdt,
                }
                results["exploratory"]["runs"].append(run)

                dg1 = (ho_oa - G1_GATE)
                print(f"{set_name:>6} {variant:>15} "
                      f"{(baseA_oa if baseA_oa is not None else float('nan')):>10.4f} "
                      f"{head:>7} {oof_oa:>9.4f} {ho_oa:>8.4f} {dg1:>+7.4f} "
                      f"{(mcn['p_value'] if mcn else float('nan')):>7} "
                      f"{(ci['ci95_low'] if ci else float('nan')):>7} "
                      f"{(ci['ci95_high'] if ci else float('nan')):>7}  {vdt}")

                cur = best_by_head.get(head)
                if cur is None or oof_oa > cur.oof_oa:
                    best_by_head[head] = BestRun(
                        oof_oa=oof_oa, ho_oa=ho_oa, run=run, scaler=scaler,
                        clf=clf, cols=cols, pred_te=pred_te,
                        oof_correct=oof_correct)

    print("\n(sweep held-out column above is exploratory / selection-biased — "
          "config choice is on train-OOF only; do not headline those numbers)")

    # ── R3: logreg vs MLP, decided on TRAIN-OOF correctness (735), not 209 ───
    # Default to logreg (guards MLP overfit on 735). Promote MLP only if it
    # beats logreg outside the paired bootstrap CI AND McNemar<0.05 on the OOF
    # correctness vectors — the held-out set plays NO part in this choice.
    best: BestRun | None = None
    lg = best_by_head.get("logreg")
    ml = best_by_head.get("mlp")
    if lg is not None and ml is not None:
        mcn_hm = mcnemar_exact(ml.oof_correct, lg.oof_correct)
        ci_hm = bootstrap_ci_delta(ml.oof_correct, lg.oof_correct,
                                   n_resamples=args.bootstrap)
        mlp_beats = (not ci_hm["spans_zero"]) and mcn_hm["p_value"] < 0.05
        r3 = {"selected_on": "train_oof_correctness_735",
              "best_logreg_oof": lg.oof_oa, "best_mlp_oof": ml.oof_oa,
              "mlp_vs_logreg_mcnemar": mcn_hm,
              "mlp_vs_logreg_bootstrap": ci_hm,
              "mlp_beats_logreg_out_of_ci": mlp_beats,
              "reported_head": "mlp" if mlp_beats else "logreg"}
        results["_meta"]["r3_head_selection"] = r3
        best = ml if mlp_beats else lg
        print(f"\nR3 (on train-OOF): best logreg OOF={lg.oof_oa:.4f}  "
              f"best mlp OOF={ml.oof_oa:.4f}  "
              f"MLP−logreg CI=[{ci_hm['ci95_low']}, {ci_hm['ci95_high']}] "
              f"McN-p={mcn_hm['p_value']} → report "
              f"{'MLP' if mlp_beats else 'LOGREG (MLP not robustly better)'}")
    else:
        best = lg or ml

    # ── Locked config → the SINGLE honest held-out evaluation on the 209 ────
    if best is not None:
        run, scaler, clf, cols = best.run, best.scaler, best.clf, best.cols
        suite_te = run["holdout_suite"]
        ho_oa = suite_te["overall_accuracy_5class"]
        print(f"\nlocked config (OOF-selected): set={run['member_set']} "
              f"variant={run['variant']} head={run['head']} "
              f"(OOF={best.oof_oa:.4f})")

        # This is the one number that headlines: the locked config on the 209,
        # with its McNemar/bootstrap vs the 0.579 Trädslag gate.
        headline = {
            "member_set": run["member_set"], "variant": run["variant"],
            "head": run["head"], "n_features": run["n_features"],
            "train_oof_accuracy_5class": best.oof_oa,
            "holdout_suite": suite_te,
        }
        if trad_correct is not None:
            comb_correct = (best.pred_te == y_te).astype(np.int64)
            mcn = mcnemar_exact(comb_correct, trad_correct)
            ci = bootstrap_ci_delta(comb_correct, trad_correct,
                                    n_resamples=args.bootstrap)
            vdt = verdict(ci, mcn)
            headline.update(mcnemar_vs_tradslag=mcn,
                            bootstrap_delta_vs_tradslag=ci, g1_verdict=vdt)
            print(f"  held-out OA = {ho_oa:.4f} (ΔG1 {ho_oa - G1_GATE:+.4f}), "
                  f"McN-p={mcn['p_value']} "
                  f"CIΔ=[{ci['ci95_low']}, {ci['ci95_high']}] → {vdt}")
        results["reported"] = headline

        out_head = Path(args.out_head)
        export_combiner_npz(out_head, scaler, clf, cols,
                            run["member_set"], run["variant"], run["head"])
        # Port check on the full design matrix for the best run's channels.
        X_best = stack_df[cols].to_numpy(dtype=np.float32)
        npz = np.load(out_head, allow_pickle=True)
        port = head_forward_npz(X_best, npz)
        sk = clf.predict(scaler.transform(X_best))
        mism = int((port != sk).sum())
        if mism:
            raise SystemExit(f"npz port disagrees with sklearn on {mism}/"
                             f"{len(sk)} plots")
        print(f"  port check: npz forward == sklearn .predict on all "
              f"{len(sk)} plots ✓")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote results → {out_json}")


if __name__ == "__main__":
    main()
