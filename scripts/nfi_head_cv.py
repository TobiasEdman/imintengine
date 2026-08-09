#!/usr/bin/env python3
"""scripts/nfi_head_cv.py — k-fold CV a small head on NFI-plot features.

Hybrid-experiment step 2. Take the per-plot 256-dim pre-classifier features
(from ``extract_plot_features.py``) and the NFI field-truth forest class, and
k-fold-cross-validate a small head on them. Every plot gets an OUT-OF-FOLD
prediction (trained on the other folds), so the reported accuracy is leakage-
free. The question: can a head trained DIRECTLY on field-truth beat the NMD
label ceiling for forest-type classification?

Truth space is the 5-class accuracy-suite space {1 tall, 2 gran, 3 löv, 4 bland,
0 non-forest} — ``nfi_forest == -1`` (treeless) collapses to 0. We standardize
the 256 features (fit on train folds only, applied to the held-out fold), train
each head on the train folds, and predict the held-out fold. ``accuracy_suite``
(reused from ``validate_against_nfi``) scores the pooled OOF predictions the same
way the seg-model validation does, so the numbers are directly comparable to the
fixed baselines NMD2023=0.493 and v8b=0.465.

    python scripts/nfi_head_cv.py \
        --features /data/nfi_eval/nfi_plot_features_nmd2023.parquet \
        --folds 5 --heads logreg,mlp \
        --out /data/nfi_eval/nfi_head_cv.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_against_nfi import accuracy_suite

SEED = 42
N_FEATURES = 256
FEATURE_COLS = [f"f{i:03d}" for i in range(N_FEATURES)]

# Fixed reference baselines (seg-model overall_accuracy_5class at these plots).
BASELINES = {"NMD2023": 0.493, "v8b": 0.465}


def build_head(name: str):
    """Return a fresh, unfitted sklearn classifier for the head ``name``."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    if name == "logreg":
        return LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128,), max_iter=500, early_stopping=True,
            random_state=SEED,
        )
    raise ValueError(f"unknown head {name!r} (known: logreg, mlp)")


def oof_predict(X: np.ndarray, y: np.ndarray, head_name: str, folds: int) -> np.ndarray:
    """StratifiedKFold out-of-fold predictions — one prediction per plot.

    Standardization is fit on the TRAIN folds only and applied to the held-out
    fold, so no held-out statistic leaks into training. Each plot is predicted
    exactly once, by a model that never saw it.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.full(len(y), -99, dtype=np.int64)

    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        clf = build_head(head_name)
        clf.fit(scaler.transform(X[tr]), y[tr])
        oof[te] = clf.predict(scaler.transform(X[te]))

    assert (oof != -99).all(), "some plot got no out-of-fold prediction"
    return oof


def _fmt_delta(v: float, base: float) -> str:
    d = v - base
    return f"{d:+.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="parquet from extract_plot_features.py")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--heads", default="logreg,mlp",
                    help="comma-separated: logreg,mlp")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"features parquet missing {len(missing)} feature cols "
                         f"(first: {missing[:3]})")

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    # Truth → 5-class suite space: -1 (treeless) → 0 (non-forest).
    y = df["nfi_forest"].to_numpy(dtype=np.int64)
    y = np.where(y == -1, 0, y)

    n = len(y)
    class_counts = {int(c): int((y == c).sum()) for c in sorted(np.unique(y))}
    print(f"{n} plots, {X.shape[1]} features; class support {class_counts}")

    # StratifiedKFold needs every class to have >= folds members.
    min_support = min(class_counts.values())
    if min_support < args.folds:
        raise SystemExit(
            f"smallest class has {min_support} plots < folds={args.folds}; "
            f"reduce --folds or merge classes"
        )

    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    results: dict = {
        "_meta": {
            "features": args.features, "n_plots": n, "folds": args.folds,
            "seed": SEED, "class_support": class_counts, "baselines": BASELINES,
        },
        "heads": {},
    }

    print(f"\n{'head':<10} {'overall':>8} {'kappa':>8}  "
          f"{'ΔNMD2023':>9} {'Δv8b':>8}")
    print("-" * 48)
    for head_name in heads:
        oof = oof_predict(X, y, head_name, args.folds)
        suite = accuracy_suite(y, oof)
        overall = suite["overall_accuracy_5class"]
        kappa = suite["cohen_kappa"]
        results["heads"][head_name] = suite
        print(f"{head_name:<10} {overall:>8.4f} {kappa:>8.4f}  "
              f"{_fmt_delta(overall, BASELINES['NMD2023']):>9} "
              f"{_fmt_delta(overall, BASELINES['v8b']):>8}")

    print("\nbaselines:  NMD2023=0.493  v8b=0.465  (seg-model overall @ same plots)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
