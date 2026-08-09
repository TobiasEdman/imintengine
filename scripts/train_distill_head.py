#!/usr/bin/env python3
"""scripts/train_distill_head.py — train the FINAL distillation head + tile split.

Hybrid-NFI distillation step 1 (local, no GPU). The CV experiment
(``nfi_head_cv.py``) showed a small MLP on the 256-dim pre-classifier features
beats NMD2023 for forest-type accuracy (OOF 0.637 vs 0.493). This script trains
the *deployable* head on a GROUPED-BY-TILE train split only, so the held-out
tiles' plots stay an honest final eval for the distilled dense model (the parent
runs that eval). It then exports the head to a dependency-light npz that
``distill_forest_labels.py`` forwards densely on the cluster — no sklearn needed
at inference, just numpy matmuls.

Why group by TILE, not plot? Plots on the same 512×512 tile share the same
feature grid and spatial context; a plot-level split would leak tile context
across train/test. GroupShuffleSplit on ``tile_name`` guarantees every test
plot's tile is entirely unseen at head-train time — the same isolation the
distilled dense pseudo-labels rely on (the head never saw test tiles' plots, so
pseudo-labelling them densely is leakage-free).

Truth space is the 5-class suite {0 non-forest, 1 tall, 2 gran, 3 löv, 4 bland};
``nfi_forest == -1`` (treeless) collapses to 0 — identical to ``nfi_head_cv.py``.

    python scripts/train_distill_head.py \
        --features data/nfi/nfi_plot_features_nmd2023.parquet \
        --test-frac 0.2 --seed 42 \
        --out-head data/distill/distill_head.npz \
        --out-split data/distill/distill_split.json
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


def grouped_split(tiles: np.ndarray, y: np.ndarray, test_frac: float, seed: int):
    """Group-by-tile train/test split → (train_idx, test_idx) plot masks.

    GroupShuffleSplit keeps every tile wholly on one side, so no tile's context
    straddles the split. We retry a few seeds and keep the split whose test side
    best covers all 5 classes (GroupShuffleSplit is not class-aware; with 261
    tiles a single draw can drop a rare class from the test side, which would
    make the preview accuracy uninformative). This is a *preview* accuracy only
    — the real eval is the distilled dense model — so we optimise coverage, not
    balance, and never touch the seed used for the export/reproducibility path.
    """
    from sklearn.model_selection import GroupShuffleSplit

    n_classes = len(np.unique(y))
    best = None
    for trial in range(20):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac,
                                random_state=seed + trial)
        tr, te = next(gss.split(np.zeros(len(y)), y, groups=tiles))
        # Coverage score: how many of the 5 classes appear on the test side.
        cover = len(np.unique(y[te]))
        if best is None or cover > best[0]:
            best = (cover, tr, te, trial)
        if cover == n_classes and trial == 0:
            break  # the requested seed already covers all classes — use it
    cover, tr, te, trial = best
    if trial != 0:
        print(f"  note: base seed dropped a class from the test side; "
              f"used seed+{trial} for a {cover}/{n_classes}-class test split "
              f"(export/reproducibility unaffected — the split JSON records the "
              f"resulting tiles)")
    return tr, te


def head_forward_npz(X: np.ndarray, mean_, scale_, W1, b1, W2, b2,
                     classes_) -> np.ndarray:
    """Dependency-light forward reproducing sklearn MLPClassifier.predict.

    Standardize → relu(X@W1+b1) → (@W2+b2) → argmax → map through classes_.
    Softmax is monotonic so argmax of logits == argmax of probabilities; the
    output layer's softmax is irrelevant for the hard prediction. ``classes_``
    maps the argmax index back to the true label — MLPClassifier orders its
    outputs by ``np.unique(y)``, which is ascending, but we never assume that;
    we always index through the exported ``classes_``.
    """
    Xs = (X - mean_) / scale_
    h = np.maximum(Xs @ W1 + b1, 0.0)          # relu hidden
    logits = h @ W2 + b2                         # output pre-activation
    return classes_[logits.argmax(axis=1)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True,
                    help="parquet from extract_plot_features.py")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-head", required=True, help="head export npz path")
    ap.add_argument("--out-split", required=True, help="split JSON path")
    args = ap.parse_args()

    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet(args.features)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"features parquet missing {len(missing)} feature cols "
                         f"(first: {missing[:3]})")

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["nfi_forest"].to_numpy(dtype=np.int64)
    y = np.where(y == -1, 0, y)                 # treeless → non-forest
    tiles = df["tile_name"].to_numpy()

    n = len(y)
    class_counts = {int(c): int((y == c).sum()) for c in sorted(np.unique(y))}
    print(f"{n} plots on {df['tile_name'].nunique()} tiles, {X.shape[1]} features")
    print(f"  class support (all): {class_counts}")

    # ── Grouped split by tile ──────────────────────────────────────────────
    tr, te = grouped_split(tiles, y, args.test_frac, args.seed)
    train_tiles = sorted(set(tiles[tr].tolist()))
    test_tiles = sorted(set(tiles[te].tolist()))
    assert not (set(train_tiles) & set(test_tiles)), "tile leaked across split"

    def _support(mask):
        return {int(c): int((y[mask] == c).sum()) for c in sorted(np.unique(y))}

    print(f"\n  split (test_frac={args.test_frac}, seed={args.seed}):")
    print(f"    train: {len(tr):>3} plots on {len(train_tiles):>3} tiles  "
          f"support {_support(tr)}")
    print(f"    test:  {len(te):>3} plots on {len(test_tiles):>3} tiles  "
          f"support {_support(te)}")

    # ── Train scaler + MLP on the TRAIN split only ─────────────────────────
    scaler = StandardScaler().fit(X[tr])
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                        early_stopping=True, random_state=args.seed)
    clf.fit(scaler.transform(X[tr]), y[tr])

    # ── Preview accuracy on held-out test plots (NOT the final eval) ───────
    pred_te = clf.predict(scaler.transform(X[te]))
    suite = accuracy_suite(y[te], pred_te)
    overall = suite["overall_accuracy_5class"]
    print(f"\n  TEST-SPLIT PREVIEW (head only, not the distilled dense model):")
    print(f"    overall 5-class accuracy = {overall:.4f}  "
          f"(kappa {suite['cohen_kappa']:.4f})")
    print(f"    baselines: NMD2023={BASELINES['NMD2023']}  "
          f"v8b={BASELINES['v8b']}  → ΔNMD2023 {overall - BASELINES['NMD2023']:+.3f}")

    # ── Export dependency-light head ───────────────────────────────────────
    # MLPClassifier((128,)) → coefs_ = [W1 (256,128), W2 (128,5)],
    #                         intercepts_ = [b1 (128,), b2 (5,)].
    W1, W2 = clf.coefs_
    b1, b2 = clf.intercepts_
    classes_ = clf.classes_.astype(np.int64)
    mean_ = scaler.mean_.astype(np.float32)
    scale_ = scaler.scale_.astype(np.float32)

    # VERIFY the numpy port reproduces sklearn .predict EXACTLY on ALL plots.
    # (Guards the port against any activation/ordering drift; both train and
    # test plots are checked — the forward is split-agnostic.)
    port_all = head_forward_npz(
        X, mean_, scale_, W1.astype(np.float32), b1.astype(np.float32),
        W2.astype(np.float32), b2.astype(np.float32), classes_,
    )
    sk_all = clf.predict(scaler.transform(X))
    assert np.array_equal(port_all, sk_all), (
        f"npz port disagrees with sklearn on "
        f"{int((port_all != sk_all).sum())}/{n} plots"
    )
    print(f"\n  port check: npz forward == sklearn .predict on all {n} plots ✓")

    out_head = Path(args.out_head)
    out_head.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_head,
        mean_=mean_, scale_=scale_,
        W1=W1.astype(np.float32), b1=b1.astype(np.float32),
        W2=W2.astype(np.float32), b2=b2.astype(np.float32),
        classes_=classes_,
        # provenance
        seed=np.int64(args.seed), n_train_plots=np.int64(len(tr)),
        hidden=np.int64(128), n_features=np.int64(N_FEATURES),
    )
    print(f"  wrote head → {out_head} "
          f"({out_head.stat().st_size / 1024:.1f} KB)")

    out_split = Path(args.out_split)
    out_split.parent.mkdir(parents=True, exist_ok=True)
    out_split.write_text(json.dumps({
        "seed": args.seed,
        "test_frac": args.test_frac,
        "features": str(args.features),
        "train_tiles": train_tiles,
        "test_tiles": test_tiles,
        "n_train_plots": int(len(tr)),
        "n_test_plots": int(len(te)),
        "train_support": _support(tr),
        "test_support": _support(te),
        "test_preview_accuracy_5class": overall,
        "baselines": BASELINES,
    }, indent=2, ensure_ascii=False))
    print(f"  wrote split → {out_split}")


if __name__ == "__main__":
    main()
