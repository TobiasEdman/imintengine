#!/usr/bin/env python3
"""Development-only frozen-feature diagnostic for NMD dryland forest types.

Two pre-registered class schemes: ``binary`` (NMD 113 versus 115–117, the
first IM-023 question) and ``forest5`` (111, 112, 113, 114 and 115–117).

This never trains a serving model. Raw NMD labels measure agreement with NMD,
not independent field accuracy. Historical training locations may have been
seen by the frozen encoder/head; spatial CV applies only to the NEW probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ROLE = "historical_training_development_diagnostic"
# Ordered (name, raw NMD2023 codes) per class; the index is the probe label.
CLASS_SCHEMES = {
    "binary": (("barrblandskog", (113,)), ("lövskog (115–117)", (115, 116, 117))),
    "forest5": (("tallskog", (111,)), ("granskog", (112,)), ("barrblandskog", (113,)),
                ("lövblandad barrskog", (114,)), ("lövskog (115–117)", (115, 116, 117))),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def scheme_classes(scheme: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if scheme not in CLASS_SCHEMES:
        raise ValueError(f"unknown class scheme {scheme!r}; expected one of {sorted(CLASS_SCHEMES)}")
    return CLASS_SCHEMES[scheme]


def class_labels(raw: np.ndarray, scheme: str) -> np.ndarray:
    classes = scheme_classes(scheme)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("raw NMD must be a one-dimensional integer array")
    labels = np.full(raw.shape, -1, dtype=np.int64)
    for index, (_, codes) in enumerate(classes):
        labels[np.isin(raw, codes)] = index
    if (labels < 0).any():
        raise ValueError(f"raw NMD codes outside the {scheme} scheme")
    return labels


def sample_pixels(raw: np.ndarray, valid: np.ndarray, *, per_class: int,
                  seed: int, scheme: str, stride: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Sample interiors on a fixed 100 m lattice, capped equally per class.

    Three-by-three binary erosion avoids immediate class boundaries. This
    deliberately excludes difficult edge pixels; the report states that bias.
    """
    from scipy.ndimage import binary_erosion

    if (raw.shape != valid.shape or raw.ndim != 2 or per_class < 1 or stride < 1
            or not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError("invalid sampling arrays or cap")
    lattice = np.zeros(raw.shape, dtype=bool)
    lattice[::stride, ::stride] = True
    rng = np.random.default_rng(seed)
    selected = []
    for _, codes in scheme_classes(scheme):
        eligible = binary_erosion(np.isin(raw, codes), structure=np.ones((3, 3)))
        rc = np.argwhere(eligible & valid & lattice)
        if len(rc) < 8:
            raise ValueError("fewer than eight eligible pixels in a class")
        selected.append(rc[rng.choice(len(rc), min(per_class, len(rc)), replace=False)])
    rc = np.concatenate(selected)
    return rc[:, 0], rc[:, 1]


def spatial_folds(xy: np.ndarray, *, n_splits: int, block_m: float,
                  buffer_m: float) -> list[tuple[np.ndarray, np.ndarray]]:
    from scipy.spatial import cKDTree
    from sklearn.model_selection import GroupKFold

    if (xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all()
            or block_m <= 0 or buffer_m < 0 or n_splits < 2):
        raise ValueError("invalid projected coordinates or spatial split settings")
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError("duplicate spatial samples would overweight locations")
    blocks = np.floor(xy / block_m).astype(np.int64)
    _, groups = np.unique(blocks, axis=0, return_inverse=True)
    if len(np.unique(groups)) < n_splits:
        raise ValueError("not enough independent spatial blocks")
    folds = []
    for train, test in GroupKFold(n_splits=n_splits).split(xy, groups=groups):
        nearest = cKDTree(xy[test]).query(xy[train], k=1)[0]
        train = train[nearest >= buffer_m]
        if not len(train):
            raise ValueError("spatial buffer removed the entire training fold")
        folds.append((train, test))
    return folds


def metrics(y: np.ndarray, pred: np.ndarray, scheme: str) -> dict:
    from sklearn.metrics import confusion_matrix

    names = [name for name, _ in scheme_classes(scheme)]
    cm = confusion_matrix(y, pred, labels=list(range(len(names))))
    per_class = {}
    for k, name in enumerate(names):
        tp = int(cm[k, k])
        n_true, n_pred = int(cm[k].sum()), int(cm[:, k].sum())
        union = n_true + n_pred - tp
        per_class[name] = {"support": n_true,
                           "precision": tp / n_pred if n_pred else None,
                           "recall": tp / n_true if n_true else None,
                           "iou": tp / union if union else None}
    recalls = [v["recall"] for v in per_class.values()]
    return {"confusion_matrix": cm.tolist(), "per_class": per_class,
            "accuracy": float(np.mean(y == pred)),
            "balanced_accuracy": float(np.mean(recalls)) if None not in recalls else None}


def bootstrap_blocks(y: np.ndarray, pred: np.ndarray, xy: np.ndarray, *,
                     block_m: float, n_classes: int, draws: int = 500) -> list[float]:
    """Resample whole spatial blocks; never pretend correlated pixels are iid."""
    blocks = np.floor(xy / block_m).astype(np.int64)
    _, groups = np.unique(blocks, axis=0, return_inverse=True)
    indices = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(42)
    scores = []
    for _ in range(draws):
        take = np.concatenate([indices[i] for i in rng.integers(len(indices), size=len(indices))])
        if len(np.unique(y[take])) == n_classes:
            scores.append(np.mean([(pred[take][y[take] == k] == k).mean() for k in range(n_classes)]))
    if not scores:
        raise ValueError("bootstrap contains no resample with every class")
    return np.quantile(scores, [0.025, 0.975]).tolist()


def assigned_classes(y: np.ndarray, assigned: np.ndarray, scheme: str) -> dict:
    """Count, per raw NMD class, the unified class another labeller gave each pixel."""
    report = {}
    for k, (name, _) in enumerate(scheme_classes(scheme)):
        values, counts = np.unique(assigned[y == k], return_counts=True)
        report[name] = {str(int(v)): int(n) for v, n in zip(values, counts)}
    return report


def run_probe(bundle: Path, *, scheme: str, n_splits: int = 5, block_m: float = 50000,
              buffer_m: float = 5000) -> dict:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classes = scheme_classes(scheme)
    n_classes = len(classes)
    meta = json.loads(bundle.with_suffix(".json").read_text())
    if meta.get("role") != ROLE:
        raise ValueError("only explicitly qualified development diagnostic inputs are allowed")
    if meta.get("feature_year") not in (2022, 2023) or meta.get("reference_year") != 2023:
        raise ValueError("explicit feature and reference years are required")
    if meta.get("scheme") != scheme:
        raise ValueError(f"extraction receipt scheme {meta.get('scheme')!r} is not {scheme!r}")
    if meta.get("bundle_sha256") != digest(bundle):
        raise ValueError("feature bundle differs from its extraction receipt")
    with np.load(bundle, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    y, xy = class_labels(data["raw_nmd"], scheme), data["xy"]
    if len(y) != len(xy) or not len(y):
        raise ValueError("empty or inconsistent samples")
    folds = spatial_folds(xy, n_splits=n_splits, block_m=block_m, buffer_m=buffer_m)
    for train, test in folds:
        if len(np.unique(y[train])) != n_classes or len(np.unique(y[test])) != n_classes:
            raise ValueError("every class must be present in every train and test fold")
    reference = " versus ".join(f"{name} ({', '.join(map(str, codes))})" for name, codes in classes)
    report = {"role": ROLE, "scheme": scheme,
              "reference": f"NMD2023 raw {reference}; dryland only",
              "feature_year": meta["feature_year"], "reference_year": meta["reference_year"],
              "reference_year_gap": meta["reference_year"] - meta["feature_year"],
              "limitations": ["Historical training locations; not an unseen-area model evaluation.",
                              "NMD agreement, not independent field accuracy.",
                              "Fixed NMD2023 reference may postdate features; not feature-year field status.",
                              "Interior pixels on a 100 m lattice; edges excluded.",
                              "Tiles qualify only where every class has eight interior lattice "
                              "pixels, which excludes areas where a class is rare.",
                              "Small diagnostic cannot justify merging classes on its own."],
              "protocol": {"scheme": scheme, "folds": n_splits, "block_m": block_m, "buffer_m": buffer_m,
                           "head": "StandardScaler + LogisticRegression(C=1, balanced, max_iter=2000)",
                           "tuning": "none", "bootstrap": "500 whole-block resamples, seed42"},
              "n_samples": len(y), "n_tiles": len(np.unique(data["tile"])),
              "feature_receipt_sha256": digest(bundle.with_suffix(".json")), "models": {},
              # Where each raw NMD class actually went in the training targets and R4 output.
              "assigned_unified_class": {key: assigned_classes(y, data[key], scheme)
                                         for key in ("base_label", "distilled_label", "r4_prediction")}}
    majority = np.full(len(y), -1, dtype=np.int64)
    for train, test in folds:
        majority[test] = np.argmax(np.bincount(y[train], minlength=n_classes))
    report["majority_baseline"] = metrics(y, majority, scheme)
    for name in ("tessera", "r4"):
        X = data[name]
        if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all():
            raise ValueError("invalid frozen features")
        predicted = np.full(len(y), -1, dtype=np.int64)
        fold_reports = []
        for train, test in folds:
            head = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000, random_state=42))
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                head.fit(X[train], y[train])
            predicted[test] = head.predict(X[test])
            fold_reports.append({"n_train": len(train), "n_test": len(test),
                                 **metrics(y[test], predicted[test], scheme)})
        if np.any(predicted < 0):
            raise AssertionError("not every sample has an out-of-fold prediction")
        report["models"][name] = {"features": X.shape[1], **metrics(y, predicted, scheme),
            "balanced_accuracy_95pct_block_bootstrap": bootstrap_blocks(
                y, predicted, xy, block_m=block_m, n_classes=n_classes), "folds": fold_reports}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--scheme", required=True, choices=sorted(CLASS_SCHEMES))
    ap.add_argument("--folds", default=5, type=int)
    ap.add_argument("--block-m", default=50000, type=float)
    ap.add_argument("--buffer-m", default=5000, type=float)
    args = ap.parse_args()
    write_json(args.output, run_probe(args.features, scheme=args.scheme, n_splits=args.folds,
                                     block_m=args.block_m, buffer_m=args.buffer_m))


if __name__ == "__main__":
    main()
