"""Phase 1 — in-distribution sanity check.

Loads the holdout 10% test split, runs the ensemble + each baseline,
computes the metric palette from :mod:`imint.eval.metrics`. Produces a
single ``EvalResult`` per (model, split) and a comparison table.

Model loading is delegated to a ``model_factory`` callable so this
module doesn't hard-import the project's segmentation architecture
(which is still moving). Callers pass a function that takes the
checkpoint path and returns a ready-to-eval ``nn.Module`` on the
target device. A reference random-prediction factory is provided
for dry-running the pipeline before the ensemble is trained.

Pass criteria from the plan document live in the report generator;
this module just produces the numbers.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .metrics import EvalResult, expected_calibration_error, per_class_iou


# ── Random-prediction reference factory (for dry-runs) ──────────────────────


def random_prediction_factory(num_classes: int = 23, seed: int = 0):
    """Return a ``predict_tile(npz_path) -> (label_pred, probs)`` callable.

    The ``probs`` array uses Dirichlet-ish noise so the ECE measurement
    has something realistic to bin. Use this for pipeline shake-out
    before a real checkpoint exists; the IoU numbers will be
    near-chance.
    """
    rng = np.random.default_rng(seed)

    def _predict(npz_path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
        with np.load(npz_path, allow_pickle=True) as data:
            label = data["label"]
        h, w = label.shape
        logits = rng.normal(size=(num_classes, h, w)).astype(np.float32)
        # softmax along class axis
        e = np.exp(logits - logits.max(axis=0, keepdims=True))
        probs = e / e.sum(axis=0, keepdims=True)
        pred = probs.argmax(axis=0).astype(np.int64)
        return pred, probs

    return _predict


# ── Main entry point ────────────────────────────────────────────────────────


def run(
    ensemble_checkpoint: Path | None,
    tiles_dir: str,
    split_dir: Path,
    *,
    model_factory: Callable[[Path], Callable] | None = None,
    baselines: list[str] | None = None,
    output_dir: Path | None = None,
    num_classes: int = 23,
    ignore_index: int = 0,
    ece_pixel_sample: int = 200_000,
    device: str = "cuda",
) -> dict[str, EvalResult]:
    """Run the in-distribution evaluation.

    Args:
        ensemble_checkpoint: Path to the trained ensemble. Ignored if
            ``model_factory`` is ``None`` — in that case the
            reference random-prediction factory is used and the
            results are dry-run quality.
        tiles_dir: Where the test tiles live.
        split_dir: Folder with ``test.txt`` + ``manifest.json``.
        model_factory: ``factory(ckpt_path) -> predict_fn`` where
            ``predict_fn(npz_path) -> (pred, probs)``. Falls back to
            :func:`random_prediction_factory` for shake-out.
        baselines: Subset of registry keys from
            :data:`imint.eval.baselines.BASELINES`. ``None`` skips
            baselines entirely.
        output_dir: Predictions + per-tile metrics dumped here.
        num_classes: 23 for the unified schema.
        ignore_index: 0 (background) by convention.
        ece_pixel_sample: ECE is computed on this many randomly-
            sampled pixels across the test set (full-cube ECE on a
            512×512×Ntiles ×23-class softmax would be GB-scale).

    Returns:
        ``{"ensemble": EvalResult, ...baselines...}``.
    """
    split_dir = Path(split_dir)
    test_file = split_dir / "test.txt"
    if not test_file.exists():
        raise FileNotFoundError(f"No test.txt in {split_dir}")
    test_tiles = [
        line.strip()
        for line in test_file.read_text().splitlines()
        if line.strip()
    ]
    if not test_tiles:
        raise ValueError(f"{test_file} is empty")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the prediction function for the ensemble.
    if model_factory is None:
        if ensemble_checkpoint is not None:
            print(
                "  [phase 1] WARN: ensemble_checkpoint given but no "
                "model_factory — using random predictions (dry-run).",
                flush=True,
            )
        predict_fn = random_prediction_factory(num_classes=num_classes)
    else:
        if ensemble_checkpoint is None:
            raise ValueError(
                "model_factory provided but ensemble_checkpoint is None"
            )
        predict_fn = model_factory(ensemble_checkpoint)

    results: dict[str, EvalResult] = {}
    results["ensemble"] = _eval_one_predictor(
        name="ensemble",
        predict_fn=predict_fn,
        tile_names=test_tiles,
        tiles_dir=tiles_dir,
        num_classes=num_classes,
        ignore_index=ignore_index,
        ece_pixel_sample=ece_pixel_sample,
        output_dir=output_dir,
    )

    # Baselines.
    if baselines:
        from .baselines import BASELINES
        train_file = split_dir / "train.txt"
        train_tiles = (
            [line.strip() for line in train_file.read_text().splitlines()
             if line.strip()]
            if train_file.exists()
            else []
        )
        for key in baselines:
            if key not in BASELINES:
                print(f"  [phase 1] skipping unknown baseline: {key}",
                      flush=True)
                continue
            cls = BASELINES[key]
            try:
                baseline = cls()
                baseline.train(train_tiles, tiles_dir)
                baseline_predict = (
                    lambda npz_path, _b=baseline:
                    _adapt_baseline_predict(_b, npz_path, tiles_dir)
                )
                results[key] = _eval_one_predictor(
                    name=key,
                    predict_fn=baseline_predict,
                    tile_names=test_tiles,
                    tiles_dir=tiles_dir,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    ece_pixel_sample=0,  # baselines don't emit calibrated probs
                    output_dir=output_dir,
                )
            except NotImplementedError as e:
                print(f"  [phase 1] baseline {key} not implemented: {e}",
                      flush=True)

    if output_dir is not None:
        with (output_dir / "phase_1_in_distribution.json").open("w") as f:
            json.dump(
                {k: r.to_jsonable() for k, r in results.items()},
                f, indent=2,
            )
    return results


# ── Helpers ─────────────────────────────────────────────────────────────────


def _adapt_baseline_predict(baseline, npz_path, tiles_dir):
    """Bridge baselines (which take ``tile_name``) to the ``npz_path``
    interface used by ensemble predictors. Baselines don't carry
    softmax distributions; we return a one-hot to keep the call
    site uniform."""
    name = os.path.basename(str(npz_path))
    if name.endswith(".npz"):
        name = name[:-4]
    pred = baseline.predict(name, tiles_dir)
    return pred, None


def _eval_one_predictor(
    *,
    name: str,
    predict_fn,
    tile_names: list[str],
    tiles_dir: str,
    num_classes: int,
    ignore_index: int,
    ece_pixel_sample: int,
    output_dir: Path | None,
) -> EvalResult:
    """Run a single predictor across the test split and aggregate metrics.

    Strategy: accumulate the full confusion matrix as we go (cheap —
    23×23 int64), and reservoir-sample ``ece_pixel_sample`` pixel-level
    (probs, target) pairs for ECE so we don't need to keep all softmax
    tensors in memory.
    """
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_tile_metrics: list[dict] = []
    total_pixels = 0
    accuracy_correct = 0
    accuracy_total = 0

    # Reservoir for ECE — collected only when the predictor returns
    # ``probs`` (i.e., not for the trivial baselines).
    ece_probs: list[np.ndarray] = []
    ece_targets: list[np.ndarray] = []
    ece_collected = 0

    rng = np.random.default_rng(0)

    for i, tile_name in enumerate(tile_names):
        npz_path = os.path.join(tiles_dir, f"{tile_name}.npz")
        if not os.path.exists(npz_path):
            continue
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "label" not in data.files:
                    continue
                target = np.asarray(data["label"]).astype(np.int64)
        except Exception:
            continue

        try:
            pred, probs = predict_fn(npz_path)
        except Exception as exc:
            print(f"  [phase 1:{name}] {tile_name}: predict failed "
                  f"{type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue

        pred = np.asarray(pred).astype(np.int64)
        if pred.shape != target.shape:
            # Center-crop / pad to match — eval-time defensive resize.
            pred = _resize_or_pad_to(pred, target.shape)
            if probs is not None and probs.ndim == 3:
                probs = np.stack(
                    [_resize_or_pad_to(probs[c], target.shape, fill=0.0)
                     for c in range(probs.shape[0])]
                )

        valid = target != ignore_index
        if not valid.any():
            continue
        p = pred[valid]
        t = target[valid]
        # Bound predictions into [0, num_classes) — predictors might
        # emit garbage outside that range; we still want a confusion
        # matrix that sums to total valid pixels.
        p = np.clip(p, 0, num_classes - 1)
        t = np.clip(t, 0, num_classes - 1)

        # Accumulate confusion matrix via bincount on flat index.
        idx = t * num_classes + p
        bc = np.bincount(idx, minlength=num_classes * num_classes)
        conf_mat += bc.reshape(num_classes, num_classes)

        n_pix = int(valid.sum())
        total_pixels += n_pix
        accuracy_correct += int((p == t).sum())
        accuracy_total += n_pix

        # Per-tile mIoU for failure-mode use.
        tile_iou = per_class_iou(pred, target, num_classes,
                                 ignore_index=ignore_index)
        per_tile_metrics.append({
            "tile":   tile_name,
            "n_pix":  n_pix,
            "mIoU":   tile_iou.get("mean_iou"),
        })

        # ECE reservoir sampling.
        if probs is not None and ece_pixel_sample > 0:
            cls_first = probs.transpose(1, 2, 0).reshape(-1, num_classes)
            tgt_flat = target.reshape(-1)
            valid_flat = tgt_flat != ignore_index
            cls_first = cls_first[valid_flat]
            tgt_flat = tgt_flat[valid_flat]
            if cls_first.size > 0:
                take = min(
                    cls_first.shape[0],
                    max(1, ece_pixel_sample // max(1, len(tile_names))),
                )
                pick = rng.choice(cls_first.shape[0], size=take, replace=False)
                ece_probs.append(cls_first[pick])
                ece_targets.append(tgt_flat[pick])
                ece_collected += take

        if (i + 1) % 50 == 0 or (i + 1) == len(tile_names):
            print(
                f"  [phase 1:{name}] {i + 1}/{len(tile_names)} tiles "
                f"({accuracy_correct/max(1,accuracy_total):.3f} acc, "
                f"{total_pixels:,} pixels)",
                flush=True,
            )

    # Per-class IoU from the global confusion matrix.
    per_class = {}
    ious = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        tp = conf_mat[c, c]
        fn = conf_mat[c, :].sum() - tp
        fp = conf_mat[:, c].sum() - tp
        denom = tp + fn + fp
        if denom == 0:
            iou = float("nan")
        else:
            iou = float(tp) / float(denom)
            ious.append(iou)
        per_class[str(c)] = iou
    mIoU = float(np.nanmean(ious)) if ious else 0.0

    # Cohen's kappa (linear) — overall agreement vs chance.
    n = float(conf_mat.sum())
    if n > 0:
        po = float(np.trace(conf_mat)) / n
        pe = float((conf_mat.sum(axis=0) * conf_mat.sum(axis=1)).sum()) / (n * n)
        kappa = (po - pe) / (1 - pe) if pe < 1.0 else 0.0
    else:
        kappa = 0.0

    metrics = {
        "mIoU":         mIoU,
        "accuracy":     accuracy_correct / max(1, accuracy_total),
        "kappa":        float(kappa),
        "num_tiles":    len(per_tile_metrics),
        "num_pixels":   total_pixels,
    }

    if ece_collected > 0:
        all_probs = np.concatenate(ece_probs, axis=0)
        all_targets = np.concatenate(ece_targets, axis=0)
        metrics["ECE"] = expected_calibration_error(
            all_probs, all_targets,
            ignore_index=ignore_index,
        )
        metrics["ECE_n_pixels_sampled"] = ece_collected

    # Per-tile mIoU CDF data for the report generator.
    per_tile_miou = sorted(
        m["mIoU"] for m in per_tile_metrics if m["mIoU"] is not None
    )
    if per_tile_miou:
        metrics["per_tile_mIoU_p10"] = float(np.percentile(per_tile_miou, 10))
        metrics["per_tile_mIoU_p50"] = float(np.percentile(per_tile_miou, 50))
        metrics["per_tile_mIoU_p90"] = float(np.percentile(per_tile_miou, 90))

    # Persist per-tile metrics + confusion matrix for the failure-mode
    # phase to consume.
    if output_dir is not None:
        np.save(
            output_dir / f"confusion_phase_1_{name}.npy",
            conf_mat,
        )
        with (output_dir / f"per_tile_phase_1_{name}.json").open("w") as f:
            json.dump(per_tile_metrics, f)

    return EvalResult(
        phase="in_distribution",
        split_name="test",
        num_tiles=len(per_tile_metrics),
        num_pixels=total_pixels,
        metrics=metrics,
        per_class=per_class,
        confusion_matrix=conf_mat,
        notes={"predictor": name, "ece_collected": ece_collected},
    )


def _resize_or_pad_to(arr: np.ndarray, target_shape, fill=0) -> np.ndarray:
    """Defensive centre-crop / zero-pad — never resamples.

    Predictor may emit a slightly differently-sized tile (e.g. 224
    Prithvi patch vs 256 label). Cropping + padding keeps the
    confusion-matrix-builder honest without sneaking in a bilinear
    resize that would silently change pixel-level metrics.
    """
    if arr.shape == target_shape:
        return arr
    th, tw = target_shape
    ph, pw = arr.shape[-2], arr.shape[-1]
    out = np.full(target_shape, fill, dtype=arr.dtype)
    h = min(ph, th)
    w = min(pw, tw)
    # Centre crop source.
    src_y = max(0, (ph - th) // 2)
    src_x = max(0, (pw - tw) // 2)
    # Centre place target.
    dst_y = max(0, (th - ph) // 2)
    dst_x = max(0, (tw - pw) // 2)
    out[dst_y:dst_y + h, dst_x:dst_x + w] = (
        arr[src_y:src_y + h, src_x:src_x + w]
    )
    return out
