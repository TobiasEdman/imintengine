"""scripts/validate_against_nfi.py — score the LULC model against NFI plots.

Independent field-truth validation: sample the model's per-pixel softmax at
SLU NFI plot locations (from the Phase-1 plot→tile index built by
``nfi_tile_coverage.py``) and score the prediction against the plot's measured
forest type.

**Scope.** The production model is single-head — 23-class LULC, where "harvest"
is class 22 *hygge* (a clear-cut), NOT a standing-maturity signal. So this
validates **forest type**: NFI dominant species → {tallskog, granskog,
lövskog, blandskog} vs the predicted class, plus per-class AUROC of the softmax
sampled at plot pixels. Validating standing-maturity or biophysical regression
needs the Track-T heads (which don't exist yet).

**Design.** The scoring core ``score_against_nfi(index_df, predict_fn)`` takes a
``predict_fn(tile_path) -> (class_map, probs)`` so it is unit-testable with a
mock (``tests/test_validate_against_nfi.py``). ``make_model_predict_fn`` is the
real wiring (``load_model`` + sliding-window inference) the ICE job uses with
the checkpoint's matching aux flags. Run on the ICE PVC, where the full
``unified_v2`` tiles and the co-located plots live — locally there is no
plot∩tile overlap on unified-format tiles to score against.

    python scripts/validate_against_nfi.py \
        --checkpoint checkpoints/unified_v6a/best_model.pt \
        --data-dir /data/unified_v2 --plot-index data/nfi/nfi_plot_tile_index.parquet \
        --enable-all-aux --out docs/data/nfi-validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.eval.metrics import auroc_aupr

# Unified-schema forest classes (imint/training/unified_schema.py).
TALLSKOG, GRANSKOG, LOVSKOG, BLANDSKOG = 1, 2, 3, 4
FOREST_CLASSES = (TALLSKOG, GRANSKOG, LOVSKOG, BLANDSKOG)
FOREST_NAMES = {1: "tallskog", 2: "granskog", 3: "lövskog", 4: "blandskog"}
MATURE_FROM_CLASS = 41  # NFI Maturityclass ≥ 41 = final-felling-age / overmature


def derive_nfi_forest_class(row, *, dominant_frac: float = 0.7) -> int | None:
    """NFI per-species volume → unified forest class, or None if non-treed.

    A labelling decision the harness owns (the loader stays neutral). Rule:
    split standing volume into conifer (pine + contorta + spruce) vs deciduous
    (birch + other); if one side is ≥ ``dominant_frac`` of the total it's that
    type (pine vs spruce by larger volume), else *blandskog* (mixed). Returns
    None when the plot carries no standing volume (treeless / non-forest) —
    *sumpskog* (swamp forest, class 5) is a site condition, not derivable from
    species, so it is deliberately not produced here.
    """
    pine = float(row["VolPine"]) + float(row["VolContorta"])
    conifer = pine + float(row["VolSpruce"])
    decid = float(row["VolBirch"]) + float(row["VolOtherDec"])
    total = conifer + decid
    if total <= 0:
        return None
    if conifer / total >= dominant_frac:
        return TALLSKOG if pine >= float(row["VolSpruce"]) else GRANSKOG
    if decid / total >= dominant_frac:
        return LOVSKOG
    return BLANDSKOG


def nfi_is_mature(row) -> int:
    """1 if the plot is final-felling-age (NFI Maturityclass ≥ 41), else 0."""
    m = row.get("Maturityclass")
    return int(m is not None and not pd.isna(m) and float(m) >= MATURE_FROM_CLASS)


def score_against_nfi(
    index_df: pd.DataFrame,
    predict_fn,
    *,
    num_classes: int = 23,
    dominant_frac: float = 0.7,
) -> dict:
    """Sample predictions at plot pixels and score forest-type agreement.

    Args:
        index_df: the plot→tile index (``tile_name``, ``tile_path``, ``row``,
            ``col`` + the NFI columns), from ``nfi_tile_coverage.py``.
        predict_fn: ``tile_path -> (class_map (H,W) int, probs (C,H,W) float)``.
            Called once per tile.
        num_classes: softmax width (23 for the unified schema).
        dominant_frac: conifer/deciduous dominance threshold.

    Returns:
        A JSON-able dict: plot counts, forest-type overall accuracy, the
        forest-class confusion matrix (NFI truth × predicted), and per-class
        AUROC/AUPR of the sampled softmax.
    """
    pred_class: list[int] = []
    nfi_class: list[int | None] = []
    mature: list[int] = []
    probs_at_plot: list[np.ndarray] = []

    for tile_name, grp in index_df.groupby("tile_name", sort=False):
        tile_path = grp["tile_path"].iloc[0] if "tile_path" in grp else tile_name
        class_map, probs = predict_fn(tile_path)
        for _, r in grp.iterrows():
            rr, cc = int(r["row"]), int(r["col"])
            pred_class.append(int(class_map[rr, cc]))
            nfi_class.append(derive_nfi_forest_class(r, dominant_frac=dominant_frac))
            mature.append(nfi_is_mature(r))
            probs_at_plot.append(np.asarray(probs[:, rr, cc], dtype=np.float64))

    pred = np.array(pred_class)
    truth = np.array([c if c is not None else -1 for c in nfi_class])
    P = np.vstack(probs_at_plot) if probs_at_plot else np.zeros((0, num_classes))

    forest = truth >= 1  # plots with a derivable forest class
    n_forest = int(forest.sum())
    accuracy = float((pred[forest] == truth[forest]).mean()) if n_forest else float("nan")

    confusion = {
        FOREST_NAMES[t]: {
            FOREST_NAMES.get(int(p), f"class_{int(p)}"): int(((truth == t) & (pred == p)).sum())
            for p in np.unique(pred[truth == t])
        }
        for t in FOREST_CLASSES
        if (truth == t).any()
    }

    per_class_auroc = {}
    if len(P):
        for c in FOREST_CLASSES:
            y = (truth == c).astype(int)
            if 0 < y.sum() < len(y):
                a, p = auroc_aupr(P[:, c], y)
                per_class_auroc[FOREST_NAMES[c]] = {"auroc": round(a, 4), "aupr": round(p, 4)}

    return {
        "n_plots": int(len(pred)),
        "n_forest": n_forest,
        "n_mature": int(np.array(mature).sum()),
        "forest_type_accuracy": accuracy,
        "confusion_nfi_x_pred": confusion,
        "per_class_auroc": per_class_auroc,
    }


def make_model_predict_fn(checkpoint: str, device, img_size: int):
    """Real ``predict_fn`` for a UNIFIED-format checkpoint (v8+, 10-aux).

    Reuses ``inference_comparison.{load_model, run_inference}`` — the same
    multitemporal + aux normalization the model was trained with (spectral
    reflectance → Prithvi z-score, ``AUX_CHANNEL_NAMES`` from
    ``unified_dataset``, temporal/location coords). This replaces the retired
    ``LULCDataset`` wiring: the 512 tiles are ``spectral``/``multitemporal``
    format, and the checkpoints since v8 are 10-aux (no leaky
    ``harvest_probability``), so the old 11-aux ``LULCDataset`` path could
    never be scored (see docs/data/nfi_validation_findings.md).

    ``run_inference`` centre-crops each tile to ``img_size`` (504 for the
    600M patch-14 backbone on 512 tiles); the crop offset is returned so the
    caller can remap plot ``(row, col)`` and drop plots in the discarded
    border. ``predict_fn(tile_path) -> (class_map (cs,cs), probs (C,cs,cs))``
    is in CROP coordinates.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_infcmp", str(Path(__file__).resolve().parent / "inference_comparison.py"),
    )
    infcmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infcmp)

    model, epoch, miou, model_img_size = infcmp.load_model(checkpoint, device)
    print(f"  [load_model] epoch={epoch} ckpt_mIoU={miou} native_img={model_img_size}")

    def predict_fn(tile_path):
        probs, _raw_spectral, _raw_aux = infcmp.run_inference(
            model, tile_path, device, img_size=img_size, return_probs=True,
        )  # probs: (C, cs, cs)
        return probs.argmax(0).astype(np.int64), probs

    return predict_fn


def crop_offset(tile_h: int, img_size: int) -> int:
    """Centre-crop top-left offset ``run_inference`` applies (matches its
    ``(h - crop_sz) // 2``)."""
    return (tile_h - min(img_size, tile_h)) // 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--plot-index", required=True, help="parquet from nfi_tile_coverage.py")
    ap.add_argument("--out", default="docs/data/nfi-validation.json")
    ap.add_argument("--img-size", type=int, default=504,
                    help="inference crop (504 = 600M patch-14 on 512 tiles)")
    ap.add_argument("--num-classes", type=int, default=23)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    index_df = pd.read_parquet(args.plot_index)
    print(f"plot index: {len(index_df):,} co-located plots on "
          f"{index_df['tile_name'].nunique()} tiles")

    # The index was built against an earlier unified_v2_512 snapshot; merge/
    # rename since then dropped some tiles. Filter to plots whose tile still
    # exists so a stale row can't abort the whole run (FileNotFoundError).
    import os
    exists = index_df["tile_path"].map(os.path.exists)
    if not exists.all():
        gone = int((~exists).sum())
        print(f"dropping {gone} plots on tiles no longer in the dataset "
              f"({index_df.loc[~exists, 'tile_name'].nunique()} tiles)")
        index_df = index_df[exists].copy()

    # run_inference centre-crops to img_size; remap plot (row,col) into crop
    # coords and drop plots in the discarded border (else they'd index the
    # wrong pixel / fall outside the returned array).
    sample_path = index_df["tile_path"].iloc[0]
    tile_h = int(np.load(sample_path, allow_pickle=True)["spectral"].shape[-1])
    off = crop_offset(tile_h, args.img_size)
    cs = min(args.img_size, tile_h)
    before = len(index_df)
    index_df = index_df[
        (index_df["row"] >= off) & (index_df["row"] < off + cs)
        & (index_df["col"] >= off) & (index_df["col"] < off + cs)
    ].copy()
    index_df["row"] -= off
    index_df["col"] -= off
    print(f"crop offset={off} (tile {tile_h}→{cs}); kept {len(index_df)}/{before} "
          f"plots in-crop ({before - len(index_df)} border-dropped)")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    predict_fn = make_model_predict_fn(args.checkpoint, device, args.img_size)

    results = score_against_nfi(index_df, predict_fn, num_classes=args.num_classes)
    results["_meta"] = {
        "checkpoint": args.checkpoint, "img_size": args.img_size,
        "plots_in_crop": len(index_df), "plots_total": before,
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
