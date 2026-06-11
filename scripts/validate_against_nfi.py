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


def make_model_predict_fn(checkpoint: str, data_dir: str, config, device):
    """Real ``predict_fn`` (load_model + sliding-window). Verified on ICE.

    Reuses ``scripts.predict_lulc.load_model`` + ``imint.inference.sliding_window``
    and ``LULCDataset`` for the per-tile normalized input (the aux set must match
    the checkpoint — pass the matching ``--enable-*`` flags). Indexes dataset
    samples by tile name so ``predict_fn(tile_path)`` maps a plot-index row to
    the right tile. Not exercised in local CI: CPU inference is slow and there
    is no local plot∩tile overlap on unified-format tiles — the meaningful run
    is the ICE job against the PVC.
    """
    import torch

    from imint.inference.sliding_window import sliding_window_inference
    from imint.training.dataset import LULCDataset

    # load_model lives in the sibling script; import it without a package.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_predict_lulc", str(Path(__file__).resolve().parent / "predict_lulc.py"),
    )
    predict_lulc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict_lulc)

    model, _ = predict_lulc.load_model(checkpoint, config, device)
    aux_names = config.enabled_aux_names

    dataset = LULCDataset(data_dir, split="all", config=config)
    by_name = {}
    for i in range(len(dataset)):
        meta = dataset[i].get("metadata", {})
        by_name[meta.get("tile", f"tile_{i:04d}").replace(".npz", "")] = i

    def predict_fn(tile_path):
        name = Path(tile_path).stem
        sample = dataset[by_name[name]]
        image_5d = sample["spectral"].unsqueeze(0).unsqueeze(2).to(device)  # (1,6,1,H,W)
        aux_parts = [sample[n].unsqueeze(0).to(device) for n in aux_names if n in sample]
        aux = torch.cat(aux_parts, dim=1) if aux_parts else None
        probs = sliding_window_inference(
            model, image_5d, aux, num_classes=config.num_classes,
        )  # (1, C, H, W)
        probs = probs.squeeze(0).cpu().numpy()
        return probs.argmax(0).astype(np.int64), probs

    return predict_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True, help="tile dir LULCDataset reads")
    ap.add_argument("--plot-index", required=True, help="parquet from nfi_tile_coverage.py")
    ap.add_argument("--out", default="docs/data/nfi-validation.json")
    ap.add_argument("--enable-all-aux", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    from imint.training.config import TrainingConfig

    index_df = pd.read_parquet(args.plot_index)
    print(f"plot index: {len(index_df):,} co-located plots on {index_df['tile_name'].nunique()} tiles")

    config = TrainingConfig(
        data_dir=args.data_dir,
        enable_height_channel=args.enable_all_aux,
        enable_volume_channel=args.enable_all_aux,
        enable_basal_area_channel=args.enable_all_aux,
        enable_diameter_channel=args.enable_all_aux,
        enable_dem_channel=args.enable_all_aux,
        enable_vpp_channels=args.enable_all_aux,
    )
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    predict_fn = make_model_predict_fn(args.checkpoint, args.data_dir, config, device)

    results = score_against_nfi(index_df, predict_fn, num_classes=config.num_classes)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
