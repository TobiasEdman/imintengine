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

# Full accuracy-suite classes: the 4 forest types + a single "non-forest /
# other" bucket (0) that collects every treeless-truth plot and every
# non-forest prediction. Scoring over ALL plots (not just forest-truth) is
# what makes user's accuracy (precision) and Cohen's kappa well-defined —
# they need the plots the forest-only metric drops. Mirrors how NMD /
# WorldCover validation reports the standard confusion-matrix measures.
NONFOREST = 0
SUITE_CLASSES = (TALLSKOG, GRANSKOG, LOVSKOG, BLANDSKOG, NONFOREST)
SUITE_NAMES = {**FOREST_NAMES, 0: "annat (icke-skog)"}


def _collapse(v: int) -> int:
    """Map a class id to the suite space: keep 1-4, everything else → 0."""
    return int(v) if int(v) in (1, 2, 3, 4) else NONFOREST


def accuracy_suite(truth, pred) -> dict:
    """Standard confusion-matrix accuracy measures over SUITE_CLASSES.

    ``truth`` / ``pred`` are collapsed to {1,2,3,4, 0=non-forest} first, so
    every plot is counted. Returns overall accuracy, per-class user's
    accuracy (precision), producer's accuracy (recall), F1 and support, and
    Cohen's kappa (chance-corrected agreement over all classes).
    """
    import numpy as np

    t = np.array([_collapse(x) for x in truth])
    p = np.array([_collapse(x) for x in pred])
    idx = {c: i for i, c in enumerate(SUITE_CLASSES)}
    k = len(SUITE_CLASSES)
    cm = np.zeros((k, k), dtype=np.int64)  # rows = truth, cols = pred
    for tt, pp in zip(t, p):
        cm[idx[tt], idx[pp]] += 1
    n = int(cm.sum())

    overall = float(np.trace(cm) / n) if n else float("nan")
    per_class = {}
    for c in SUITE_CLASSES:
        i = idx[c]
        tp = int(cm[i, i])
        row = int(cm[i, :].sum())   # truth = c  → producer's denom
        col = int(cm[:, i].sum())   # pred  = c  → user's denom
        prod = tp / row if row else float("nan")   # recall
        user = tp / col if col else float("nan")   # precision
        f1 = (2 * user * prod / (user + prod)
              if (user and prod and not np.isnan(user) and not np.isnan(prod))
              else 0.0)
        per_class[SUITE_NAMES[c]] = {
            "users_accuracy": round(user, 4) if col else None,
            "producers_accuracy": round(prod, 4) if row else None,
            "f1": round(f1, 4),
            "support": row,
        }

    # Cohen's kappa: (po - pe) / (1 - pe).
    po = overall
    row_tot = cm.sum(axis=1)
    col_tot = cm.sum(axis=0)
    pe = float((row_tot * col_tot).sum() / (n * n)) if n else 0.0
    kappa = (po - pe) / (1 - pe) if (1 - pe) else float("nan")

    return {
        "n_plots_all": n,
        "overall_accuracy_5class": round(overall, 4),
        "cohen_kappa": round(kappa, 4),
        "per_class": per_class,
    }


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


def collapse_fractions_to_nfi_class(
    tall: float, gran: float, trivial: float, adel: float,
    *, dominant_frac: float = 0.7, forest_floor: float = 0.1,
) -> int:
    """Collapse 4 predicted crown-cover fractions with the NFI dominance rule.

    This is the fraction-head analogue of ``derive_nfi_forest_class``: instead
    of argmaxing the 28-class hard head, we take the fraction head's per-species
    crown-cover (each in [0, 1], from sigmoid) at a plot pixel and apply the
    SAME dominance logic the NFI truth uses — collapse-rule alignment is the
    whole point of the experiment.

        conifer = tall + gran
        decid   = trivial + adel
        total   = conifer + decid
        total < forest_floor            → 0  (non-forest)
        conifer/total ≥ dominant_frac   → 1 tall  if tall ≥ gran else 2 gran
        decid/total   ≥ dominant_frac   → 3 löv
        otherwise                       → 4 bland

    Args:
        tall, gran, trivial, adel: predicted crown-cover fractions in [0, 1].
        dominant_frac: one-side dominance threshold (matches NFI's 0.7).
        forest_floor: minimum summed crown-cover to count as forest at all;
            below it the pixel collapses to non-forest (0).

    Returns:
        Unified forest class in {0, 1, 2, 3, 4}.
    """
    conifer = float(tall) + float(gran)
    decid = float(trivial) + float(adel)
    total = conifer + decid
    if total < forest_floor:
        return NONFOREST
    if conifer / total >= dominant_frac:
        return TALLSKOG if tall >= gran else GRANSKOG
    if decid / total >= dominant_frac:
        return LOVSKOG
    return BLANDSKOG


def nfi_is_mature(row) -> int:
    """1 if the plot is final-felling-age (NFI Maturityclass ≥ 41), else 0."""
    m = row.get("Maturityclass")
    return int(m is not None and not pd.isna(m) and float(m) >= MATURE_FROM_CLASS)


# Plot identifiers carried into a per-plot dump when requested. TractID+PlotID
# uniquely key an NFI plot, so a downstream consumer can re-join to the full
# nfi_plots table for coordinates even if Easting/Northing are absent here.
_PER_PLOT_ID_COLS = ("TractID", "PlotID", "Year", "Easting", "Northing")


def score_against_nfi(
    index_df: pd.DataFrame,
    predict_fn,
    *,
    num_classes: int = 23,
    dominant_frac: float = 0.7,
    per_plot_sink: list | None = None,
) -> dict:
    """Sample predictions at plot pixels and score forest-type agreement.

    Args:
        index_df: the plot→tile index (``tile_name``, ``tile_path``, ``row``,
            ``col`` + the NFI columns), from ``nfi_tile_coverage.py``.
        predict_fn: ``tile_path -> (class_map (H,W) int, probs (C,H,W) float)``.
            Called once per tile.
        num_classes: softmax width (23 for the unified schema).
        dominant_frac: conifer/deciduous dominance threshold.
        per_plot_sink: if given, one dict per scored plot is appended
            (identifiers + NFI forest truth + model prediction). Enables an
            external same-plots comparison (e.g. NMD2023/NMD2018 sampled at the
            identical coordinates) without re-running inference.

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
            pc = int(class_map[rr, cc])
            nc = derive_nfi_forest_class(r, dominant_frac=dominant_frac)
            pred_class.append(pc)
            nfi_class.append(nc)
            mature.append(nfi_is_mature(r))
            probs_at_plot.append(np.asarray(probs[:, rr, cc], dtype=np.float64))
            if per_plot_sink is not None:
                rec = {k: r[k] for k in _PER_PLOT_ID_COLS if k in r}
                rec.update(tile_name=str(tile_name),
                           nfi_forest=int(nc) if nc is not None else -1,
                           model_pred=pc)
                # Channels 1-4 of the prob vector: softmax probs in hard mode,
                # RAW crown-cover fractions in --use-fraction-head mode — the
                # latter is what lets collapse thresholds (forest_floor /
                # dominant_frac) be calibrated offline from the dump.
                rec.update({f"p{k}": float(probs[k, rr, cc]) for k in (1, 2, 3, 4)})
                per_plot_sink.append(rec)

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
        # Standard confusion-matrix suite over ALL plots (forest + non-forest):
        # overall accuracy, per-class user's/producer's/F1, Cohen's kappa.
        "accuracy_suite": accuracy_suite(
            [c if c is not None else 0 for c in nfi_class], pred_class),
    }


def make_model_predict_fn(checkpoint: str, device, img_size: int,
                          aux_channel_names=None, backbone_name=None):
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

    model, epoch, miou, model_img_size = infcmp.load_model(
        checkpoint, device, backbone_name=backbone_name)
    print(f"  [load_model] epoch={epoch} ckpt_mIoU={miou} native_img={model_img_size}")

    def predict_fn(tile_path):
        probs, _raw_spectral, _raw_aux = infcmp.run_inference(
            model, tile_path, device, img_size=img_size, return_probs=True,
            aux_channel_names=aux_channel_names,
        )  # probs: (C, cs, cs)
        return probs.argmax(0).astype(np.int64), probs

    return predict_fn


def make_fraction_predict_fn(
    checkpoint: str, device, img_size: int, aux_channel_names=None,
    *, dominant_frac: float = 0.7, forest_floor: float = 0.1,
    num_classes: int = 28, backbone_name=None,
):
    """``predict_fn`` that collapses the FRACTION HEAD with the NFI rule.

    Instead of argmaxing the 28-class hard head, this runs the fraction head,
    takes its 4 sigmoid crown-cover maps, and applies
    ``collapse_fractions_to_nfi_class`` per pixel → a {0..4} class map. The
    returned ``probs`` array is num_classes-wide but only channels 1-4 are
    populated (with the tall/gran/(trivial+adel löv) fractions) so the existing
    per-class AUROC over FOREST_CLASSES still resolves; every other channel is
    left at 0. ``predict_fn(tile_path) -> (class_map (cs,cs), probs (C,cs,cs))``
    in CROP coordinates (same centre-crop as the hard path).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_infcmp", str(Path(__file__).resolve().parent / "inference_comparison.py"),
    )
    infcmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infcmp)

    model, epoch, miou, model_img_size = infcmp.load_model(
        checkpoint, device, backbone_name=backbone_name)
    if getattr(model, "frac_head", None) is None:
        raise ValueError(
            "checkpoint has no fraction head — retrain with "
            "--enable-tradslag-head or drop --use-fraction-head"
        )
    print(f"  [load_model] epoch={epoch} ckpt_mIoU={miou} native_img={model_img_size} "
          f"(fraction-head mode)")

    def predict_fn(tile_path):
        fracs = infcmp.run_fraction_inference(
            model, tile_path, device, img_size=img_size,
            aux_channel_names=aux_channel_names,
        )  # (4, cs, cs) in [0,1], order tall/gran/trivial/adel
        k, cs, _ = fracs.shape
        # Vectorized collapse over the whole crop.
        tall, gran, trivial, adel = fracs[0], fracs[1], fracs[2], fracs[3]
        conifer = tall + gran
        decid = trivial + adel
        total = conifer + decid
        with np.errstate(divide="ignore", invalid="ignore"):
            conifer_share = np.where(total > 0, conifer / total, 0.0)
            decid_share = np.where(total > 0, decid / total, 0.0)
        class_map = np.zeros((cs, cs), dtype=np.int64)  # default non-forest
        is_forest = total >= forest_floor
        conif_dom = is_forest & (conifer_share >= dominant_frac)
        decid_dom = is_forest & (decid_share >= dominant_frac)
        bland = is_forest & ~conif_dom & ~decid_dom
        class_map[conif_dom & (tall >= gran)] = TALLSKOG
        class_map[conif_dom & (tall < gran)] = GRANSKOG
        class_map[decid_dom] = LOVSKOG
        class_map[bland] = BLANDSKOG
        # probs: put the raw fractions on the forest-class channels so
        # per-class AUROC (which reads P[:, c] for c in 1..4) stays defined.
        probs = np.zeros((num_classes, cs, cs), dtype=np.float32)
        probs[TALLSKOG] = tall
        probs[GRANSKOG] = gran
        probs[LOVSKOG] = decid          # deciduous total drives "löv" ranking
        probs[BLANDSKOG] = np.minimum(conifer, decid)  # mixedness proxy
        return class_map, probs

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
    ap.add_argument("--dump-per-plot", default=None,
                    help="parquet path: one row per scored plot (identifiers + "
                         "NFI forest truth + model prediction) for an external "
                         "same-plots comparison")
    ap.add_argument("--img-size", type=int, default=504,
                    help="inference crop (504 = 600M patch-14 on 512 tiles)")
    ap.add_argument("--num-classes", type=int, default=23)
    ap.add_argument("--backbone-name", default=None,
                    help="override the checkpoint's backbone (required for "
                    "non-Prithvi families, e.g. tessera_v1, whose minimal "
                    "config omits backbone_name and has no pos_embed to infer)")
    ap.add_argument("--use-fraction-head", action="store_true",
                    help="Collapse the Trädslag fraction head with the NFI "
                         "dominance rule instead of argmaxing the class head. "
                         "Requires a checkpoint trained with "
                         "--enable-tradslag-head.")
    ap.add_argument("--forest-floor", type=float, default=0.1,
                    help="Min summed crown-cover to count as forest in the "
                         "fraction-head collapse (below → non-forest). "
                         "Default 0.1.")
    ap.add_argument("--dominant-frac", type=float, default=0.7,
                    help="One-side dominance threshold for the collapse rule "
                         "(matches NFI's 0.7).")
    ap.add_argument("--enable-markfukt", action="store_true",
                    help="feed markfukt as the 11th aux (for a wetness-aux "
                         "checkpoint); appends it to the canonical 10")
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
    aux_names = None
    if args.enable_markfukt:
        from imint.training.unified_dataset import AUX_CHANNEL_NAMES
        aux_names = list(AUX_CHANNEL_NAMES) + ["markfukt"]
        print(f"  markfukt enabled → {len(aux_names)} aux channels")
    if args.use_fraction_head:
        predict_fn = make_fraction_predict_fn(
            args.checkpoint, device, args.img_size,
            aux_channel_names=aux_names,
            dominant_frac=args.dominant_frac, forest_floor=args.forest_floor,
            num_classes=args.num_classes, backbone_name=args.backbone_name,
        )
        print(f"  fraction-head collapse: dominant_frac={args.dominant_frac}, "
              f"forest_floor={args.forest_floor}")
    else:
        predict_fn = make_model_predict_fn(args.checkpoint, device, args.img_size,
                                           aux_channel_names=aux_names,
                                           backbone_name=args.backbone_name)

    per_plot: list | None = [] if args.dump_per_plot else None
    results = score_against_nfi(index_df, predict_fn, num_classes=args.num_classes,
                                dominant_frac=args.dominant_frac,
                                per_plot_sink=per_plot)
    results["_meta"] = {
        "checkpoint": args.checkpoint, "img_size": args.img_size,
        "plots_in_crop": len(index_df), "plots_total": before,
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")

    if args.dump_per_plot:
        pp = Path(args.dump_per_plot)
        pp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_plot).to_parquet(pp, index=False)
        print(f"wrote {pp} ({len(per_plot)} plots)")


if __name__ == "__main__":
    main()
