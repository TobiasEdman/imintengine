"""scripts/validate_against_lucas.py — validate ensemble members against LUCAS.

L2 of ``docs/experiments/lucas_validation_plan.md``. Swedish LUCAS field points
are an independent ground truth (the models trained on NMD2023 labels, never on
LUCAS) that touches ~20 of the 28 unified classes at ~160× the NFI sample. This
script samples each member's dense prediction at every co-located LUCAS point
(the L1 index ``data/lucas/lucas_tile_index.parquet``) and scores it two ways:

**L2b — hard 28-class head (all members).** Sample the argmax class map at each
point's ``(row, col)``; build the 28×28 confusion (truth × pred), per-class
producer/user accuracy + overall. Reported for three splits separately: all
co-located points, held-out ``test`` (~71, thin — flagged), and ``train``
(memorization-optimistic). Classes below ``--min-support`` → "insufficient
support", never a score.

**L2a — forest fraction (frac-head members only).** For forest points
(``unified_class`` in 1-4):
  * dominant-species argmax agreement — argmax of the model's {tall, gran, löv}
    fraction channels vs LUCAS ``forest_dominant`` (tall→1/gran→2/löv→3),
    threshold-free, on the dominated subset (``forest_dominant`` not None).
  * mixedness AUC — ROC-AUC of fraction *dispersion* (1 − normalized max forest
    fraction) vs ``is_mixed`` (class-4 mixed woodland = True).

**Crop year-sensitivity (critical).** Crops rotate annually, so a LUCAS crop
label is valid ONLY for its observation year. The L1 index is year-matched at
tile level (``require_year_match``); this script re-asserts it at runtime: every
crop-class point (``unified_class`` in 11..21) must have ``Year == tile spectral
year`` (read via ``imint.training.nfi_colocate.tile_year``). A violation raises
(fail-loud) — crops are never silently included on a wrong-year tile. Land-cover
and forest classes are year-robust (stable land cover), so their year-matching
is a safety default, not a correctness requirement — they are NOT subject to the
strict filter.

The scoring core ``score_against_lucas(index_df, predict_fn, ...)`` takes a
``predict_fn(tile_path) -> (class_map (H,W) int, probs (C,H,W) float)`` so it is
unit-testable with a mock (``tests/test_validate_against_lucas.py``). The real
wiring reuses ``validate_against_nfi.make_model_predict_fn`` /
``make_fraction_predict_fn`` — the same multitemporal + aux inference the model
was trained with. Run on the ICE PVC where the full ``unified_v2_512`` tiles and
the co-located points live.

    .venv/bin/python scripts/validate_against_lucas.py \
        --checkpoint checkpoints/distill/best_model.pt \
        --backbone-name prithvi_600m --img-size 504 --num-classes 28 \
        --data-dir /data/unified_v2_512 \
        --lucas-index /data/lucas/lucas_tile_index.parquet \
        --out docs/data/lucas-validation-distill.json \
        --dump-per-point docs/data/lucas-per-point-distill.parquet
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imint.training.unified_schema import (  # noqa: E402
    NUM_UNIFIED_CLASSES,
    UNIFIED_CLASSES,
)

# Reuse the NFI validator's real inference wiring (make_*_predict_fn) rather
# than re-implementing load_model + sliding-window.
_van_spec = importlib.util.spec_from_file_location(
    "validate_against_nfi",
    str(Path(__file__).resolve().parent / "validate_against_nfi.py"),
)
van = importlib.util.module_from_spec(_van_spec)
_van_spec.loader.exec_module(van)

# ── Class groupings (year semantics) ──────────────────────────────────────────
# Forest (1-4) is validated via L2a fraction when a frac head exists, else L2b.
FOREST_CLASSES = (1, 2, 3, 4)
# "Year-strict" crops: a LUCAS crop label is valid ONLY for its observation year
# (crops rotate annually), so these points MUST be on an exact-year-matched tile.
CROP_CLASSES = tuple(range(11, 22))  # 11..21 inclusive
# Everything else observable is year-robust land cover (forest, water, wetland,
# open/shrub/bare, built). Year-matching for these is a safety default, not a
# correctness requirement — they are NOT subject to the strict year filter.

# forest_dominant string → unified forest class for the L2a argmax check.
DOMINANT_TO_CLASS = {"tall": 1, "gran": 2, "lov": 3}


def class_name(c: int) -> str:
    return UNIFIED_CLASSES.get(int(c), f"class_{int(c)}")


def _confusion_and_per_class(
    truth: np.ndarray, pred: np.ndarray, *, num_classes: int, min_support: int
) -> dict:
    """28×28 confusion (truth×pred) + per-class producer/user accuracy.

    Classes with row-support (truth count) below ``min_support`` are reported as
    ``{"status": "insufficient support", "support": n}`` rather than a score —
    an accuracy on a handful of points is noise, not a metric.
    """
    n = int(len(truth))
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # rows=truth cols=pred
    for t, p in zip(truth, pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    overall = float(np.trace(cm) / n) if n else float("nan")

    per_class: dict[str, dict] = {}
    for c in range(num_classes):
        row = int(cm[c, :].sum())   # truth == c  → producer's denom (recall)
        if row == 0:
            continue  # class not present in truth at all → omit, not "0 support"
        col = int(cm[:, c].sum())   # pred == c    → user's denom (precision)
        tp = int(cm[c, c])
        if row < min_support:
            per_class[class_name(c)] = {
                "class_id": c,
                "status": "insufficient support",
                "support": row,
            }
            continue
        prod = tp / row if row else float("nan")
        user = tp / col if col else float("nan")
        f1 = (2 * user * prod / (user + prod)) if (user and prod) else 0.0
        per_class[class_name(c)] = {
            "class_id": c,
            "producers_accuracy": round(prod, 4),
            "users_accuracy": round(user, 4) if col else None,
            "f1": round(f1, 4),
            "support": row,
        }

    # Cohen's kappa over the populated classes for a chance-corrected headline.
    row_tot = cm.sum(axis=1)
    col_tot = cm.sum(axis=0)
    pe = float((row_tot * col_tot).sum() / (n * n)) if n else 0.0
    kappa = (overall - pe) / (1 - pe) if (1 - pe) else float("nan")

    return {
        "n_points": n,
        "overall_accuracy": round(overall, 4) if n else None,
        "cohen_kappa": round(kappa, 4) if n else None,
        "per_class": per_class,
    }


def score_l2b(
    df: pd.DataFrame, *, num_classes: int, min_support: int
) -> dict:
    """L2b hard 28-class breakdowns for all / test / train splits.

    ``df`` must already carry a ``pred_class`` column (sampled from the member's
    class map). Each split is scored independently so the held-out ``test``
    anchor and the memorization-optimistic ``train`` numbers are both visible.
    """
    truth = df["unified_class"].to_numpy(dtype=np.int64)
    pred = df["pred_class"].to_numpy(dtype=np.int64)

    out = {
        "all": _confusion_and_per_class(
            truth, pred, num_classes=num_classes, min_support=min_support
        )
    }
    for split in ("test", "train"):
        m = (df["split"] == split).to_numpy()
        sub = _confusion_and_per_class(
            truth[m], pred[m], num_classes=num_classes, min_support=min_support
        )
        if split == "test":
            sub["_note"] = (
                f"held-out split — thin (n={int(m.sum())}); treat per-class "
                "numbers as a sanity anchor, not a headline"
            )
        else:
            sub["_note"] = (
                "train split — on tiles the backbone saw; memorization-"
                "optimistic (LUCAS itself was never a training target)"
            )
        out[split] = sub
    return out


def score_l2a_fraction(df: pd.DataFrame) -> dict:
    """L2a forest-fraction metrics from the four forest-channel values.

    ``df`` must carry ``frac_tall``/``frac_gran``/``frac_trivial``/``frac_adel``
    (raw sigmoid crown-cover, order from the fraction head) for forest points
    (``unified_class`` in 1-4). Löv = trivial + adel. Returns:
      * dominant-species argmax agreement on the dominated subset
        (``forest_dominant`` not None), threshold-free.
      * mixedness ROC-AUC: dispersion (1 − normalized max forest fraction) vs
        ``is_mixed``.
    """
    forest = df[df["unified_class"].isin(FOREST_CLASSES)].copy()
    if forest.empty:
        return {"status": "no forest points"}

    tall = forest["frac_tall"].to_numpy(dtype=np.float64)
    gran = forest["frac_gran"].to_numpy(dtype=np.float64)
    lov = (forest["frac_trivial"].to_numpy(dtype=np.float64)
           + forest["frac_adel"].to_numpy(dtype=np.float64))
    three = np.vstack([tall, gran, lov]).T  # columns: tall, gran, löv

    # ── dominant-species argmax agreement (threshold-free) ──
    dominated = forest["forest_dominant"].notna().to_numpy()
    n_dom = int(dominated.sum())
    if n_dom:
        argmax_cls = np.array([1, 2, 3])[np.argmax(three[dominated], axis=1)]
        truth_cls = forest.loc[dominated, "forest_dominant"].map(
            DOMINANT_TO_CLASS
        ).to_numpy(dtype=np.int64)
        dom_acc = float((argmax_cls == truth_cls).mean())
        # per-species agreement for the failure-mode breakdown
        per_species = {}
        for name, cid in DOMINANT_TO_CLASS.items():
            m = truth_cls == cid
            per_species[name] = {
                "n": int(m.sum()),
                "agreement": round(float((argmax_cls[m] == cid).mean()), 4)
                if m.any() else None,
            }
    else:
        dom_acc, per_species = None, {}

    # ── mixedness ROC-AUC (dispersion vs is_mixed) ──
    all_forest3 = three  # over all forest points, mixed included
    row_sum = all_forest3.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        max_norm = np.where(row_sum > 0, all_forest3.max(axis=1) / row_sum, 0.0)
    dispersion = 1.0 - max_norm  # high for evenly-mixed, low for dominated
    y_mixed = forest["is_mixed"].to_numpy(dtype=int)

    auc = None
    if 0 < y_mixed.sum() < len(y_mixed):
        from sklearn.metrics import roc_auc_score
        auc = round(float(roc_auc_score(y_mixed, dispersion)), 4)

    return {
        "n_forest": int(len(forest)),
        "dominant_argmax": {
            "n_dominated": n_dom,
            "agreement": round(dom_acc, 4) if dom_acc is not None else None,
            "per_species": per_species,
            "_note": "argmax of {tall,gran,löv} fraction channels vs LUCAS "
                     "dominant; threshold-free (no collapse floor)",
        },
        "mixedness_auc": {
            "n_forest": int(len(forest)),
            "n_mixed": int(y_mixed.sum()),
            "roc_auc": auc,
            "_note": "ROC-AUC of fraction dispersion (1 − max/sum) vs is_mixed",
        },
    }


def assert_crop_year_match(index_df: pd.DataFrame, tile_years: dict) -> None:
    """Fail-loud if any crop-class point sits on a wrong-year tile.

    Crops rotate annually so a LUCAS crop label (unified_class 11..21) is valid
    ONLY for its observation year: the point's ``Year`` MUST equal the tile's
    spectral year. ``tile_years`` maps ``tile_name -> int year | None`` (read
    from the npz via ``nfi_colocate.tile_year``). A ``None`` year on a crop tile
    is also a violation — an undeterminable year cannot be certified year-exact.
    Non-crop (year-robust land-cover / forest) points are exempt.

    Raises ``ValueError`` listing the offending points on any mismatch.
    """
    crop = index_df[index_df["unified_class"].isin(CROP_CLASSES)]
    violations = []
    for _, r in crop.iterrows():
        ty = tile_years.get(r["tile_name"])
        if ty is None or int(ty) != int(r["Year"]):
            violations.append(
                f"point {r['point_id']} class {int(r['unified_class'])} "
                f"({class_name(r['unified_class'])}) Year={int(r['Year'])} "
                f"tile {r['tile_name']} spectral_year={ty}"
            )
    if violations:
        raise ValueError(
            f"CROP YEAR-MATCH VIOLATION — {len(violations)} crop point(s) on a "
            f"tile whose spectral year != the LUCAS observation year. Crop labels "
            f"are year-strict (annual rotation); refusing to score.\n  "
            + "\n  ".join(violations[:20])
            + ("\n  ..." if len(violations) > 20 else "")
        )


def score_against_lucas(
    index_df: pd.DataFrame,
    predict_fn,
    *,
    num_classes: int = NUM_UNIFIED_CLASSES,
    min_support: int = 20,
    is_fraction: bool = False,
    tile_years: dict | None = None,
    per_point_sink: list | None = None,
) -> dict:
    """Sample member predictions at LUCAS points and score L2b (+ L2a if frac).

    Args:
        index_df: L1 index (``tile_name``, ``tile_path``, ``row``, ``col``,
            ``unified_class``, ``forest_dominant``, ``is_mixed``, ``Year``,
            ``split``, ``source`` …).
        predict_fn: ``tile_path -> (class_map (H,W) int, probs (C,H,W) float)``,
            called once per tile. For a frac member, channels 1-4 of ``probs``
            hold the raw {tall, gran, löv-proxy, mixedness-proxy} — but L2a
            reads the RAW four fraction channels, so pass ``frac_predict_fn``
            (see ``make_fraction_frac_fn``) instead when ``is_fraction``.
        num_classes: confusion width (28 for the unified schema).
        min_support: per-class row-support floor; below → "insufficient support".
        is_fraction: True to compute L2a. ``predict_fn`` must then also populate
            ``frac_*`` — handled by the ``make_lucas_frac_predict_fn`` wrapper.
        tile_years: ``tile_name -> year`` for the crop year-match assertion. If
            None the assertion is skipped (mock tests pass it explicitly).
        per_point_sink: if given, one dict per scored point is appended.

    Returns:
        JSON-able dict: L2b split breakdowns, L2a (if frac), meta + year note.
    """
    # ── Crop year-match assertion (fail-loud) — BEFORE any scoring ──
    if tile_years is not None:
        assert_crop_year_match(index_df, tile_years)

    records = []
    for tile_name, grp in index_df.groupby("tile_name", sort=False):
        tile_path = grp["tile_path"].iloc[0] if "tile_path" in grp else tile_name
        out = predict_fn(tile_path)
        # frac path returns (class_map, probs, fracs); hard path (class_map, probs)
        if is_fraction:
            class_map, _probs, fracs = out
        else:
            class_map, _probs = out
            fracs = None
        for _, r in grp.iterrows():
            rr, cc = int(r["row"]), int(r["col"])
            rec = {
                "point_id": r["point_id"],
                "unified_class": int(r["unified_class"]),
                "unified_name": r.get("unified_name", class_name(r["unified_class"])),
                "pred_class": int(class_map[rr, cc]),
                "split": r["split"],
                "Year": int(r["Year"]),
                "source": r.get("source"),
                "forest_dominant": r.get("forest_dominant"),
                "is_mixed": bool(r.get("is_mixed", False)),
            }
            if fracs is not None:
                rec["frac_tall"] = float(fracs[0, rr, cc])
                rec["frac_gran"] = float(fracs[1, rr, cc])
                rec["frac_trivial"] = float(fracs[2, rr, cc])
                rec["frac_adel"] = float(fracs[3, rr, cc])
            records.append(rec)
            if per_point_sink is not None:
                per_point_sink.append(dict(rec))

    scored = pd.DataFrame(records)

    results = {
        "l2b_hard_28class": score_l2b(
            scored, num_classes=num_classes, min_support=min_support
        ),
        "_year_match_note": (
            "Crop metrics (unified_class 11-21) are year-matched: every crop "
            "point's LUCAS observation Year equals its tile's spectral year "
            "(asserted at runtime via nfi_colocate.tile_year; fail-loud on "
            "mismatch). Year-stable classes (forest 1-4, water 10, wetland 7, "
            "open/shrub/bare 8/24/27, built 9) are validated on all matched "
            "points — their year-matching is a safety default, not a correctness "
            "requirement (stable land cover)."
        ),
        "_class_groups": {
            "year_stable": sorted(
                set(int(c) for c in scored["unified_class"].unique())
                - set(CROP_CLASSES)
            ),
            "year_strict_crops": sorted(
                set(int(c) for c in scored["unified_class"].unique())
                & set(CROP_CLASSES)
            ),
        },
    }
    # Record the year of each crop point (per the plan) for auditability.
    crop_scored = scored[scored["unified_class"].isin(CROP_CLASSES)]
    if not crop_scored.empty:
        results["_crop_point_years"] = (
            crop_scored.groupby("unified_class")["Year"]
            .agg(lambda s: sorted(set(int(y) for y in s)))
            .to_dict()
        )
        results["_crop_point_years"] = {
            class_name(k): v for k, v in results["_crop_point_years"].items()
        }

    if is_fraction:
        results["l2a_forest_fraction"] = score_l2a_fraction(scored)

    return results


def make_lucas_frac_predict_fn(checkpoint, device, img_size, aux_channel_names,
                               *, num_classes, backbone_name):
    """Frac ``predict_fn`` returning (class_map, probs, RAW 4-frac array).

    Reuses ``inference_comparison.run_fraction_inference`` for the raw
    (4, cs, cs) crown-cover, plus the hard class map from
    ``validate_against_nfi.make_model_predict_fn`` so L2b (hard 28-class) is
    scored on the SAME member. L2a reads the raw fractions directly (not the
    collapsed proxy channels that ``make_fraction_predict_fn`` produces).
    """
    infcmp = _load_infcmp()
    model, epoch, miou, native = infcmp.load_model(
        checkpoint, device, backbone_name=backbone_name)
    if getattr(model, "frac_head", None) is None:
        raise ValueError(
            "checkpoint has no fraction head — cannot run L2a. Drop "
            "--fraction (score L2b only) or use a --enable-tradslag-head member."
        )
    print(f"  [load_model] epoch={epoch} ckpt_mIoU={miou} native_img={native} "
          f"(L2a fraction mode)")

    def predict_fn(tile_path):
        # hard class map (argmax of the 28-class head) — L2b on the same member
        probs, _rs, _ra = infcmp.run_inference(
            model, tile_path, device, img_size=img_size, return_probs=True,
            aux_channel_names=aux_channel_names)
        class_map = probs.argmax(0).astype(np.int64)
        # raw fractions (4, cs, cs) order tall/gran/trivial/adel
        fracs = infcmp.run_fraction_inference(
            model, tile_path, device, img_size=img_size,
            aux_channel_names=aux_channel_names)
        return class_map, probs, fracs

    return predict_fn


def _load_infcmp():
    spec = importlib.util.spec_from_file_location(
        "_infcmp", str(Path(__file__).resolve().parent / "inference_comparison.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def checkpoint_has_frac_head(checkpoint: str) -> bool:
    """Detect an L2a-capable member without building the model.

    Reads the checkpoint config's ``enable_tradslag_head`` flag (set by the
    trainer when the fraction head is present) and, as a fallback, checks for
    any ``frac_head.*`` weight in the state dict. Either signal → frac member.
    """
    import torch
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck.get("config", {}).get("enable_tradslag_head", False):
        return True
    sd = ck.get("model_state_dict", ck.get("state_dict", {}))
    return any("frac_head" in k for k in sd)


def _apply_crop_offset(index_df: pd.DataFrame, img_size: int) -> pd.DataFrame:
    """Remap (row,col) into the centre-crop run_inference applies; drop border.

    Mirrors ``validate_against_nfi.main``: read tile H from a sample npz,
    compute the crop offset, keep only in-crop points and shift coordinates.
    """
    sample_path = index_df["tile_path"].iloc[0]
    sample = np.load(sample_path, allow_pickle=True)
    key = "spectral" if "spectral" in sample else "image"
    tile_h = int(sample[key].shape[-1])
    off = van.crop_offset(tile_h, img_size)
    cs = min(img_size, tile_h)
    before = len(index_df)
    kept = index_df[
        (index_df["row"] >= off) & (index_df["row"] < off + cs)
        & (index_df["col"] >= off) & (index_df["col"] < off + cs)
    ].copy()
    kept["row"] -= off
    kept["col"] -= off
    print(f"crop offset={off} (tile {tile_h}→{cs}); kept {len(kept)}/{before} "
          f"points in-crop ({before - len(kept)} border-dropped)")
    return kept


def build_tile_years(index_df: pd.DataFrame) -> dict:
    """Read each tile's spectral year via nfi_colocate.tile_year for the assert."""
    from imint.training.nfi_colocate import tile_year
    years = {}
    for tile_name, grp in index_df.groupby("tile_name", sort=False):
        path = grp["tile_path"].iloc[0]
        try:
            d = np.load(path, allow_pickle=True)
            years[tile_name] = tile_year(d)
        except (FileNotFoundError, OSError, EOFError, ValueError):
            years[tile_name] = None
    return years


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lucas-index", required=True,
                    help="parquet from L1 (lucas_tile_index.parquet)")
    ap.add_argument("--data-dir", required=True,
                    help="tile root; tile_path = <data-dir>/{tile_name}.npz")
    ap.add_argument("--out", default="docs/data/lucas-validation.json")
    ap.add_argument("--dump-per-point", default=None,
                    help="parquet: one row per scored point (point_id, "
                         "unified_class, pred_class, split, year, source, "
                         "+ 4 forest-channel values)")
    ap.add_argument("--img-size", type=int, default=504,
                    help="inference crop (504 = 600M patch-14 on 512 tiles)")
    ap.add_argument("--num-classes", type=int, default=NUM_UNIFIED_CLASSES)
    ap.add_argument("--backbone-name", default=None)
    ap.add_argument("--min-support", type=int, default=20,
                    help="per-class row-support floor; below → not scored")
    ap.add_argument("--fraction", action="store_true",
                    help="also compute L2a forest-fraction metrics (requires a "
                         "checkpoint with a fraction head)")
    ap.add_argument("--auto-fraction", action="store_true",
                    help="auto-detect a fraction head and enable L2a if present")
    ap.add_argument("--enable-markfukt", action="store_true",
                    help="feed markfukt as the 11th aux channel")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    index_df = pd.read_parquet(args.lucas_index)
    # tile_path from data-dir + tile_name.npz (subdir-agnostic: try flat first).
    def _resolve(tile_name):
        flat = os.path.join(args.data_dir, f"{tile_name}.npz")
        return flat
    index_df["tile_path"] = index_df["tile_name"].map(_resolve)
    print(f"LUCAS index: {len(index_df):,} co-located points on "
          f"{index_df['tile_name'].nunique()} tiles")

    exists = index_df["tile_path"].map(os.path.exists)
    if not exists.all():
        gone = int((~exists).sum())
        print(f"dropping {gone} points on tiles not in {args.data_dir} "
              f"({index_df.loc[~exists, 'tile_name'].nunique()} tiles)")
        index_df = index_df[exists].copy()
    if index_df.empty:
        raise SystemExit(f"no LUCAS tiles found under {args.data_dir}")

    index_df = _apply_crop_offset(index_df, args.img_size)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    aux_names = None
    if args.enable_markfukt:
        from imint.training.unified_dataset import AUX_CHANNEL_NAMES
        aux_names = list(AUX_CHANNEL_NAMES) + ["markfukt"]
        print(f"  markfukt enabled → {len(aux_names)} aux channels")

    is_fraction = args.fraction
    if args.auto_fraction and not is_fraction:
        is_fraction = checkpoint_has_frac_head(args.checkpoint)
        print(f"  auto-fraction: frac head "
              f"{'DETECTED → L2a on' if is_fraction else 'absent → L2b only'}")

    if is_fraction:
        predict_fn = make_lucas_frac_predict_fn(
            args.checkpoint, device, args.img_size, aux_names,
            num_classes=args.num_classes, backbone_name=args.backbone_name)
    else:
        hard = van.make_model_predict_fn(
            args.checkpoint, device, args.img_size,
            aux_channel_names=aux_names, backbone_name=args.backbone_name)
        # wrap to the (class_map, probs) contract score_against_lucas expects
        predict_fn = hard

    # Crop year-match assertion: read each tile's spectral year.
    tile_years = build_tile_years(index_df)

    per_point = [] if args.dump_per_point else None
    results = score_against_lucas(
        index_df, predict_fn, num_classes=args.num_classes,
        min_support=args.min_support, is_fraction=is_fraction,
        tile_years=tile_years, per_point_sink=per_point)
    results["_meta"] = {
        "checkpoint": args.checkpoint, "img_size": args.img_size,
        "num_classes": args.num_classes, "min_support": args.min_support,
        "is_fraction_member": is_fraction,
        "points_scored": len(index_df),
        "n_tiles": int(index_df["tile_name"].nunique()),
    }
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {out}")

    if args.dump_per_point:
        pp = Path(args.dump_per_point)
        pp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_point).to_parquet(pp, index=False)
        print(f"wrote {pp} ({len(per_point)} points)")


if __name__ == "__main__":
    main()
