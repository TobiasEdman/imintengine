#!/usr/bin/env python3
"""Score the IM-023 amendment cells on the frozen NFI holdout (items 3 and 5).

Run only after Tobias's go and after all three cells' dumps exist. Reads
saved predictions; never constructs a model. Both analyses verify the
amendment, the frozen holdout, this process's source and runtime and every
dump, and sample the frozen NMD rasters for the common coverage of the
primary NFI scoring.

``primary`` (item 3): hard head, a predicted 28 folded to tallskog or
granskog by the fraction head's tall vs gran share; control and R4 under the
same rule. The paired report is ``score_nfi_holdout.paired_report``
unchanged, so the support, blocks, bootstrap and Holm family follow the
frozen protocol. Secondary: 28 as its own confusion column. Sensitivity
(Tobias 2026-11-07): the same scores without plots where contorta is at
least 30 % of the pine volume, descriptive only.

``mixed`` (item 5): blandskog and barrblandskog as own classes or derived
from fraction shares, with the prespecified conditions, their bounds and a
computed verdict per class.

    /opt/venvs/model/bin/python /opt/imintengine/scripts/score_nfi_amendment.py primary \
        --amendment-manifest M \
        --expected-amendment-sha256 S --dump-dir D \
        --nmd2023 /cephfs/nmd2023/NMD2023bas_v2_1.tif \
        --nmd2018 /cephfs/nmd/nmd2018bas_ogeneraliserad_v1_1.tif --out primary.json
    /opt/venvs/model/bin/python /opt/imintengine/scripts/score_nfi_amendment.py mixed ... --out mixed.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imint.eval.fieldtruth import restrict_to_frozen, verify_file_identity
from imint.eval.nfi_amendment import (
    BARRBLANDSKOG, BLANDSKOG, GRANSKOG, LOVSKOG, TALLSKOG, contorta_share, derive_barrblandskog,
    dump_name, fold_barrblandskog, load_amended_holdout, total_volume,
    verify_amendment_dump, verify_amendment_source,
)
from compare_nmd2023_nfi import sample_nmd_unified
from scripts.nfi_nmd_coverage import nmd_covered
from race_rigor_stats import block_ids
from score_nfi_holdout import json_metrics, paired_report, tract_block_frame, validate_baseline_paths
from validate_against_nfi import accuracy_suite, collapse_fractions_to_nfi_class, derive_nfi_forest_class

#: Truth v1 classes (confusion rows); predictions add 28 before the fold.
TRUTH_CLASSES = [0, TALLSKOG, GRANSKOG, LOVSKOG, BLANDSKOG]
SUITE_WITH_28 = TRUTH_CLASSES + [BARRBLANDSKOG]

#: Predictor -> (cell, head). a-d are the prespecified predictors; R4 is
#: scored as b and c (descriptive).
PREDICTORS = {"a": ("im023_main", "hard"), "b": ("im023_control", "hard"),
              "c": ("im023_control", "share"), "d": ("im023_main", "share"),
              "b_r4": ("r4", "hard"), "c_r4": ("r4", "share")}
#: Class -> (truth population, predictors scored). Hard heads without class
#: 28 cannot predict barrblandskog and are not scored for it.
EVALUATED = {"blandskog": ("volume", list(PREDICTORS)),
             "barrblandskog": ("volume", ["a", "c", "d", "c_r4"]),
             "tallskog": ("pure", list(PREDICTORS)),
             "granskog": ("pure", list(PREDICTORS))}
CLASS_IDS = {"blandskog": BLANDSKOG, "barrblandskog": BARRBLANDSKOG,
             "tallskog": TALLSKOG, "granskog": GRANSKOG}


# ── shared ───────────────────────────────────────────────────────────────────

def load_scored_cells(args) -> tuple[pd.DataFrame, dict, dict, dict, dict[str, pd.DataFrame], dict]:
    """Verified frozen holdout, manifests, runtime and one dump per cell, in holdout order."""
    holdout, amendment, base = load_amended_holdout(args.amendment_manifest,
                                                    args.expected_amendment_sha256)
    runtime_identity = verify_amendment_source(amendment, environment="model")
    if args.out.exists():
        raise ValueError("result already exists; choose a new output")
    cells, hashes = {}, {}
    for cell in sorted(amendment["cells"]):
        dump = verify_amendment_dump(args.dump_dir / dump_name(cell), cell, amendment)
        selected = restrict_to_frozen(dump, holdout)
        if not np.array_equal(selected["nfi_forest"].to_numpy(), holdout["nfi_forest"].to_numpy()):
            raise ValueError(f"{cell}: truth differs from frozen observations")
        cells[cell] = selected
        hashes[cell] = dump.attrs["authenticated_sha256"]
    return holdout, amendment, base, runtime_identity, cells, hashes


def sample_baselines(holdout: pd.DataFrame, base: dict, args) -> dict:
    """The frozen NMD rasters at the frozen locations, as the primary scoring samples them."""
    paths = {"NMD2023": args.nmd2023, **({"NMD2018": args.nmd2018} if args.nmd2018 else {})}
    validate_baseline_paths(base, paths)
    baselines = {name: sample_nmd_unified(str(path), holdout.Easting, holdout.Northing)
                 for name, path in paths.items()}
    for name in paths:
        verify_file_identity(base["baselines"][name])
    return baselines


def common_coverage(baselines: dict) -> np.ndarray:
    covered = np.logical_and.reduce([nmd_covered(raw) for _, raw in baselines.values()])
    if not covered.any():
        raise ValueError("no common NMD coverage on the frozen observations")
    return covered


def write_report(path: Path, report: dict) -> None:
    serialized = json.dumps(json_metrics(report), indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(serialized)
    print(f"wrote {path}")


def provenance(amendment: dict, runtime_identity: dict, hashes: dict) -> dict:
    return {"protocol_id": amendment["protocol_id"],
            "amendment_manifest_sha256": amendment["_manifest_sha256"],
            "base_manifest_sha256": amendment["base"]["manifest_sha256"],
            "execution_runtime": runtime_identity, "prediction_sha256": hashes}


# ── item 3 ───────────────────────────────────────────────────────────────────

def primary_predictions(cells: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    return {cell: fold_barrblandskog(frame["model_pred"].to_numpy(), frame["frac_tall"].to_numpy(),
                                     frame["frac_gran"].to_numpy())
            for cell, frame in cells.items()}


def class_28_columns(truth: np.ndarray, pred: np.ndarray) -> list[list[int]]:
    """Truth 0-4 (rows) x prediction 0-4 and 28 (columns), before the fold."""
    columns = np.where(np.isin(pred, SUITE_WITH_28), pred, 0)
    index = {c: i for i, c in enumerate(SUITE_WITH_28)}
    matrix = np.zeros((len(TRUTH_CLASSES), len(SUITE_WITH_28)), dtype=np.int64)
    np.add.at(matrix, (truth, [index[c] for c in columns]), 1)
    return matrix.tolist()


def contorta_sensitivity(holdout: pd.DataFrame, predictions: dict[str, np.ndarray],
                         baselines: dict, covered: np.ndarray, fraction: float) -> dict:
    """The primary scores without plots where contorta is >= ``fraction`` of the pine volume.

    Descriptive: the subset is no longer year-balanced, so the frozen paired
    report (which refuses that) does not apply, and no comparison is made.
    """
    kept = contorta_share(holdout) < fraction
    mask = covered & kept
    truth = holdout["nfi_forest"].replace(-1, 0).to_numpy(dtype=int)[mask]
    sources = dict(predictions, **{name: pred for name, (pred, _) in baselines.items()})
    scores = {}
    for name, pred in sources.items():
        pred = np.asarray(pred)[mask]
        collapsed = np.where(np.isin(pred, TRUTH_CLASSES[1:]), pred, 0).astype(int)
        correct = collapsed == truth
        k = len(TRUTH_CLASSES)
        scores[name] = dict(accuracy_suite(truth, pred), n=int(mask.sum()),
                            correct=int(correct.sum()),
                            overall_exact=float(correct.mean()) if mask.any() else None,
                            confusion_matrix=np.bincount(truth * k + collapsed, minlength=k * k)
                            .reshape(k, k).tolist())
    return {"contorta_fraction_of_pine_volume": fraction,
            "excluded_contorta_observations": int((covered & ~kept).sum()),
            "compared_observations": int(mask.sum()),
            "interpretation": "Descriptive sensitivity analysis; no comparison and no verdict.",
            "scores": scores}


def primary_report(holdout: pd.DataFrame, cells: dict[str, pd.DataFrame],
                   baselines: dict, protocol: dict, amendment_protocol: dict) -> dict:
    predictions = primary_predictions(cells)
    report = paired_report(holdout, predictions, baselines, protocol)
    covered = common_coverage(baselines)
    truth = holdout["nfi_forest"].replace(-1, 0).to_numpy(dtype=int)[covered]
    secondary, folded = {}, {}
    for cell, frame in cells.items():
        raw = frame["model_pred"].to_numpy(dtype=int)[covered]
        secondary[cell] = class_28_columns(truth, raw)
        moved = raw == BARRBLANDSKOG
        folded[cell] = {"predicted_28": int(moved.sum()),
                        "to_tallskog": int((predictions[cell][covered][moved] == TALLSKOG).sum()),
                        "to_granskog": int((predictions[cell][covered][moved] == GRANSKOG).sum())}
    return {
        "schema": "nfi-amendment-primary-v1",
        "primary_rule": "hard head; predicted 28 -> tallskog if tall >= gran share, else granskog",
        "primary": report,
        "secondary_class_28": {
            "axes": {"rows": "NFI truth v1", "columns": "hard-head prediction before the fold",
                     "row_order": [0, 1, 2, 3, 4], "column_order": SUITE_WITH_28},
            "confusion_matrix": secondary},
        "fold_counts": folded,
        "sensitivity_without_contorta_plots": contorta_sensitivity(
            holdout, predictions, baselines, covered,
            amendment_protocol["contorta_sensitivity_fraction"]),
    }


# ── item 5 ───────────────────────────────────────────────────────────────────

def share_rule(fractions: pd.DataFrame, protocol: dict) -> np.ndarray:
    """``collapse_fractions_to_nfi_class``, then one level down for conifer:
    tall or gran share >= species fraction of the conifer share, else 28.
    Blandskog and the forest floor are exactly the collapse rule's."""
    labels = []
    for tall, gran, trivial, adel in fractions[["frac_tall", "frac_gran", "frac_trivial",
                                                 "frac_adel"]].itertuples(index=False):
        label = collapse_fractions_to_nfi_class(
            tall, gran, trivial, adel, dominant_frac=protocol["share_rule_dominant_fraction"],
            forest_floor=protocol["forest_floor"])
        if label in (TALLSKOG, GRANSKOG):
            conifer = tall + gran
            species = protocol["share_rule_species_fraction"]
            label = (TALLSKOG if tall / conifer >= species
                     else GRANSKOG if gran / conifer >= species else BARRBLANDSKOG)
        labels.append(label)
    return np.asarray(labels, dtype=np.int64)


def predictor_labels(cells: dict[str, pd.DataFrame], protocol: dict) -> dict[str, np.ndarray]:
    return {name: (cells[cell]["model_pred"].to_numpy(dtype=np.int64) if head == "hard"
                   else share_rule(cells[cell], protocol))
            for name, (cell, head) in PREDICTORS.items()}


def f1_from_counts(tp, fp, fn) -> np.ndarray:
    tp, fp, fn = (np.asarray(v, dtype=float) for v in (tp, fp, fn))
    denominator = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, 2 * tp / denominator, np.nan)


def class_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict:
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    f1 = float(f1_from_counts(tp, fp, fn))
    return {"tp": tp, "fp": fp, "fn": fn, "support": tp + fn, "predicted": tp + fp,
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "f1": None if np.isnan(f1) else f1}


def block_f1_draws(truth: np.ndarray, predicted: dict[str, np.ndarray],
                   blocks: np.ndarray, sample: np.ndarray) -> dict[str, np.ndarray]:
    """F1 per bootstrap draw: per-block TP/FP/FN summed over resampled blocks."""
    _, codes = np.unique(blocks, return_inverse=True)
    n_blocks = sample.shape[1]

    def per_block(mask):
        return np.bincount(codes, weights=mask, minlength=n_blocks)[sample].sum(axis=1)
    return {name: f1_from_counts(per_block(truth & p), per_block(~truth & p),
                                 per_block(truth & ~p))
            for name, p in predicted.items()}


def compare(draws: dict, metrics: dict, x: str, y: str, quantiles: list[float],
            *, bound: str | None = None, sesoi: float | None = None) -> dict:
    """Paired ΔF1 = F1(x) - F1(y); with ``bound``, whether it clears -SESOI."""
    diff = draws[x] - draws[y]
    undefined = int(np.isnan(diff).sum())
    lower, upper = (None, None) if undefined else (float(v) for v in np.quantile(diff, quantiles))
    fx, fy = metrics[x]["f1"], metrics[y]["f1"]
    result = {"delta_f1": None if fx is None or fy is None else fx - fy,
              "comparison": f"F1({x}) - F1({y})", "interval": [lower, upper],
              "interval_quantiles": quantiles, "undefined_draws": undefined}
    if bound is not None:
        value = lower if bound == "lower" else upper
        holds = None if value is None else bool(value > -sesoi if bound == "lower" else value < -sesoi)
        result.update(bound=bound, bound_value=value, threshold=-sesoi, holds=holds)
    return result


def _kleene_all(values):
    return False if False in values else None if None in values else True


def _kleene_any(values):
    return True if True in values else None if None in values else False


def evaluate(truth: dict[str, np.ndarray], labels: dict[str, np.ndarray], blocks: np.ndarray,
             populations: dict[str, np.ndarray], protocol: dict) -> dict:
    """Metrics, conditions and descriptive comparisons on one population pair."""
    quantiles = protocol["condition_bound_quantiles"]
    samples = {}
    for name, mask in populations.items():
        n_blocks = len(np.unique(blocks[mask]))
        rng = np.random.default_rng(protocol["bootstrap_seed"])
        samples[name] = rng.integers(0, n_blocks, size=(protocol["bootstrap_samples"], n_blocks))
    metrics, draws = {}, {}
    for cls, (population, names) in EVALUATED.items():
        mask = populations[population]
        t = truth[cls][mask]
        predicted = {n: labels[n][mask] == CLASS_IDS[cls] for n in names}
        metrics[cls] = {n: class_metrics(t, p) for n, p in predicted.items()}
        draws[cls] = block_f1_draws(t, predicted, blocks[mask], samples[population])
    sesoi = protocol["sesoi"]

    def condition(cls, x, y, bound):
        return compare(draws[cls], metrics[cls], x, y, quantiles, bound=bound, sesoi=sesoi)
    pure = {cls: condition(cls, "a", "b", "upper") for cls in ("tallskog", "granskog")}
    conditions = {
        "barrblandskog": {
            "condition_1": condition("barrblandskog", "c", "a", "lower"),
            "condition_2": dict(pure, holds=_kleene_any([pure[c]["holds"] for c in pure])),
        },
        "blandskog": {"condition_1": condition("blandskog", "c", "b", "lower")},
    }
    descriptive = {cls: compare(draws[cls], metrics[cls], "d", "a", quantiles)
                   for cls in ("barrblandskog", "blandskog")}
    return {"observations": {name: int(mask.sum()) for name, mask in populations.items()},
            "metrics": metrics, "conditions": conditions, "descriptive_d_minus_a": descriptive}


def verdicts(conditions: dict) -> dict:
    """Step 3's two readings. A condition whose bound is undefined (``holds``
    None, kept visible in the conditions) is not met: "annars behålls klassen"
    (Tobias 2026-11-07)."""
    barr = conditions["barrblandskog"]
    both = _kleene_all([barr["condition_1"]["holds"], barr["condition_2"]["holds"]])
    bland = conditions["blandskog"]["condition_1"]["holds"]
    return {"barrblandskog": "derive_from_shares" if both is True else "keep_class",
            "blandskog": ("derivable_from_shares_without_loss" if bland is True
                          else "not_established")}


def mixed_report(holdout: pd.DataFrame, cells: dict[str, pd.DataFrame], covered: np.ndarray,
                 protocol: dict) -> dict:
    """Item 5 on the primary scoring's support: every covered frozen plot-year with volume."""
    recomputed = np.array([derive_nfi_forest_class(row, dominant_frac=protocol["truth_dominant_fraction"])
                           or 0 for _, row in holdout.iterrows()])
    if not np.array_equal(recomputed, holdout["nfi_forest"].to_numpy(dtype=int)):
        raise ValueError("species volumes do not reproduce the frozen truth v1")
    blocks = block_ids(tract_block_frame(holdout), protocol["block_km"])[covered]
    selected = holdout.loc[covered].reset_index(drop=True)
    labels = {name: values[covered] for name, values in predictor_labels(cells, protocol).items()}
    volume = total_volume(selected) > 0
    barr = derive_barrblandskog(selected, species_fraction=protocol["barrblandskog_species_fraction"])
    pure = volume & ~barr
    kept = contorta_share(selected) < protocol["contorta_sensitivity_fraction"]
    v1 = selected["nfi_forest"].to_numpy(dtype=int)
    truth = {"blandskog": v1 == BLANDSKOG, "barrblandskog": barr,
             "tallskog": v1 == TALLSKOG, "granskog": v1 == GRANSKOG}
    primary = evaluate(truth, labels, blocks, {"volume": volume, "pure": pure}, protocol)
    sensitivity = evaluate(truth, labels, blocks,
                           {"volume": volume & kept, "pure": pure & kept}, protocol)
    return {
        "schema": "nfi-amendment-mixed-forest-v1",
        "support": {"covered_frozen_observations": int(covered.sum()),
                    "excluded_without_volume": int((~volume).sum()),
                    "derived_barrblandskog": int(barr.sum()),
                    "excluded_in_sensitivity_contorta": int((volume & ~kept).sum())},
        "predictors": {name: {"cell": cell, "head": head} for name, (cell, head) in PREDICTORS.items()},
        "primary": primary,
        "sensitivity_without_contorta_plots": sensitivity,
        "verdict": verdicts(primary["conditions"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="analysis", required=True)
    for name in ("primary", "mixed"):
        p = sub.add_parser(name)
        p.add_argument("--amendment-manifest", type=Path, required=True)
        p.add_argument("--expected-amendment-sha256", required=True,
                       help="amendment digest approved at go/no-go")
        p.add_argument("--dump-dir", type=Path, required=True)
        p.add_argument("--nmd2023", type=Path, required=True)
        p.add_argument("--nmd2018", type=Path)
        p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    holdout, amendment, base, runtime_identity, cells, hashes = load_scored_cells(args)
    baselines = sample_baselines(holdout, base, args)
    if args.analysis == "primary":
        report = primary_report(holdout, cells, baselines, base["protocol"], amendment["protocol"])
    else:
        report = mixed_report(holdout, cells, common_coverage(baselines), amendment["protocol"])
    report.update(provenance(amendment, runtime_identity, hashes))
    write_report(args.out, report)


if __name__ == "__main__":
    main()
