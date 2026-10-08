#!/usr/bin/env python3
"""Score all frozen NFI model dumps and NMD on one observation set.

Run only after the user's go/no-go and completion of all frozen model cells.
This reads saved predictions; it never constructs or fits a model.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imint.eval.fieldtruth import (
    NFI_KEY, load_frozen_holdout, restrict_to_frozen, sha256_file,
    verify_file_identity, verify_prediction_dump, verify_evaluation_source, validate_year_balance,
)
from compare_nmd2023_nfi import sample_nmd_unified
from validate_against_nfi import accuracy_suite
from race_rigor_stats import block_ids, compare_pair, holm


def block_signflip_pvalue(difference: np.ndarray, blocks: np.ndarray) -> float:
    """Exact paired label swaps over independent spatial blocks.

    Correctness differences are integers, so dynamic programming builds the
    complete distribution of signed block sums in O(blocks * observations).
    No Monte Carlo p-value floor or simulation noise enters Holm correction.
    """
    difference = np.asarray(difference)
    blocks = np.asarray(blocks)
    if (difference.shape != np.asarray(blocks).shape
            or not np.isin(difference, [-1, 0, 1]).all()):
        raise ValueError("block test requires paired correctness differences")
    sums = np.array([difference[blocks == b].sum() for b in np.unique(blocks)], dtype=int)
    weights = np.abs(sums[sums != 0])
    distribution = np.ones(1)
    for weight in weights:
        updated = np.zeros(len(distribution) + weight)
        updated[:len(distribution)] += distribution * 0.5
        updated[weight:] += distribution * 0.5
        distribution = updated
    signed = 2 * np.arange(len(distribution)) - int(weights.sum())
    return min(1.0, float(distribution[np.abs(signed) >= abs(sums.sum())].sum()))


def tract_block_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep each tract-year in one 50 km block, without altering NMD locations."""
    result = frame.copy()
    coordinates = ["Easting", "Northing"]
    if not np.isfinite(result[coordinates].to_numpy(dtype=float)).all():
        raise ValueError("spatial blocks require finite observation coordinates")
    result[coordinates] = result.groupby(["TractID", "Year"])[coordinates].transform("mean")
    return result


def paired_report(
    holdout: pd.DataFrame, predictions: dict[str, np.ndarray],
    baselines: dict[str, tuple[np.ndarray, np.ndarray]], protocol: dict,
) -> dict:
    """One denominator across every model and baseline, after coverage only."""
    validate_year_balance(holdout, protocol)
    truth = holdout["nfi_forest"].replace(-1, 0).to_numpy(dtype=int)
    covered = np.ones(len(holdout), dtype=bool)
    for _, raw in baselines.values():
        covered &= np.asarray(raw) != 0
    if not covered.any():
        raise ValueError("no common NMD coverage on the frozen observations")
    selected = holdout.loc[covered].copy()
    validate_year_balance(selected, protocol)
    statistical = tract_block_frame(holdout).loc[covered].copy()
    truth = truth[covered]
    sources = dict(predictions)
    sources.update({name: pred for name, (pred, _) in baselines.items()})
    scores, frames, strata = {}, {}, {}
    for name, pred in sources.items():
        pred = np.asarray(pred)
        if pred.shape != (len(holdout),):
            raise ValueError(f"{name}: predictions do not match frozen support")
        if not np.isfinite(pred).all() or (pred % 1 != 0).any():
            raise ValueError(f"{name}: invalid class predictions")
        collapsed = np.where(np.isin(pred[covered], [1, 2, 3, 4]), pred[covered], 0)
        correct = collapsed == truth
        scores[name] = dict(accuracy_suite(truth, pred[covered]),
                            n=len(truth), correct=int(correct.sum()),
                            overall_exact=float(correct.mean()),
                            confusion_matrix=np.bincount(
                                truth * 5 + collapsed.astype(int), minlength=25).reshape(5, 5).tolist())
        frames[name] = statistical.assign(correct=correct.astype(int)).set_index(NFI_KEY)
        if "tile_role" in selected:
            for role in sorted(selected["tile_role"].unique()):
                mask = selected["tile_role"].to_numpy() == role
                strata.setdefault(str(role), {})[name] = dict(
                    accuracy_suite(truth[mask], pred[covered][mask]),
                    n=int(mask.sum()), correct=int(correct[mask].sum()),
                    overall_exact=float(correct[mask].mean()),
                    confusion_matrix=np.bincount(
                        truth[mask] * 5 + collapsed[mask].astype(int), minlength=25).reshape(5, 5).tolist())
    # All model-model and model-NMD comparisons form one prespecified family;
    # the winner's comparisons are never selected only after seeing scores.
    pairs = {}
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    for a, b in itertools.combinations(sorted(sources), 2):
        if a in baselines and b in baselines:
            continue
        pair = compare_pair(
            frames[a], frames[b], sesoi=protocol["sesoi"],
            block_km=protocol["block_km"], n_boot=protocol["bootstrap_samples"], rng=rng)
        difference = frames[a]["correct"].to_numpy() - frames[b]["correct"].to_numpy()
        pair["block_signflip_p"] = block_signflip_pvalue(
            difference, block_ids(statistical, protocol["block_km"]))
        pairs[f"{a} vs {b}"] = dict(pair, a=a, b=b)
    n_blocks = int(len(np.unique(block_ids(statistical, protocol["block_km"]))))
    adjusted = holm([p["block_signflip_p"] for p in pairs.values()])
    for pair, p_adj in zip(pairs.values(), adjusted):
        pair["block_signflip_holm"] = p_adj
        lo, hi = pair["diff_ci95"]
        pair["difference_supported"] = bool(
            n_blocks > 1 and p_adj < 0.05 and (lo > 0 or hi < 0))
        pair["equivalent_marginal"] = bool(
            n_blocks > 1 and pair["tost_equivalent"])
    best_score = max(scores[m]["overall_exact"] for m in predictions)
    best = sorted(m for m in predictions if scores[m]["overall_exact"] == best_score)
    return {
        "schema": "nfi-paired-model-nmd-v1",
        "primary_population": protocol.get("primary_population", "all"),
        "year_selection": protocol.get("year_selection"),
        "compared_year_support": {str(int(y)): int(n) for y, n in selected["Year"].value_counts().sort_index().items()},
        "confusion_matrix_axes": {"rows": "NFI truth", "columns": "prediction", "class_order": [0,1,2,3,4]},
        "tile_role_scores": strata,
        "tile_role_interpretation": "Descriptive strata; confirmatory inference applies only to the prespecified primary population.",
        "frozen_observations": len(holdout), "compared_observations": len(selected),
        "excluded_for_nmd_coverage": int((~covered).sum()),
        "block_assignment": "50km_grid_at_frozen_tract_year_centroid",
        "spatial_blocks": n_blocks, "scores": scores, "pairs": pairs,
        "block_signflip_method": "exact_integer_sum_distribution",
        "highest_point_estimate": best,
        "interpretation": (
            "Highest point estimate is descriptive. Difference support requires "
            "a nonzero spatial-block CI and Holm-adjusted block-signflip p<0.05. "
            "Pair CIs and equivalence checks are marginal, not simultaneous; "
            "absence of a supported difference does not establish a tie."
        ),
    }


def json_metrics(value):
    """Undefined statistics are JSON null, not fabricated zero or NaN."""
    if isinstance(value, dict):
        return {k: json_metrics(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_metrics(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def validate_baseline_paths(manifest: dict, paths: dict[str, Path]) -> None:
    if set(paths) != set(manifest["baselines"]):
        raise ValueError("baseline roster differs from the frozen comparison")
    for name, path in paths.items():
        expected = manifest["baselines"][name]
        if str(path.resolve()) != expected["path"]:
            raise ValueError(f"{name} path differs from its frozen role")
        verify_file_identity(expected)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout-manifest", type=Path, required=True)
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--nmd2023", type=Path, required=True)
    ap.add_argument("--nmd2018", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected-manifest-sha256", help="SHA256 approved at go/no-go")
    args = ap.parse_args()
    holdout, manifest = load_frozen_holdout(args.holdout_manifest, args.expected_manifest_sha256)
    runtime_identity = verify_evaluation_source(manifest)
    if args.out.exists():
        raise ValueError("result already exists; choose a new output")
    predictions, dump_hashes = {}, {}
    for cell in sorted(manifest["cells"]):
        path = args.dump_dir / f"nfi-per-plot-{cell}.parquet"
        dump = verify_prediction_dump(path, args.holdout_manifest, cell, manifest=manifest)
        selected = restrict_to_frozen(dump, holdout)
        if not np.array_equal(selected["nfi_forest"].replace(-1, 0),
                              holdout["nfi_forest"].replace(-1, 0)):
            raise ValueError(f"{cell}: truth differs from frozen observations")
        predictions[cell] = selected["model_pred"].to_numpy()
        dump_hashes[cell] = dump.attrs["authenticated_sha256"]
    baseline_paths = {"NMD2023": args.nmd2023}
    if args.nmd2018:
        baseline_paths["NMD2018"] = args.nmd2018
    validate_baseline_paths(manifest, baseline_paths)
    baselines = {}
    for name, path in baseline_paths.items():
        baselines[name] = sample_nmd_unified(str(path), holdout.Easting, holdout.Northing)
        verify_file_identity(manifest["baselines"][name])
    result = paired_report(holdout, predictions, baselines, manifest["protocol"])
    result["execution_runtime"] = runtime_identity
    result["manifest_sha256"] = manifest["_manifest_sha256"]
    result["prediction_sha256"] = dump_hashes
    serialized = json.dumps(json_metrics(result), indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(serialized)
    print(f"wrote {args.out}; compared {result['compared_observations']} plot-years")


if __name__ == "__main__":
    main()
