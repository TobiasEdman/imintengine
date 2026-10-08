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
    verify_file_identity, verify_prediction_dump, verify_evaluation_source,
)
from compare_nmd2023_nfi import sample_nmd_unified
from validate_against_nfi import accuracy_suite
from race_rigor_stats import block_ids, compare_pair, holm


def block_signflip_pvalue(
    difference: np.ndarray, blocks: np.ndarray, rng: np.random.Generator,
    samples: int,
) -> float:
    """Paired label swaps operate on whole spatial blocks, never on plots.

    The null assumes exchangeability of model labels within independent
    blocks. Enumerate small block sets; otherwise use Monte Carlo with the
    plus-one correction. Zero-difference blocks do not affect the statistic.
    """
    sums = np.array([difference[blocks == b].sum() for b in np.unique(blocks)])
    sums = sums[sums != 0]
    if not len(sums):
        return 1.0
    observed = abs(sums.sum())
    if len(sums) <= 16:
        draws = np.arange(2 ** len(sums), dtype=np.uint64)[:, None]
        signs = 2 * ((draws >> np.arange(len(sums), dtype=np.uint64)) & 1).astype(int) - 1
        stats = np.abs(signs @ sums)
        return float((stats >= observed).mean())
    signs = rng.choice([-1, 1], size=(samples, len(sums)))
    exceed = int((np.abs(signs @ sums) >= observed).sum())
    return (exceed + 1) / (samples + 1)


def paired_report(
    holdout: pd.DataFrame, predictions: dict[str, np.ndarray],
    baselines: dict[str, tuple[np.ndarray, np.ndarray]], protocol: dict,
) -> dict:
    """One denominator across every model and baseline, after coverage only."""
    truth = holdout["nfi_forest"].replace(-1, 0).to_numpy(dtype=int)
    covered = np.ones(len(holdout), dtype=bool)
    for _, raw in baselines.values():
        covered &= np.asarray(raw) != 0
    if not covered.any():
        raise ValueError("no common NMD coverage on the frozen observations")
    selected = holdout.loc[covered].copy()
    truth = truth[covered]
    sources = dict(predictions)
    sources.update({name: pred for name, (pred, _) in baselines.items()})
    scores, frames = {}, {}
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
                            overall_exact=float(correct.mean()))
        frames[name] = selected.assign(correct=correct.astype(int)).set_index(NFI_KEY)
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
            difference, block_ids(selected, protocol["block_km"]),
            rng, protocol["bootstrap_samples"])
        pairs[f"{a} vs {b}"] = dict(pair, a=a, b=b)
    n_blocks = int(len(np.unique(block_ids(selected, protocol["block_km"]))))
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
        "frozen_observations": len(holdout), "compared_observations": len(selected),
        "excluded_for_nmd_coverage": int((~covered).sum()),
        "spatial_blocks": n_blocks, "scores": scores, "pairs": pairs,
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
    args = ap.parse_args()
    holdout, manifest = load_frozen_holdout(args.holdout_manifest)
    verify_evaluation_source(manifest)
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
        dump_hashes[cell] = sha256_file(path)
    baseline_paths = {"NMD2023": args.nmd2023}
    if args.nmd2018:
        baseline_paths["NMD2018"] = args.nmd2018
    validate_baseline_paths(manifest, baseline_paths)
    baselines = {}
    for name, path in baseline_paths.items():
        baselines[name] = sample_nmd_unified(str(path), holdout.Easting, holdout.Northing)
        verify_file_identity(manifest["baselines"][name])
    result = paired_report(holdout, predictions, baselines, manifest["protocol"])
    result["manifest_sha256"] = manifest["_manifest_sha256"]
    result["prediction_sha256"] = dump_hashes
    serialized = json.dumps(json_metrics(result), indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(serialized)
    print(f"wrote {args.out}; compared {result['compared_observations']} plot-years")


if __name__ == "__main__":
    main()
