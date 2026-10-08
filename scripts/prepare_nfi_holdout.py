#!/usr/bin/env python3
"""Prepare a reproducible NFI holdout on CPU, only after Tessera promotion.

Run in the data environment: raw observations and the resulting parquet stay
there. This command performs no inference, NMD sampling or model fitting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imint.eval.fieldtruth import (
    NFI_KEY, exclude_training_observations, observation_keys,
    require_columns, resolve_observation_year, sha256_file,
)
from build_pinned_plot_set import npz_key_names, npz_version_ok
from gen_ladder_manifests import DISTILL
from scripts.crop_distill_provenance import (
    verify_runtime, snapshot_tree, tree_payload_sha256,
)


def check_promotion(job: dict, report: dict, cohort_count: int) -> None:
    """Counter reports omit zero-valued FLAG_ON_EMPTY; require full coverage."""
    status = job.get("status", {})
    complete = any(c.get("type") == "Complete" and c.get("status") == "True"
                   for c in status.get("conditions", []))
    if (job.get("metadata", {}).get("name") != "tessera-promote-v2"
            or not complete or status.get("active", 0) or status.get("failed", 0)):
        raise ValueError("tessera-promote-v2 must be terminal and successful")
    verify = report.get("verify", {})
    if (report.get("source") != "geotessera-0.10.2"
            or not cohort_count
            or verify.get("has_v1") != cohort_count
            or verify.get("stamped") != cohort_count
            or verify.get("FLAG_ON_EMPTY", 0) != 0
            or verify.get("has_v2_left", 0) != 0
            or report.get("states", {}).get("unreadable", 0) != 0
            or report.get("states", {}).get("no_v2", 0) != 0):
        raise ValueError("promotion report lacks complete, clean v2 coverage")


def teacher_training_set(
    features: pd.DataFrame, split: dict, source_index: pd.DataFrame,
    staging_names: set[str],
) -> pd.DataFrame:
    train = set(map(str, split["train_tiles"]))
    test = set(map(str, split["test_tiles"]))
    if train & test:
        raise ValueError("teacher train and test tiles overlap")
    if train & staging_names:
        raise ValueError("evaluation-only campaign tiles appear in training")
    resolved = resolve_observation_year(features, source_index)
    names = resolved["tile_name"].astype(str)
    if set(names) != train | test:
        raise ValueError("teacher feature tiles do not match the recorded split")
    selected = resolved[names.isin(train)]
    if (len(selected) != split["n_train_plots"]
            or int(names.isin(test).sum()) != split["n_test_plots"]):
        raise ValueError("teacher feature row counts do not match training provenance")
    return selected[NFI_KEY].drop_duplicates().reset_index(drop=True)


def tile_readiness(path: Path) -> tuple[dict, str | None]:
    """Read data prerequisites before inference; all decisions ignore scores."""
    keys = npz_key_names(path)
    if keys is None:
        raise ValueError(f"unreadable tile: {path}")
    required = {"spectral", "tessera", "s1_vv_vh", "b08", "rededge"}
    if missing := required - keys:
        return {}, "missing_keys:" + ",".join(sorted(missing))
    if not npz_version_ok(path, ("s1_vv_vh",)):
        return {}, "s1_version"
    with np.load(path, allow_pickle=False) as data:
        tessera = data["tessera"]
        nonempty = (tessera.ndim == 3 and tessera.size > 0
                    and np.isfinite(tessera).all() and np.any(tessera))
        flagged = int(data.get("has_tessera", 0)) > 0
        if flagged and not nonempty:
            raise ValueError(f"positive has_tessera on empty/invalid array: {path}")
        if str(data.get("tessera_source", "")) != "geotessera-0.10.2":
            raise ValueError(f"unpromoted Tessera source: {path}")
        if not flagged or not nonempty:
            return {}, "tessera_gap"
        sar = data["s1_vv_vh"]
        if (not int(data.get("has_s1", 0)) or sar.ndim != 3 or sar.shape[0] != 2
                or not np.isfinite(sar).all() or not np.any(sar)):
            return {}, "s1_gap"
        spectral = data["spectral"]
        if spectral.ndim < 3 or not np.isfinite(spectral).all():
            raise ValueError(f"invalid spectral array: {path}")
        year = data.get("year", data.get("lpis_year"))
        if year is None:
            raise ValueError(f"tile has no explicit spectral year: {path}")
        return {"height": int(spectral.shape[-2]),
                "width": int(spectral.shape[-1]), "year": int(year)}, None


def select_holdout(
    index: pd.DataFrame, training: pd.DataFrame, tile_meta: dict,
) -> tuple[pd.DataFrame, dict]:
    require_columns(index, NFI_KEY + ["tile_name", "row", "col"])
    if index.duplicated(NFI_KEY + ["tile_name"]).any():
        raise ValueError("duplicate plot-year/tile in the NFI index")
    eligible = exclude_training_observations(index, training)
    counts = {"index_rows": len(index),
              "index_observations": len(observation_keys(index).unique()),
              "training_overlap_rows": len(index) - len(eligible)}
    min_crop = min(c["img_size"] for c in DISTILL.values())
    candidates = []
    for name, group in eligible.groupby("tile_name", sort=True):
        meta = tile_meta.get(str(name))
        if meta is None:
            continue
        h, w = meta["height"], meta["width"]
        size = min(min_crop, h, w)
        y0, x0 = (h - size) // 2, (w - size) // 2
        if not (group["Year"] == meta["year"]).all():
            raise ValueError(f"NFI observation year differs from spectral year: {name}")
        keep = (group["row"].between(y0, y0 + size - 1)
                & group["col"].between(x0, x0 + size - 1))
        candidates.append(group.loc[keep])
    if not candidates:
        raise ValueError("no eligible observations with common input support")
    paired = pd.concat(candidates).sort_values(NFI_KEY + ["tile_name"])
    holdout = paired.drop_duplicates(NFI_KEY).reset_index(drop=True)
    if holdout.empty:
        raise ValueError("holdout is empty")
    assert not observation_keys(holdout).isin(observation_keys(training)).any()
    counts.update(common_support_rows=len(paired),
                  observations=len(holdout), duplicate_rows_removed=len(paired)-len(holdout),
                  training_observation_overlap=0)
    return holdout, counts


def file_identity(path: Path) -> dict:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"input changed while hashing: {path}")
    return {"path": str(path.resolve()), "bytes": after.st_size, "sha256": digest}


def preparation_runtime(
    root: Path, runtime_manifest: Path, source_git_sha: str, image_ref: str,
) -> dict:
    """Authenticate the sealed source and CPU interpreter, without Git."""
    runtime = verify_runtime(runtime_manifest, source_git_sha=source_git_sha,
                             image_ref=image_ref)
    if tree_payload_sha256(snapshot_tree(root)) != runtime["source"]["payload_sha256"]:
        raise ValueError("preparation is not running from the verified source tree")
    expected_python = runtime["environments"]["scoring"]["python"]["path"]
    # Do not resolve venv symlinks: both environments can share a base binary.
    if Path(sys.executable).absolute() != Path(expected_python).absolute():
        raise ValueError("preparation must use the verified scoring interpreter")
    return runtime


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("plot-index", "teacher-index", "teacher-root", "checkpoint-root",
                 "cohort-dir", "staging-dir", "promotion-job-json",
                 "promotion-report", "nmd2023", "out-dir"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--nmd2018", type=Path)
    ap.add_argument("--runtime-image", required=True,
                    help="reviewed inference image pinned with @sha256")
    ap.add_argument("--runtime-manifest", type=Path, required=True,
                    help="sealed image provenance, normally /opt/provenance/runtime.json")
    ap.add_argument("--source-git-sha", required=True, help="reviewed build source SHA")
    args = ap.parse_args()
    if args.out_dir.exists():
        raise ValueError("freeze output already exists; never overwrite it")
    root = Path(__file__).resolve().parents[1]
    runtime = preparation_runtime(root, args.runtime_manifest,
                                  args.source_git_sha, args.runtime_image)
    code_sha = runtime["source"]["git_sha"]
    cohort = {p.stem: p for p in args.cohort_dir.glob("*.npz")}
    staging = {p.stem: p for p in args.staging_dir.glob("*.npz")}
    if len(staging) != 475 or set(staging) & set(cohort):
        raise ValueError("campaign must contain 475 evaluation-only, disjoint tiles")
    input_stats = {}
    def capture(path):
        stat = path.stat()
        input_stats[path] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    for path in [args.plot_index, args.teacher_index, args.promotion_job_json,
                 args.promotion_report, args.nmd2023] + ([args.nmd2018] if args.nmd2018 else []):
        capture(path)
    check_promotion(json.loads(args.promotion_job_json.read_text()),
                    json.loads(args.promotion_report.read_text()), len(cohort))
    index = pd.read_parquet(args.plot_index)
    source = pd.read_parquet(args.teacher_index)
    paths = {**cohort, **staging}
    if not set(index["tile_name"].astype(str)) <= set(paths):
        raise ValueError("indexed tile absent from the two declared data roots")
    index["tile_path"] = index["tile_name"].map(lambda n: str(paths[str(n)].resolve()))
    inputs = [args.plot_index, args.teacher_index, args.promotion_job_json,
              args.promotion_report, args.nmd2023]
    if args.nmd2018:
        inputs.append(args.nmd2018)
    trained = []
    for model in DISTILL:
        features = args.teacher_root / f"{model}_r2_plot_features.parquet"
        split = args.teacher_root / f"{model}_r2_split.json"
        capture(features)
        capture(split)
        capture(args.teacher_root / f"{model}_r2_head.npz")
        trained.append(teacher_training_set(pd.read_parquet(features),
                       json.loads(split.read_text()), source, set(staging)))
        inputs.extend([features, split, args.teacher_root / f"{model}_r2_head.npz"])
    training = pd.concat(trained).drop_duplicates(NFI_KEY).sort_values(NFI_KEY)
    metadata, excluded = {}, {}
    # Inspect all campaign tiles, including tiles carrying no indexed plots.
    names = set(staging) | set(index["tile_name"].astype(str))
    for name in sorted(names):
        capture(paths[name])
        meta, reason = tile_readiness(paths[name])
        if reason:
            excluded[name] = reason
        else:
            metadata[name] = meta
    holdout, counts = select_holdout(index, training, metadata)
    from validate_against_nfi import derive_nfi_forest_class
    holdout["nfi_forest"] = [
        derive_nfi_forest_class(row, dominant_frac=0.7) or 0
        for _, row in holdout.iterrows()
    ]
    # A plot-year cannot have different field measurements across its tiles.
    truth = index.assign(nfi_forest=[
        derive_nfi_forest_class(row, dominant_frac=0.7) or 0
        for _, row in index.iterrows()])
    if (truth.groupby(NFI_KEY)["nfi_forest"].nunique() > 1).any():
        raise ValueError("conflicting NFI truth for a plot-year")
    cells = {
        f"{m}_r{r}": {
            "checkpoint": file_identity(args.checkpoint_root / f"{m}_r{r}" / "best_model.pt"),
            "backbone": cfg["backbone"], "img_size": cfg["img_size"],
            "num_classes": 23 if r == 1 else 28, "head": "class",
        }
        for m, cfg in DISTILL.items() for r in range(1, 5)
    }
    class_support = holdout["nfi_forest"].value_counts().sort_index().to_dict()
    counts["class_support"] = {str(int(c)): int(n) for c, n in class_support.items()}
    counts["year_support"] = {
        str(int(y)): int(n) for y, n in holdout["Year"].value_counts().sort_index().items()
    }
    manifest = {
        "schema": "nfi-fieldtruth-freeze-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": code_sha, "runtime_image": args.runtime_image,
        "preparation_runtime": runtime,
        "source_sha256": {
            str(p.relative_to(root)): sha256_file(p)
            for p in [
                root / "imint/eval/fieldtruth.py",
                root / "scripts/prepare_nfi_holdout.py",
                root / "scripts/validate_against_nfi.py",
                root / "scripts/inference_comparison.py",
                root / "scripts/compare_nmd2023_nfi.py",
                root / "scripts/score_nfi_holdout.py",
                root / "scripts/race_rigor_stats.py",
                root / "imint/training/unified_dataset.py",
                root / "imint/training/errors.py",
            ]
        },
        "identity": NFI_KEY, "cells": cells,
        "inputs": [file_identity(p) for p in inputs],
        "baselines": {
            "NMD2023": file_identity(args.nmd2023),
            **({"NMD2018": file_identity(args.nmd2018)} if args.nmd2018 else {}),
        },
        "tiles": {n: dict(file_identity(paths[n]), geometry=metadata[n])
                  for n in sorted(holdout["tile_name"].unique())},
        "selection": counts, "excluded_tiles": excluded,
        "campaign": {"tiles": len(staging), "training_tiles": 0,
                     "root": str(args.staging_dir.resolve())},
        "protocol": {"truth_dominant_fraction": 0.7, "classes": [0, 1, 2, 3, 4],
                     "primary_head": "class", "bootstrap_seed": 20260818,
                     "bootstrap_samples": 10000, "block_km": 50, "sesoi": 0.02},
        "execution": "awaiting_user_go_no_go",
    }
    for path, original in input_stats.items():
        stat = path.stat()
        if original != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns):
            raise ValueError(f"input changed during holdout preparation: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for label, frame in (("holdout", holdout), ("training", training)):
        dest = args.out_dir / f"{label}.parquet"
        frame.to_parquet(dest, index=False)
        assert len(pd.read_parquet(dest)) == len(frame)
        manifest[label] = {"file": dest.name, "sha256": sha256_file(dest),
                           "observations": len(frame)}
    dest = args.out_dir / "manifest.json"
    dest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(dest), "selection": counts,
                      "campaign_tiles_in_training": 0,
                      "state": "awaiting_user_go_no_go"}, indent=2))


if __name__ == "__main__":
    main()
