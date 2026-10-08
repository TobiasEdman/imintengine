#!/usr/bin/env python3
"""Prepare a reproducible NFI holdout on CPU, only after Tessera promotion.

Run in the data environment: raw observations and the resulting parquet stay
there. Only NMD coverage is checked; no inference, scoring or model fitting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
import io
import os
import re
from pathlib import Path
import sys
import subprocess

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imint.eval.fieldtruth import (
    NFI_KEY, NFI_SOURCE_FILES, NFI_PROTOCOL, exclude_training_observations, observation_keys,
    require_columns, resolve_observation_year, sha256_file,
    NFI_YEAR_SAMPLING, validate_year_selection, validate_year_balance,
)
from build_pinned_plot_set import npz_key_names, npz_version_ok
from gen_ladder_manifests import DISTILL
from imint.training.tile_time import resolve_tile_year
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
    verify = report.get("verify")
    states = report.get("states")
    if not isinstance(verify, dict) or not isinstance(states, dict):
        raise ValueError("promotion report lacks required counter objects")
    required = {"has_v1", "has_v2_left", "stamped"}
    if not required <= set(verify):
        raise ValueError("promotion report lacks required verification counters")
    for counters in (verify, states):
        if any(type(v) is not int or v < 0 for v in counters.values()):
            raise ValueError("promotion counters must be nonnegative integers")
    if (report.get("source") != "geotessera-0.10.2"
            or not cohort_count
            or verify.get("has_v1") != cohort_count
            or verify.get("stamped") != cohort_count
            or verify.get("FLAG_ON_EMPTY", 0) != 0
            or verify["has_v2_left"] != 0
            or states.get("promoted", 0) + states.get("already", 0) != cohort_count
            or states.get("unreadable", 0) != 0
            or states.get("no_v2", 0) != 0):
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


def teacher_head_metadata(payload: bytes, split: dict, features: pd.DataFrame) -> dict:
    """Check producer metadata; historical files do not bind the exact split."""
    digest = hashlib.sha256(payload).hexdigest()
    with np.load(io.BytesIO(payload), allow_pickle=False) as head:
        actual = {}
        for name in ("seed", "n_train_plots", "n_features"):
            if name not in head or head[name].shape != () or head[name].dtype.kind not in "iu":
                raise ValueError(f"teacher head lacks integer scalar {name}")
            actual[name] = int(head[name])
    for name in ("seed", "n_train_plots"):
        if type(split.get(name)) is not int or actual[name] != split[name]:
            raise ValueError(f"teacher head/split {name} mismatch")
    width = sum(re.fullmatch(r"f\d{3}", str(col)) is not None for col in features.columns)
    if not width or actual["n_features"] != width:
        raise ValueError("teacher head/feature width mismatch")
    return dict(actual, head_sha256=digest, sidecar_head_sha=digest[:16])


def teacher_sidecar_inventory(directory: Path, expected: set[str], head_sha: str,
                              staging: set[str]) -> dict:
    """Authenticate all dense-label files, including tiles without NFI plots."""
    paths = {p.stem: p for p in directory.glob("*.npz")}
    if not expected or expected - set(paths):
        raise ValueError(f"missing teacher sidecars in {directory}: {len(expected - set(paths))}")
    if set(paths) & staging:
        raise ValueError("evaluation-only campaign has distilled training sidecars")
    records = {}
    for name, path in sorted(paths.items()):
        payload = path.read_bytes()
        with np.load(io.BytesIO(payload), allow_pickle=False) as sidecar:
            stamp = sidecar.get("head_sha")
            if (stamp is None or stamp.shape != () or stamp.dtype.kind not in "US"
                    or str(stamp) != head_sha):
                raise ValueError(f"stale or missing teacher head stamp: {path}")
        records[name] = {"path": str(path.resolve()), "bytes": len(payload),
                         "sha256": hashlib.sha256(payload).hexdigest()}
    return {"expected_tiles": len(expected), "verified_tiles": len(records),
            "head_sha": head_sha, "files": records}


def tile_readiness(path: Path) -> tuple[dict, str | None]:
    """Read data prerequisites before inference; all decisions ignore scores."""
    keys = npz_key_names(path)
    if keys is None:
        raise ValueError(f"unreadable tile: {path}")
    with np.load(path, allow_pickle=False) as data:
        flagged = int(data.get("has_tessera", 0)) > 0
        if "tessera" not in keys:
            if flagged:
                raise ValueError(f"positive has_tessera without an array: {path}")
            return {}, "tessera_gap"
        tessera = data["tessera"]
        nonempty = (tessera.ndim == 3 and tessera.size > 0
                    and np.isfinite(tessera).all() and np.any(tessera))
        if flagged and not nonempty:
            raise ValueError(f"positive has_tessera on empty/invalid array: {path}")
        if str(data.get("tessera_source", "")) != "geotessera-0.10.2":
            raise ValueError(f"unpromoted Tessera source: {path}")
        if not flagged or not nonempty:
            return {}, "tessera_gap"
        required = {"spectral", "s1_vv_vh", "b08", "rededge"}
        if missing := required - keys:
            return {}, "missing_keys:" + ",".join(sorted(missing))
        if not npz_version_ok(path, ("s1_vv_vh",)):
            return {}, "s1_version"
        sar = data["s1_vv_vh"]
        if (not int(data.get("has_s1", 0)) or sar.ndim != 3 or sar.shape[0] != 2
                or not np.isfinite(sar).all() or not np.any(sar)):
            return {}, "s1_gap"
        spectral = data["spectral"]
        if spectral.ndim != 3 or not np.isfinite(spectral).all():
            raise ValueError(f"invalid spectral array: {path}")
        h, w = spectral.shape[-2:]
        expected_shapes = {"spectral": (24, h, w), "tessera": (128, h, w),
                           "s1_vv_vh": (2, h, w), "b08": (4, h, w),
                           "rededge": (12, h, w)}
        for name, expected in expected_shapes.items():
            if data[name].shape != expected:
                return {}, "mandatory_input_shape:" + name
        for name in ("b08", "rededge"):
            if not np.isfinite(data[name]).all():
                return {}, "mandatory_input_nonfinite:" + name
        explicit_year_keys = [key for key in ("year", "lpis_year") if key in data]
        for key in [*explicit_year_keys, "easting", "northing"]:
            if (key not in data or data[key].shape != ()
                    or data[key].dtype.kind not in "iuf"
                    or not np.isfinite(data[key])):
                return {}, "invalid_model_metadata:" + key
            if key in explicit_year_keys and not float(data[key]).is_integer():
                return {}, "invalid_model_metadata:" + key
        try:
            year = resolve_tile_year(data)
        except (TypeError, ValueError):
            return {}, "conflicting_or_invalid_spectral_year"
        if year is None:
            return {}, "unknown_spectral_year"
        year_source = explicit_year_keys[0] if explicit_year_keys else "dates"
        if ("doy" not in data or data["doy"].shape != (4,)
                or not np.isfinite(data["doy"]).all()
                or not ((data["doy"] >= 0) & (data["doy"] <= 366)).all()):
            return {}, "invalid_model_metadata:doy"
        return {"height": int(spectral.shape[-2]),
                "width": int(spectral.shape[-1]), "year": int(year),
                "year_source": year_source, "model_location_present": True}, None


def audit_auxiliary_inputs(path: Path, geometry: dict, requirements: dict) -> tuple[dict, list[str]]:
    """Audit stored channels against the union of authenticated checkpoints.

    Zeros are valid physical values. Partial markfukt NaNs and an absent SAR
    baseline retain the training loader's documented neutral-fill policy.
    Optional CROMA optical padding is recorded by frame for go/no-go review.
    """
    names = {name for cell in requirements["cells"].values()
             for name in cell["enabled_aux_names"]}
    shape = (geometry["height"], geometry["width"])
    computed = set(requirements["computed_channels"])
    nan_channels = set(requirements["nan_nodata_channels"])
    gaps, nodata, failures = {}, {}, []

    def fail(name, reason):
        gaps[name] = reason
        failures.append(f"{name}:{reason}")

    with np.load(path, allow_pickle=False) as data:
        for name in sorted(names - computed):
            if name not in data:
                fail(name, "missing")
                continue
            arr = data[name]
            if arr.shape != shape or arr.dtype.kind not in "biuf":
                fail(name, "invalid_shape_or_dtype")
                continue
            if name in nan_channels:
                nodata[name] = float(np.isnan(arr).mean())
                if np.isinf(arr).any() or not np.isfinite(arr).any():
                    fail(name, "invalid_or_all_nodata")
            elif not np.isfinite(arr).all():
                fail(name, "nonfinite")
        if names & computed:
            name = "s1_vv_vh_2016"
            if name not in data:
                gaps[name] = "native_neutral_delta_no_baseline"
            else:
                arr = data[name]
                if arr.shape != (2, *shape) or arr.dtype.kind not in "biuf":
                    fail(name, "invalid_shape_or_dtype")
                elif not np.isfinite(arr).any() or not np.any(np.nan_to_num(arr)):
                    fail(name, "all_nodata_baseline")
                else:
                    nodata[name] = float((~np.isfinite(arr) | (arr <= 0)).mean())
        if any(cell.startswith("croma_") for cell in requirements["cells"]):
            for name in ("b01", "b09"):
                if int(data.get("has_" + name, 1)) == 0 or name not in data:
                    gaps[name] = "native_optical_zero_padding"
                    continue
                arr = data[name]
                if (arr.ndim not in (2, 3) or arr.shape[-2:] != shape
                        or not arr.size or arr.dtype.kind not in "biuf"):
                    fail(name, "invalid_shape_or_dtype")
                    continue
                frames = arr[None, ...] if arr.ndim == 2 else arr
                empty = [i for i, frame in enumerate(frames)
                         if not np.any(np.isfinite(frame) & (frame != 0))]
                if empty:
                    gaps[name] = {"native_zero_padded_frames": empty}
    return {"gaps": gaps, "nodata_fraction": nodata}, sorted(failures)


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
    paired = pd.concat(candidates)
    if "tile_role" in paired:
        paired["_tile_priority"] = paired["tile_role"].map({"campaign": 0, "evaluation": 1, "cohort": 2})
        if paired["_tile_priority"].isna().any():
            raise ValueError("unknown tile role")
        paired = paired.sort_values(NFI_KEY + ["_tile_priority", "tile_name"]).drop(columns="_tile_priority")
    else:
        paired = paired.sort_values(NFI_KEY + ["tile_name"])
    holdout = paired.drop_duplicates(NFI_KEY).reset_index(drop=True)
    if holdout.empty:
        raise ValueError("holdout is empty")
    assert not observation_keys(holdout).isin(observation_keys(training)).any()
    counts.update(common_support_rows=len(paired),
                  observations=len(holdout), duplicate_rows_removed=len(paired)-len(holdout),
                  training_observation_overlap=0)
    if "tile_role" in holdout:
        counts["tile_role_support"] = {str(role): int(n) for role, n in holdout["tile_role"].value_counts().items()}
    return holdout, counts


def filter_nmd_coverage(holdout: pd.DataFrame, baselines: dict[str, Path],
                        model_python: str) -> tuple[pd.DataFrame, dict]:
    """Check coverage only before annual quotas; never derive or score classes."""
    coordinates = holdout[["Easting", "Northing"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("NMD coverage requires finite observation coordinates")
    try:
        result = subprocess.run(
            [model_python, str(Path(__file__).with_name("nfi_nmd_coverage.py"))],
            input=json.dumps({"coordinates": coordinates.tolist(),
                              "baselines": {name: str(path) for name, path in baselines.items()}},
                             allow_nan=False),
            capture_output=True, text=True, check=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    except subprocess.CalledProcessError as exc:
        raise ValueError("NMD coverage preparation failed: " + exc.stderr[-2000:]) from exc
    masks = json.loads(result.stdout)
    if (set(masks) != set(baselines)
            or any(not isinstance(mask, list) or len(mask) != len(holdout)
                   or any(type(v) is not bool for v in mask) for mask in masks.values())):
        raise ValueError("invalid NMD coverage support returned by model interpreter")
    common = np.ones(len(holdout), dtype=bool)
    counts = {}
    for name, mask in masks.items():
        covered = np.asarray(mask, dtype=bool)
        common &= covered
        counts[name] = {"covered": int(covered.sum()), "uncovered": int((~covered).sum())}
    if not common.any():
        raise ValueError("no common NMD coverage before holdout selection")
    selected = holdout.loc[common].copy().reset_index(drop=True)
    return selected, {
        "nmd_coverage": counts,
        "pre_nmd_coverage_observations": len(holdout),
        "excluded_for_nmd_coverage": int((~common).sum()),
        "nmd_coverage_excluded_by_year": {
            str(int(year)): int((~common[holdout.Year.to_numpy() == year]).sum())
            for year in sorted(holdout.Year.unique())
        },
    }


def select_balanced_years(holdout: pd.DataFrame, selection: dict) -> tuple[pd.DataFrame, dict]:
    """Select equal plot-year counts by identity hash, independent of truth/scores."""
    protocol = {"primary_population": "balanced", "year_selection": selection}
    validate_year_selection(protocol)
    keys = observation_keys(holdout)
    if keys.has_duplicates:
        raise ValueError("year selection requires unique plot-years")
    years, count = selection["years"], selection["observations_per_year"]
    candidates = holdout.loc[holdout["Year"].isin(years)].copy()
    available = {str(y): int((candidates["Year"] == y).sum()) for y in years}
    if any(n < count for n in available.values()):
        raise ValueError(f"insufficient eligible observations for equal annual target {count}: {available}")
    seed = selection["seed"]
    candidates["_year_rank"] = [
        hashlib.sha256(f"{seed}:{tract}:{plot}:{year}".encode("ascii")).hexdigest()
        for tract, plot, year in observation_keys(candidates)
    ]
    selected = (candidates.sort_values(["_year_rank", *NFI_KEY])
                .groupby("Year", sort=True).head(count)
                .drop(columns="_year_rank").sort_values(NFI_KEY).reset_index(drop=True))
    validate_year_balance(selected, protocol)
    return selected, {
        "pre_balance_observations": len(holdout),
        "outside_selected_years": len(holdout) - len(candidates),
        "eligible_by_selected_year": available,
        "excluded_by_year_quota": len(candidates) - len(selected),
    }


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
    for name in ("plot-index", "teacher-index", "teacher-root", "distill-root", "checkpoint-root",
                 "cohort-dir", "staging-dir", "promotion-job-json",
                 "promotion-report", "nmd2023", "out-dir"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--nmd2018", type=Path)
    ap.add_argument("--evaluation-dir", type=Path, action="append", default=[],
                    help="additional evaluation-only data root; repeat for multiple years")
    ap.add_argument("--population", choices=("balanced", "all", "campaign"), default="balanced",
                    help="balanced is the required primary study; all/campaign are diagnostic")
    ap.add_argument("--inventory-years", type=int, nargs="+",
                    help="explicit inventory years approved for equal-count selection")
    ap.add_argument("--observations-per-year", type=int,
                    help="explicit positive equal target after common input support checks")
    ap.add_argument("--runtime-image", required=True,
                    help="reviewed inference image pinned with @sha256")
    ap.add_argument("--runtime-manifest", type=Path, required=True,
                    help="sealed image provenance, normally /opt/provenance/runtime.json")
    ap.add_argument("--source-git-sha", required=True, help="reviewed build source SHA")
    args = ap.parse_args()
    year_selection = None
    if args.population == "balanced":
        year_selection = dict(NFI_YEAR_SAMPLING, years=args.inventory_years,
                              observations_per_year=args.observations_per_year)
        validate_year_selection({"primary_population": "balanced", "year_selection": year_selection})
    elif args.inventory_years is not None or args.observations_per_year is not None:
        raise ValueError("year targets require --population balanced")
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
    paths = {**cohort, **staging}
    evaluation_names = set(staging)
    evaluation_roots = [{"root": str(args.staging_dir.resolve()), "tiles": len(staging),
                         "role": "campaign"}]
    for directory in args.evaluation_dir:
        additional = {p.stem: p for p in directory.glob("*.npz") if not p.name.endswith(".tmp.npz")}
        if not additional or set(additional) & set(paths):
            raise ValueError("evaluation roots must be nonempty with globally unique tile names")
        paths.update(additional)
        evaluation_names.update(additional)
        evaluation_roots.append({"root": str(directory.resolve()), "tiles": len(additional),
                                 "role": "evaluation"})
    input_stats = {}
    parsed_identities = {}
    def capture(path):
        stat = path.stat()
        input_stats[path] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def read_input(path, kind):
        # Identity describes exactly the bytes parsed, not a later reread.
        payload = path.read_bytes()
        parsed_identities[path.resolve()] = {"path": str(path.resolve()), "bytes": len(payload),
                                   "sha256": hashlib.sha256(payload).hexdigest()}
        if kind == "parquet":
            return pd.read_parquet(io.BytesIO(payload))
        return json.loads(payload)

    for path in [args.plot_index, args.teacher_index, args.promotion_job_json,
                 args.promotion_report, args.nmd2023] + ([args.nmd2018] if args.nmd2018 else []):
        capture(path)
    check_promotion(read_input(args.promotion_job_json, "json"),
                    read_input(args.promotion_report, "json"), len(cohort))
    index = read_input(args.plot_index, "parquet")
    source = read_input(args.teacher_index, "parquet")
    if not set(index["tile_name"].astype(str)) <= set(paths):
        raise ValueError("indexed tile absent from the declared data roots")
    index["tile_path"] = index["tile_name"].map(lambda n: str(paths[str(n)].resolve()))
    index["tile_role"] = index["tile_name"].map(
        lambda n: "campaign" if str(n) in staging else "evaluation" if str(n) in evaluation_names else "cohort")
    inputs = [args.plot_index, args.teacher_index, args.promotion_job_json,
              args.promotion_report, args.nmd2023]
    if args.nmd2018:
        inputs.append(args.nmd2018)
    trained, exposure = [], []
    teacher_provenance = {}
    campaign_training_names = set()
    cohort_keys = {}
    for name, path in cohort.items():
        keys = npz_key_names(path)
        if keys is None:
            raise ValueError(f"unreadable cohort tile during teacher audit: {path}")
        cohort_keys[name] = keys
    for model in DISTILL:
        features = args.teacher_root / f"{model}_r2_plot_features.parquet"
        split = args.teacher_root / f"{model}_r2_split.json"
        capture(features)
        capture(split)
        head_path = args.teacher_root / f"{model}_r2_head.npz"
        capture(head_path)
        feature_frame = read_input(features, "parquet")
        split_record = read_input(split, "json")
        campaign_training_names.update(set(split_record["train_tiles"]) & set(staging))
        trained.append(teacher_training_set(feature_frame, split_record, source, evaluation_names))
        # Older heads lack split/feature digests: conservatively exclude every
        # observation ever in the recorded feature pool, including its test set.
        exposure.append(resolve_observation_year(feature_frame, source)[NFI_KEY])
        payload = head_path.read_bytes()
        head = teacher_head_metadata(payload, split_record, feature_frame)
        parsed_identities[head_path.resolve()] = {"path": str(head_path.resolve()), "bytes": len(payload),
                                        "sha256": head["head_sha256"]}
        required = DISTILL[model].get("require_keys", ())
        expected = {n for n, keys in cohort_keys.items()
                    if set(required) <= keys and npz_version_ok(cohort[n], tuple(required))}
        directory = args.distill_root / f"{model}_r2"
        for path in directory.glob("*.npz"):
            capture(path)
        sidecars = teacher_sidecar_inventory(directory, expected, head["sidecar_head_sha"], evaluation_names)
        teacher_provenance[model] = {"head": head, "sidecars": sidecars,
                                   "historical_split_digest_available": False}
        inputs.extend([features, split, head_path])
    observed_training = pd.concat(trained).drop_duplicates(NFI_KEY)
    training = pd.concat(exposure).drop_duplicates(NFI_KEY).sort_values(NFI_KEY)
    campaign_training_tiles = len(campaign_training_names)
    cells = {
        f"{m}_r{r}": {
            "checkpoint": file_identity(args.checkpoint_root / f"{m}_r{r}" / "best_model.pt"),
            "backbone": cfg["backbone"], "img_size": cfg["img_size"],
            "num_classes": 23 if r == 1 else 28, "head": "class",
        }
        for m, cfg in DISTILL.items() for r in range(1, 5)
    }
    try:
        extracted = subprocess.run(
            [runtime["environments"]["model"]["python"]["path"],
             str(root / "scripts/nfi_checkpoint_inputs.py")],
            input=json.dumps(cells), capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError("checkpoint metadata preparation failed: " + exc.stderr[-4000:]) from exc
    requirements = json.loads(extracted.stdout)
    if set(requirements["cells"]) != set(cells):
        raise ValueError("checkpoint input contract roster mismatch")
    for cell, contract in requirements["cells"].items():
        cells[cell]["input_contract"] = contract
    auxiliary_audit = {}
    metadata, excluded = {}, {}
    # Inspect every declared evaluation tile, including those without indexed plots.
    names = evaluation_names | set(index["tile_name"].astype(str))
    for name in sorted(names):
        capture(paths[name])
        meta, reason = tile_readiness(paths[name])
        if reason:
            excluded[name] = reason
        else:
            audit, failures = audit_auxiliary_inputs(paths[name], meta, requirements)
            auxiliary_audit[name] = audit
            if failures:
                excluded[name] = "auxiliary_gaps:" + ",".join(failures)
            else:
                metadata[name] = meta
    candidate_index = index[index["tile_role"] == "campaign"] if args.population == "campaign" else index
    holdout, counts = select_holdout(candidate_index, training, metadata)
    baselines = {"NMD2023": args.nmd2023}
    if args.nmd2018:
        baselines["NMD2018"] = args.nmd2018
    holdout, coverage_counts = filter_nmd_coverage(
        holdout, baselines, runtime["environments"]["model"]["python"]["path"])
    counts.update(coverage_counts, observations=len(holdout))
    counts["tile_role_support"] = {str(role): int(n) for role, n in holdout["tile_role"].value_counts().items()}
    if year_selection is not None:
        holdout, balance_counts = select_balanced_years(holdout, year_selection)
        counts.update(balance_counts, observations=len(holdout))
        counts["tile_role_support"] = {str(role): int(n) for role, n in holdout["tile_role"].value_counts().items()}
    counts["primary_population"] = args.population
    counts["all_candidate_rows"] = len(index)
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
    class_support = holdout["nfi_forest"].value_counts().sort_index().to_dict()
    counts["class_support"] = {str(int(c)): int(n) for c, n in class_support.items()}
    counts["year_support"] = {
        str(int(y)): int(n) for y, n in holdout["Year"].value_counts().sort_index().items()
    }
    input_identities = [file_identity(p) for p in inputs]
    for record in input_identities:
        parsed = parsed_identities.get(Path(record["path"]))
        if parsed is not None and parsed != record:
            raise ValueError("parsed input changed during preparation: " + record["path"])
    manifest = {
        "schema": "nfi-fieldtruth-freeze-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": code_sha, "runtime_image": args.runtime_image,
        "preparation_runtime": runtime,
        "source_sha256": {name: sha256_file(root / name) for name in sorted(NFI_SOURCE_FILES)},
        "identity": NFI_KEY, "cells": cells,
        "inputs": input_identities,
        "baselines": {
            "NMD2023": file_identity(args.nmd2023),
            **({"NMD2018": file_identity(args.nmd2018)} if args.nmd2018 else {}),
        },
        "tiles": {n: dict(file_identity(paths[n]), geometry=metadata[n])
                  for n in sorted(holdout["tile_name"].unique())},
        "selection": counts, "excluded_tiles": excluded,
        "auxiliary_audit": auxiliary_audit,
        "auxiliary_policy": "exclude missing or malformed stored aux; record native neutral-fill gaps",
        "teacher_provenance": teacher_provenance,
        "exposure_policy": "exclude_all_recorded_teacher_feature_plot_years",
        "recorded_training_plot_years": len(observed_training),
        "excluded_teacher_feature_plot_years": len(training),
        "evaluation_roots": evaluation_roots,
        "campaign": {"tiles": len(staging), "training_tiles": campaign_training_tiles,
                     "root": str(args.staging_dir.resolve())},
        "protocol": dict(NFI_PROTOCOL, primary_population=args.population,
                         **({"year_selection": year_selection} if year_selection is not None else {})),
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
    print(json.dumps({"manifest": str(dest), "manifest_sha256": sha256_file(dest), "selection": counts,
                      "campaign_tiles_in_training": campaign_training_tiles,
                      "state": "awaiting_user_go_no_go"}, indent=2))


if __name__ == "__main__":
    main()
