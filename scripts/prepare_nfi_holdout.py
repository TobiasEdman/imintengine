#!/usr/bin/env python3
"""Prepare a reproducible NFI holdout on CPU, only after Tessera promotion.

Run in the data environment: raw observations and the resulting parquet stay
there. Only NMD coverage is checked; no inference, scoring or model fitting.
"""
from __future__ import annotations

import argparse
from collections import Counter
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


def tile_files(directory: Path) -> dict[str, Path]:
    """Temporary writer outputs are never members of a declared root."""
    return {p.stem: p for p in directory.glob("*.npz")
            if not p.name.endswith(".tmp.npz")}


def parse_evaluation_root(value: str) -> tuple[str, int | None, Path]:
    parts = value.split(":", 2)
    if (len(parts) != 3 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", parts[0])
            or parts[0] in {"cohort", "campaign"} or not parts[2]):
        raise argparse.ArgumentTypeError("evaluation root requires ROLE:YEAR:PATH")
    role, year, directory = parts
    if year != "any" and not re.fullmatch(r"[1-9][0-9]{3}", year):
        raise argparse.ArgumentTypeError("evaluation root YEAR must be YYYY or any")
    return role, None if year == "any" else int(year), Path(directory)


def checked_tile_year(data) -> int:
    for key in ("year", "lpis_year"):
        if key in data:
            value = np.asarray(data[key])
            if (value.shape != () or value.dtype.kind not in "iuf"
                    or not np.isfinite(value) or not float(value).is_integer()):
                raise ValueError("invalid explicit tile year")
    year = resolve_tile_year(data)
    if year is None or year < 1:
        raise ValueError("unknown tile year")
    return year


def validate_root_year(path: Path, expected: int | None) -> None:
    with np.load(path, allow_pickle=False) as data:
        year = checked_tile_year(data)
    if expected is not None and year != expected:
        raise ValueError(f"evaluation root year mismatch or unknown tile year: {path}")


def training_footprints(paths: dict[str, Path]) -> list[dict]:
    """Conservative cohort footprints in SWEREF99 TM, at the tile's 10 m GSD.

    Invalid geometry/year fails closed, including unindexed training tiles.
    Read the full array only when legacy metadata lacks an explicit bbox.
    """
    footprints = []
    for name, path in sorted(paths.items()):
        with np.load(path, allow_pickle=False) as data:
            try:
                year = checked_tile_year(data)
            except ValueError as exc:
                raise ValueError(f"invalid training footprint year: {path}") from exc
            if "bbox_3006" in data:
                bbox = np.asarray(data["bbox_3006"], dtype=float)
            else:
                image = data["spectral"] if "spectral" in data else data["image"]
                if image.ndim != 3 or min(image.shape[-2:]) <= 0:
                    raise ValueError(f"invalid training footprint shape: {path}")
                for key in ("easting", "northing"):
                    value = np.asarray(data[key])
                    if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value):
                        raise ValueError(f"invalid training footprint center: {path}")
                x, y = float(data["easting"]), float(data["northing"])
                h, w = image.shape[-2:]
                bbox = np.array([x - w * 5, y - h * 5, x + w * 5, y + h * 5])
            if (bbox.shape != (4,) or not np.isfinite(bbox).all()
                    or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
                raise ValueError(f"invalid training footprint bbox: {path}")
            footprints.append(dict(name=name, year=int(year), bbox=bbox.tolist()))
    return footprints


def exclude_training_footprints(index: pd.DataFrame, footprints: list[dict]) -> pd.DataFrame:
    require_columns(index, ["Easting", "Northing", "Year"])
    xy = index[["Easting", "Northing"]].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("footprint exclusion requires finite plot coordinates")
    overlap = np.zeros(len(index), dtype=bool)
    for footprint in footprints:
        west, south, east, north = footprint["bbox"]
        # Include boundary points conservatively; different years stay eligible.
        overlap |= ((index.Year.to_numpy() == footprint["year"])
                    & (xy[:, 0] >= west) & (xy[:, 0] <= east)
                    & (xy[:, 1] >= south) & (xy[:, 1] <= north))
    # Exclude the entire plot-year, even if another tile carries the same key.
    keys = observation_keys(index)
    return index.loc[~keys.isin(keys[overlap])].copy()


def select_holdout(
    index: pd.DataFrame, training: pd.DataFrame, tile_meta: dict,
    *, footprints: list[dict] | None = None, campaign_only: bool = False,
) -> tuple[pd.DataFrame, dict]:
    require_columns(index, NFI_KEY + ["tile_name", "row", "col"])
    if index.duplicated(NFI_KEY + ["tile_name"]).any():
        raise ValueError("duplicate plot-year/tile in the NFI index")
    eligible = exclude_training_observations(index, training)
    counts = {"index_rows": len(index),
              "index_observations": len(observation_keys(index).unique()),
              "training_overlap_rows": len(index) - len(eligible)}
    before = len(eligible)
    if "tile_role" in eligible:
        keys = observation_keys(eligible)
        cohort_keys = keys[eligible["tile_role"].to_numpy() == "cohort"]
        eligible = eligible.loc[~keys.isin(cohort_keys)].copy()
    counts["cohort_overlap_rows"] = before - len(eligible)
    before = len(eligible)
    if footprints is not None:
        eligible = exclude_training_footprints(eligible, footprints)
    counts["training_footprint_overlap_rows"] = before - len(eligible)
    # Population selection must not hide cohort identities or footprint exposure.
    if campaign_only:
        eligible = eligible.loc[eligible["tile_role"] == "campaign"].copy()
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
        paired["_tile_priority"] = paired["tile_role"].map(lambda role: 0 if role == "campaign" else 1)
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


#: The selection's own messages for a population without support.
EMPTY_SUPPORT = frozenset({"no eligible observations with common input support", "holdout is empty",
                           "no common NMD coverage before holdout selection"})


def mean_filled_absent_only(failures: list[str], nan_channels: set[str]) -> bool:
    """True when every aux failure is an absent channel that training fills with
    its mean: a NaN-nodata channel (height, markfukt). Training zero-fills any
    other absent channel, which lands far from its mean after normalization."""
    return bool(failures) and all(
        reason == "missing" and name in nan_channels
        for name, _, reason in (failure.rpartition(":") for failure in failures))


def frame_2016_on_bar(path: Path) -> bool:
    """The census's test for the 2016 baseline: flagged, with a non-empty frame
    and 2016 SAR. Reported only; the freeze never requires it."""
    with np.load(path, allow_pickle=False) as data:
        if int(data.get("has_frame_2016", 0)) != 1:
            return False
        return all(name in data and bool(np.any(data[name])) for name in ("frame_2016", "s1_vv_vh_2016"))


def support_summary(holdout: pd.DataFrame, frame_ok: dict[str, bool]) -> dict:
    """Aggregate support of a selected population: no identities, names or coordinates."""
    years = holdout["Year"].astype(int)
    frame = holdout["tile_name"].astype(str).map(frame_ok.__getitem__).astype(bool)
    return {
        "observations": len(holdout),
        "year_support": {str(y): int(n) for y, n in years.value_counts().sort_index().items()},
        "role_year_support": {
            str(role): {str(int(y)): int(n) for y, n in group["Year"].value_counts().sort_index().items()}
            for role, group in holdout.groupby("tile_role", sort=True)},
        "frame_2016_by_year": {
            str(int(y)): {"present": int(f.sum()), "absent": int((~f).sum())}
            for y, f in frame.groupby(years, sort=True)},
    }


def support_report(index: pd.DataFrame, training: pd.DataFrame, *, footprints: list[dict],
                   metadata: dict, absent_mean_filled: dict, excluded: dict[str, str],
                   roles: dict[str, str], evaluation_names: set[str], paths: dict[str, Path],
                   baselines: dict[str, Path], model_python: str, evaluation_roots: list[dict]) -> dict:
    """Per-year support before any quota exists; reads only, writes nothing.

    Every variant goes through the freeze's own selection and NMD coverage:
    - freeze: exactly the tiles the freeze admits;
    - absent_aux_mean_fill: also tiles whose only aux failures are absent
      NaN-nodata channels, which training fills with their mean (Tobias
      2026-10-22 for height);
    - freeze_with_frame_2016: the freeze's tiles that also carry frame_2016 and
      2016 SAR (Tobias 2026-10-24: report every bar, choose at the quota).
    Counts are after NMD coverage, as in the freeze manifest's selection.
    """
    candidates = (set(metadata) | set(absent_mean_filled)) & evaluation_names
    frame_ok = {name: frame_2016_on_bar(paths[name]) for name in sorted(candidates)}
    variants = {
        "freeze": metadata,
        "absent_aux_mean_fill": {**metadata, **absent_mean_filled},
        "freeze_with_frame_2016": {n: m for n, m in metadata.items() if n not in evaluation_names or frame_ok[n]},
    }
    excluded_by_role: dict[str, Counter] = {}
    for name, reason in excluded.items():
        if name in evaluation_names:
            excluded_by_role.setdefault(roles[name], Counter())[reason] += 1
    report = {"schema": "nfi-support-report-v1", "evaluation_roots": evaluation_roots,
              "excluded_evaluation_tiles": {role: dict(c) for role, c in sorted(excluded_by_role.items())},
              "variants": {}}
    for label, tiles in variants.items():
        admitted = dict(Counter(roles[name] for name in tiles if name in evaluation_names))
        try:
            holdout, counts = select_holdout(index, training, tiles, footprints=footprints)
            holdout, coverage = filter_nmd_coverage(holdout, baselines, model_python)
        except ValueError as exc:
            # A variant may legitimately have no support; any other failure stays fatal.
            if str(exc) not in EMPTY_SUPPORT:
                raise
            report["variants"][label] = {"observations": 0, "empty_support": str(exc),
                                         "admitted_tiles_by_role": admitted}
            continue
        if not set(holdout["tile_name"].astype(str)) <= candidates:
            raise ValueError("support selected a tile outside the evaluation roots")
        counts.update(coverage, observations=len(holdout))
        counts["tile_role_support"] = {str(r): int(n) for r, n in holdout["tile_role"].value_counts().items()}
        report["variants"][label] = dict(support_summary(holdout, frame_ok), selection=counts,
                                         admitted_tiles_by_role=admitted)
    return report


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
                 "promotion-report", "nmd2023"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, help="new freeze directory; required unless --report-only")
    ap.add_argument("--report-only", action="store_true",
                    help="print per-year support under the freeze's rules and its report variants; write nothing")
    ap.add_argument("--nmd2018", type=Path)
    ap.add_argument("--evaluation-root", type=parse_evaluation_root, action="append", default=[],
                    help="repeat ROLE:YEAR:PATH; use any for a mixed-year root")
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
    # Support is measured before any quota exists, so a report is never balanced.
    if args.report_only and (args.population != "all" or args.out_dir is not None):
        raise ValueError("--report-only requires --population all and no --out-dir")
    year_selection = None
    if args.population == "balanced":
        year_selection = dict(NFI_YEAR_SAMPLING, years=args.inventory_years,
                              observations_per_year=args.observations_per_year)
        validate_year_selection({"primary_population": "balanced", "year_selection": year_selection})
    elif args.inventory_years is not None or args.observations_per_year is not None:
        raise ValueError("year targets require --population balanced")
    if not args.report_only:
        if args.out_dir is None:
            raise ValueError("--out-dir is required unless --report-only")
        if args.out_dir.exists():
            raise ValueError("freeze output already exists; never overwrite it")
    root = Path(__file__).resolve().parents[1]
    runtime = preparation_runtime(root, args.runtime_manifest,
                                  args.source_git_sha, args.runtime_image)
    code_sha = runtime["source"]["git_sha"]
    cohort = tile_files(args.cohort_dir)
    staging = tile_files(args.staging_dir)
    if len(staging) != 475 or set(staging) & set(cohort):
        raise ValueError("campaign must contain 475 evaluation-only, disjoint tiles")
    paths = {**cohort, **staging}
    evaluation_names = set(staging)
    evaluation_roots = [{"root": str(args.staging_dir.resolve()), "tiles": len(staging),
                         "role": "campaign"}]
    roles = {name: "cohort" for name in cohort}
    roles.update({name: "campaign" for name in staging})
    for role, year, directory in args.evaluation_root:
        additional = tile_files(directory)
        if not additional or set(additional) & set(paths):
            raise ValueError("evaluation roots must be nonempty with globally unique tile names")
        for path in additional.values():
            validate_root_year(path, year)
        paths.update(additional)
        roles.update({name: role for name in additional})
        evaluation_names.update(additional)
        evaluation_roots.append({"root": str(directory.resolve()), "tiles": len(additional),
                                 "role": role, "year": year})
    input_stats = {}
    parsed_identities = {}
    def capture(path):
        stat = path.stat()
        current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        # Keep the earliest snapshot: a later read cannot bless changed geometry.
        original = input_stats.setdefault(path, current)
        if current != original:
            raise ValueError(f"input changed during holdout preparation: {path}")

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
    index["tile_role"] = index["tile_name"].map(lambda n: roles[str(n)])
    # Capture every cohort file before its geometry is read, not only those
    # with NFI records. Bind exclusion inputs into the freeze manifest below.
    for path in cohort.values():
        capture(path)
    footprints = training_footprints(cohort)
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
    absent_mean_filled = {}  # ready tiles whose only aux failures are absent NaN-nodata channels
    nan_channels = set(requirements["nan_nodata_channels"])
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
                if mean_filled_absent_only(failures, nan_channels):
                    absent_mean_filled[name] = meta
            else:
                metadata[name] = meta
    baselines = {"NMD2023": args.nmd2023}
    if args.nmd2018:
        baselines["NMD2018"] = args.nmd2018
    model_python = runtime["environments"]["model"]["python"]["path"]
    if args.report_only:
        report = support_report(
            index, training, footprints=footprints, metadata=metadata,
            absent_mean_filled=absent_mean_filled, excluded=excluded, roles=roles,
            evaluation_names=evaluation_names, paths=paths, baselines=baselines,
            model_python=model_python, evaluation_roots=evaluation_roots)
        for path in list(input_stats):   # the frame_2016 reads come after the captures
            capture(path)
        print("=== SUPPORT_REPORT_BEGIN ===", json.dumps(report, indent=2),
              "=== SUPPORT_REPORT_END ===", sep="\n", flush=True)
        return
    holdout, counts = select_holdout(
        index, training, metadata, footprints=footprints,
        campaign_only=args.population == "campaign")
    holdout, coverage_counts = filter_nmd_coverage(holdout, baselines, model_python)
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
        "training_footprint_policy": "exclude_same_year_plot_points_in_any_cohort_bbox_inclusive",
        "training_footprint_inputs": {n: file_identity(p) for n, p in sorted(cohort.items())},
        "campaign": {"tiles": len(staging), "training_tiles": campaign_training_tiles,
                     "root": str(args.staging_dir.resolve())},
        "protocol": dict(NFI_PROTOCOL, primary_population=args.population,
                         **({"year_selection": year_selection} if year_selection is not None else {})),
        "execution": "awaiting_user_go_no_go",
    }
    for path in list(input_stats):   # re-stat every captured input
        capture(path)
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
