#!/usr/bin/env python3
"""Read-only, aggregate-only geographic audit of a frozen NFI holdout.

Tract/plot IDs are not persistent geographic identifiers across survey years.
Compare coordinates as well as identifiers; do not alter the frozen sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imint.eval.fieldtruth import (
    NFI_KEY, NFI_MODELS, load_frozen_holdout, observation_keys,
    read_verified_parquet, require_columns, resolve_observation_year, sha256_file,
)
from scripts.prepare_nfi_holdout import teacher_training_set

XY = ["Easting", "Northing"]
RADII_M = (0, 7, 14, 30, 100)


def coordinates(keys: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Join one unambiguous survey location to each observation identity."""
    require_columns(index, NFI_KEY + XY)
    observation_keys(index)
    locations = index[NFI_KEY + XY].drop_duplicates()
    if locations.duplicated(NFI_KEY).any():
        raise ValueError("conflicting coordinates for an observation")
    result = keys[NFI_KEY].drop_duplicates().merge(
        locations, on=NFI_KEY, how="left", validate="one_to_one")
    require_columns(result, NFI_KEY + XY)
    if not np.isfinite(result[XY].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite observation coordinates")
    return result


def compare_locations(holdout: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Count cross-year ID matches and spatial proximity, without row identities."""
    for frame in (holdout, reference):
        require_columns(frame, NFI_KEY + XY)
        if frame.empty or observation_keys(frame).has_duplicates:
            raise ValueError("comparison requires nonempty unique observations")
        if not np.isfinite(frame[XY].to_numpy(dtype=float)).all():
            raise ValueError("nonfinite observation coordinates")
    if observation_keys(holdout).isin(observation_keys(reference)).any():
        raise ValueError("same plot-year present in holdout and reference")
    proximity = {str(r): {"pairs": 0, "holdout_observations": 0} for r in RADII_M}
    nearest, id_distances = [], []
    id_observations = 0
    ref_xy = reference[XY].to_numpy(dtype=float)
    ref_year = reference.Year.to_numpy(dtype=int)
    ref_id = reference[NFI_KEY[:2]].to_numpy(dtype=int)
    # Bounded memory even if a future feature pool grows substantially.
    for start in range(0, len(holdout), 128):
        chunk = holdout.iloc[start:start + 128]
        cross_year = chunk.Year.to_numpy(dtype=int)[:, None] != ref_year[None, :]
        delta = chunk[XY].to_numpy(dtype=float)[:, None, :] - ref_xy[None, :, :]
        distance = np.sqrt(np.square(delta).sum(axis=2))
        same_id = (chunk[NFI_KEY[:2]].to_numpy(dtype=int)[:, None, :] == ref_id[None, :, :]).all(axis=2)
        matched = same_id & cross_year
        id_observations += int(matched.any(axis=1).sum())
        id_distances.extend(distance[matched].tolist())
        distance[~cross_year] = np.inf
        nearest.extend(distance.min(axis=1).tolist())
        for radius in RADII_M:
            within = distance <= radius
            proximity[str(radius)]["pairs"] += int(within.sum())
            proximity[str(radius)]["holdout_observations"] += int(within.any(axis=1).sum())
    finite = np.asarray(nearest)[np.isfinite(nearest)]
    return {
        "holdout_observations": len(holdout), "reference_observations": len(reference),
        "cross_year_shared_id_pairs": len(id_distances),
        "cross_year_shared_id_holdout_observations": id_observations,
        "shared_id_min_distance_m": min(id_distances) if id_distances else None,
        "cross_year_proximity_m": proximity,
        "holdout_with_cross_year_reference": len(finite),
        "nearest_cross_year_distance_quantiles_m": dict(zip(
            ("min", "p01", "median", "p99", "max"),
            np.quantile(finite, [0, .01, .5, .99, 1]).tolist())) if len(finite) else None,
    }


def audit(manifest_path: Path, expected_sha256: str) -> dict:
    holdout, manifest = load_frozen_holdout(manifest_path, expected_sha256)
    root = Path(__file__).resolve().parents[1]
    dependencies = ("imint/eval/fieldtruth.py", "scripts/prepare_nfi_holdout.py")
    for name in dependencies:
        if sha256_file(root / name) != manifest["source_sha256"][name]:
            raise ValueError("audit dependency differs from frozen source: " + name)
    # Resolve inputs from the authenticated manifest, never from a new directory scan.
    inputs = manifest["inputs"]
    used = {}

    def record(name: str) -> dict:
        matches = [r for r in inputs if Path(r["path"]).name == name]
        if len(matches) != 1:
            raise ValueError("missing or ambiguous frozen input: " + name)
        used[name] = matches[0]["sha256"]
        return matches[0]

    def parquet(name: str) -> pd.DataFrame:
        r = record(name)
        return read_verified_parquet(Path(r["path"]), r["sha256"])

    def document(name: str) -> dict:
        r = record(name)
        payload = Path(r["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != r["sha256"]:
            raise ValueError("frozen JSON input content changed")
        return json.loads(payload)

    source = parquet("nfi_index_unified_v2_512.parquet")
    holdout = coordinates(holdout, holdout)
    pools, trained, per_model = [], [], {}
    for model in NFI_MODELS:
        features = parquet(f"{model}_r2_plot_features.parquet")
        split = document(f"{model}_r2_split.json")
        pool = coordinates(resolve_observation_year(features, source), source)
        train = coordinates(teacher_training_set(features, split, source, set()), source)
        pools.append(pool)
        trained.append(train)
        per_model[model] = {"recorded_feature_pool": compare_locations(holdout, pool),
                            "recorded_training_split": compare_locations(holdout, train)}
    pool = pd.concat(pools).drop_duplicates(NFI_KEY)
    train = pd.concat(trained).drop_duplicates(NFI_KEY)
    frozen_training = read_verified_parquet(
        manifest_path.parent / manifest["training"]["file"], manifest["training"]["sha256"])
    if set(observation_keys(pool)) != set(observation_keys(frozen_training)):
        raise ValueError("feature pool does not reproduce frozen exclusion population")
    if len(train) != manifest["recorded_training_plot_years"]:
        raise ValueError("recorded teacher training count differs from freeze")
    return {
        "schema": "nfi-plot-reuse-audit/1", "manifest_sha256": expected_sha256,
        "audit_script_sha256": sha256_file(__file__), "verified_inputs_sha256": used,
        "scope": "cross-year teacher observation reuse; does not test whole-tile exposure or score models",
        "coordinate_crs": "EPSG:3006", "distance_unit": "metres",
        "interpretation": "shared IDs do not establish shared sites; proximity is descriptive, not an independence verdict",
        "historical_teacher_lineage": "recorded split only; historical heads lack exact split digests",
        "all_models": {"recorded_feature_pool": compare_locations(holdout, pool),
                       "recorded_training_split": compare_locations(holdout, train)},
        "per_model": per_model,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--manifest-sha256", required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.manifest, args.manifest_sha256), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
