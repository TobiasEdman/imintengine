"""Identity, pairing and provenance shared by field-observation evaluations."""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

NFI_KEY = ["TractID", "PlotID", "Year"]
LUCAS_KEY = ["point_id", "Year"]


NFI_MODELS = ("clay", "croma", "prithvi300m", "prithvi300m4f",
              "prithvi600m", "terramind", "tessera")
NFI_CELLS = frozenset(f"{model}_r{rung}" for model in NFI_MODELS for rung in range(1, 5))
NFI_SOURCE_FILES = frozenset({
    "imint/eval/fieldtruth.py", "scripts/prepare_nfi_holdout.py",
    "scripts/nfi_checkpoint_inputs.py", "scripts/nfi_nmd_coverage.py",
    "scripts/validate_against_nfi.py",
    "scripts/inference_comparison.py", "scripts/compare_nmd2023_nfi.py",
    "scripts/score_nfi_holdout.py", "scripts/race_rigor_stats.py",
    "imint/training/unified_dataset.py", "imint/training/errors.py",
})
NFI_YEAR_SAMPLING = {"method": "sha256_plot_year_rank_v1", "seed": 20261008}

NFI_PROTOCOL = {
    "truth_dominant_fraction": 0.7, "classes": [0, 1, 2, 3, 4],
    "primary_head": "class", "bootstrap_seed": 20260818,
    "bootstrap_samples": 10000, "block_km": 50, "sesoi": 0.02,
    "block_signflip_method": "exact_integer_sum_distribution",
    "block_assignment": "50km_grid_at_frozen_tract_year_centroid",
    "teacher_exclusion": "all_recorded_feature_plot_years",
}


def validate_freeze_structure(manifest: dict) -> None:
    """Reject incomplete freezes even when their hash was supplied explicitly."""
    if manifest.get("schema") != "nfi-fieldtruth-freeze-v1":
        raise ValueError("not a supported frozen NFI manifest")
    if (manifest.get("identity") != NFI_KEY
            or set(manifest.get("cells", {})) != NFI_CELLS
            or "NMD2023" not in manifest.get("baselines", {})
            or manifest.get("protocol", {}).get("primary_population") not in {"all", "campaign", "balanced"}
            or not NFI_SOURCE_FILES <= set(manifest.get("source_sha256", {}))
            or any(manifest.get("protocol", {}).get(k) != v for k, v in NFI_PROTOCOL.items())):
        raise ValueError("incomplete frozen NFI identity, roster, source or protocol")
    validate_year_selection(manifest["protocol"])
    source = manifest["source_sha256"]
    if any(not isinstance(h, str) or re.fullmatch(r"[0-9a-f]{64}", h) is None
           for h in source.values()):
        raise ValueError("invalid frozen source digest")
    runtime = manifest.get("preparation_runtime", {})
    if (runtime.get("source", {}).get("git_sha") != manifest.get("git_sha")
            or not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("git_sha", "")))
            or runtime.get("image", {}).get("ref") != manifest.get("runtime_image")
            or not re.fullmatch(r".+@sha256:[0-9a-f]{64}", str(manifest.get("runtime_image", "")))
            or not runtime.get("runtime_manifest", {}).get("sha256")
            or not runtime.get("source", {}).get("payload_sha256")):
        raise ValueError("incomplete frozen runtime identity")


def validate_year_selection(protocol: dict) -> dict | None:
    """Require an explicit, prespecified quota for the balanced population."""
    if protocol.get("primary_population") != "balanced":
        return None
    selection = protocol.get("year_selection")
    if not isinstance(selection, dict):
        raise ValueError("balanced population requires an explicit year selection")
    years = selection.get("years")
    count = selection.get("observations_per_year")
    if (not isinstance(years, list) or len(years) < 2
            or any(type(y) is not int or y < 1 for y in years)
            or years != sorted(set(years))
            or type(count) is not int or count < 1
            or type(selection.get("seed")) is not int
            or any(selection.get(k) != v for k, v in NFI_YEAR_SAMPLING.items())):
        raise ValueError("invalid prespecified balanced year selection")
    return selection


def validate_year_balance(frame: pd.DataFrame, protocol: dict) -> None:
    """No consumer may silently change the approved equal annual counts."""
    selection = validate_year_selection(protocol)
    if selection is None:
        return
    keys = observation_keys(frame)
    if keys.has_duplicates:
        raise ValueError("balanced population contains duplicate plot-years")
    actual = {int(y): int(n) for y, n in frame["Year"].value_counts().items()}
    expected = dict.fromkeys(selection["years"], selection["observations_per_year"])
    if actual != expected:
        raise ValueError(f"year balance differs from the frozen target: {actual}; expected {expected}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if frame[columns].isna().any().any():
        raise ValueError(f"null identity or value in {columns}")


def observation_keys(frame: pd.DataFrame, key: list[str] = NFI_KEY) -> pd.MultiIndex:
    require_columns(frame, key)
    canonical = frame[key].copy()
    for name in key:
        if name in NFI_KEY or name == "point_id":
            if not pd.api.types.is_numeric_dtype(frame[name].dtype):
                raise ValueError(f"non-numeric observation identity dtype: {name}")
            values = frame[name]
            if ((values % 1) != 0).any():
                raise ValueError(f"non-integer observation identity: {name}")
            canonical[name] = values.astype("int64")
    return pd.MultiIndex.from_frame(canonical)


def shared_observations(
    dumps: dict[str, pd.DataFrame], key: list[str], truth: str, prediction: str,
) -> dict[str, pd.DataFrame]:
    """Choose one COMMON tile per observation, independent of predictions.

    Missing years/tiles and conflicting truths fail closed. Choosing each
    cell's first tile before intersecting can compare different ground.
    """
    if not dumps:
        return {}
    pair_key = key + ["tile_name"]
    for cell, frame in dumps.items():
        require_columns(frame, pair_key + [truth, prediction])
        observation_keys(frame, key)
        for column in [truth, prediction]:
            if (frame.groupby(pair_key)[column].nunique() > 1).any():
                raise ValueError(f"{cell}: conflicting {column} for observation/tile")
    all_truth = pd.concat([d[key + [truth]] for d in dumps.values()])
    if (all_truth.groupby(key)[truth].nunique() > 1).any():
        raise ValueError(f"conflicting {truth} for the same observation")
    common = None
    for frame in dumps.values():
        pairs = frame[pair_key].drop_duplicates()
        common = pairs if common is None else common.merge(
            pairs, on=pair_key, how="inner", validate="one_to_one")
    chosen = common.sort_values(pair_key).drop_duplicates(key)
    if chosen.empty:
        raise ValueError("no common observation/tile support")
    return {
        cell: chosen.merge(frame.drop_duplicates(pair_key), on=pair_key,
                           validate="one_to_one").sort_values(key).reset_index(drop=True)
        for cell, frame in dumps.items()
    }


def exclude_training_observations(
    candidates: pd.DataFrame, training: pd.DataFrame,
) -> pd.DataFrame:
    """Exclude plot-years across ALL tiles, preserving different survey years."""
    training_keys = observation_keys(training).unique()
    keep = ~observation_keys(candidates).isin(training_keys)
    result = candidates.loc[keep].copy()
    assert not observation_keys(result).isin(training_keys).any()
    return result


def resolve_observation_year(features: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Recover missing teacher years only through an unambiguous index join."""
    pair = ["TractID", "PlotID", "tile_name"]
    require_columns(features, pair)
    require_columns(index, pair + ["Year"])
    observation_keys(index)
    observation_keys(features, ["TractID", "PlotID"])
    if "Year" in features:
        require_columns(features, NFI_KEY)
        if not observation_keys(features, pair + ["Year"]).isin(
            observation_keys(index, pair + ["Year"])).all():
            raise ValueError("teacher observations absent from the source index")
        return features.copy()
    identities = index[pair + ["Year"]].drop_duplicates()
    needed = identities.merge(features[pair].drop_duplicates(), on=pair)
    if needed.duplicated(pair).any():
        raise ValueError("ambiguous Year for teacher plot/tile; explicit provenance required")
    resolved = features.merge(needed, on=pair, how="left", validate="many_to_one")
    require_columns(resolved, NFI_KEY)
    resolved["Year"] = resolved["Year"].astype(int)
    return resolved


def load_frozen_holdout(manifest_path: str | Path, expected_manifest_sha256: str) -> tuple[pd.DataFrame, dict]:
    path = Path(manifest_path)
    manifest_bytes = path.read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if (not isinstance(expected_manifest_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None
            or digest != expected_manifest_sha256):
        raise ValueError("frozen manifest differs from the approved SHA256")
    manifest = json.loads(manifest_bytes)
    manifest["_manifest_sha256"] = digest
    validate_freeze_structure(manifest)
    table_path = path.parent / manifest["holdout"]["file"]
    holdout = read_verified_parquet(table_path, manifest["holdout"]["sha256"])
    require_columns(holdout, NFI_KEY + ["tile_name"])
    if holdout.empty or holdout.duplicated(NFI_KEY).any():
        raise ValueError("frozen holdout must contain unique, nonempty plot-years")
    if len(holdout) != manifest["holdout"]["observations"]:
        raise ValueError("frozen holdout count mismatch")
    validate_year_balance(holdout, manifest["protocol"])
    training_path = path.parent / manifest["training"]["file"]
    training = read_verified_parquet(training_path, manifest["training"]["sha256"])
    if (training.duplicated(NFI_KEY).any()
            or len(training) != manifest["training"]["observations"]):
        raise ValueError("frozen training identity count mismatch")
    if observation_keys(holdout).isin(observation_keys(training)).any():
        raise ValueError("training observation leaked into frozen holdout")
    if manifest.get("campaign", {}).get("training_tiles") != 0:
        raise ValueError("campaign staging entered training")
    return holdout, manifest


def restrict_to_frozen(dump: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    """Require every frozen observation exactly once on its selected tile."""
    pair = NFI_KEY + ["tile_name"]
    require_columns(dump, pair)
    if dump.duplicated(pair).any():
        raise ValueError("duplicate prediction for frozen observation/tile")
    selected = holdout[pair].merge(dump, on=pair, how="left",
                                   validate="one_to_one", indicator=True)
    if (selected["_merge"] != "both").any():
        raise ValueError("prediction dump lacks frozen observation/tile support")
    return selected.drop(columns="_merge")


class NoScoredObservations(SystemExit):
    """A failed evaluation with a serializable coverage report."""

    def __init__(self, unit: str, skipped: list[dict]):
        super().__init__(f"no {unit} could be scored")
        self.report = {"status": "no_scored_observations",
                       "skipped_tiles": skipped, "observations_scored": 0}


def verify_file_identity(record: dict) -> None:
    path = Path(record["path"])
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"frozen input content changed: {path}")


def read_verified_parquet(path: Path, expected_sha256: str) -> pd.DataFrame:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"frozen parquet content changed: {path}")
    return pd.read_parquet(io.BytesIO(data))


def verify_prediction_dump(
    path: Path, manifest_path: Path, cell: str, *, manifest: dict | None = None,
    expected_manifest_sha256: str | None = None,
) -> pd.DataFrame:
    """Verify and parse the SAME prediction bytes under the frozen run."""
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    if manifest is None:
        _, manifest = load_frozen_holdout(manifest_path, expected_manifest_sha256)
    if (metadata.get("cell") != cell
            or metadata.get("holdout_manifest_sha256") != manifest["_manifest_sha256"]
            or metadata.get("checkpoint_sha256") != manifest["cells"][cell]["checkpoint"]["sha256"]
            or any(metadata.get(k) != v for k, v in evaluation_runtime_identity(manifest).items())
            or metadata.get("status") != "success"):
        raise ValueError("prediction dump provenance does not match frozen run")
    frame = read_verified_parquet(path, metadata["prediction_sha256"])
    frame.attrs["authenticated_sha256"] = metadata["prediction_sha256"]
    return frame


def evaluation_runtime_identity(manifest: dict) -> dict:
    runtime = manifest["preparation_runtime"]
    return {"runtime_image": manifest["runtime_image"],
            "source_git_sha": manifest["git_sha"],
            "source_payload_sha256": runtime["source"]["payload_sha256"],
            "runtime_manifest_sha256": runtime["runtime_manifest"]["sha256"]}


def verify_evaluation_source(manifest: dict, *, environment: str = "model") -> dict:
    """Reauthenticate the full sealed source and execution environment."""
    from scripts.crop_distill_provenance import verify_runtime, snapshot_tree, tree_payload_sha256

    validate_freeze_structure(manifest)
    root = Path(__file__).resolve().parents[2]
    for relative, expected in manifest["source_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"evaluation code differs from frozen source: {relative}")
    frozen = manifest["preparation_runtime"]
    runtime = verify_runtime(Path(frozen["runtime_manifest"]["path"]),
                             source_git_sha=manifest["git_sha"],
                             image_ref=manifest["runtime_image"])
    if runtime != frozen:
        raise ValueError("execution runtime differs from frozen runtime")
    if tree_payload_sha256(snapshot_tree(root)) != frozen["source"]["payload_sha256"]:
        raise ValueError("execution source tree differs from frozen source")
    expected_python = runtime["environments"][environment]["python"]["path"]
    if Path(sys.executable).absolute() != Path(expected_python).absolute():
        raise ValueError(f"evaluation requires the verified {environment} interpreter")
    return evaluation_runtime_identity(manifest)
