"""Identity, pairing and provenance shared by field-observation evaluations."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd

NFI_KEY = ["TractID", "PlotID", "Year"]
LUCAS_KEY = ["point_id", "Year"]


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


def load_frozen_holdout(manifest_path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(manifest_path)
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest["_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest.get("schema") != "nfi-fieldtruth-freeze-v1":
        raise ValueError("not a supported frozen NFI manifest")
    table_path = path.parent / manifest["holdout"]["file"]
    holdout = read_verified_parquet(table_path, manifest["holdout"]["sha256"])
    require_columns(holdout, NFI_KEY + ["tile_name"])
    if holdout.empty or holdout.duplicated(NFI_KEY).any():
        raise ValueError("frozen holdout must contain unique, nonempty plot-years")
    if len(holdout) != manifest["holdout"]["observations"]:
        raise ValueError("frozen holdout count mismatch")
    training_path = path.parent / manifest["training"]["file"]
    training = read_verified_parquet(training_path, manifest["training"]["sha256"])
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
) -> pd.DataFrame:
    """Verify and parse the SAME prediction bytes under the frozen run."""
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    if manifest is None:
        _, manifest = load_frozen_holdout(manifest_path)
    if (metadata.get("cell") != cell
            or metadata.get("holdout_manifest_sha256") != manifest["_manifest_sha256"]
            or metadata.get("checkpoint_sha256") != manifest["cells"][cell]["checkpoint"]["sha256"]
            or metadata.get("status") != "success"):
        raise ValueError("prediction dump provenance does not match frozen run")
    return read_verified_parquet(path, metadata["prediction_sha256"])


def verify_evaluation_source(manifest: dict) -> None:
    root = Path(__file__).resolve().parents[2]
    for relative, expected in manifest["source_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"evaluation code differs from frozen source: {relative}")
