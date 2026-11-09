"""IM-023 amendment to the 2026-10-27 NFI freeze: protocol, binding, runtime.

Tobias 2026-11-06 (docs/experiments/class-separability/retrain_barrblandskog.md,
step 3): the freeze gets an amendment, not a change. The amendment has its
own protocol id, manifest, cells and code pin. It references the frozen
holdout by the original manifest's digest, pinned here, and re-verifies that
manifest, its parquets and its tiles through ``imint.eval.fieldtruth``. The
original freeze, its cells, protocol and code pin are never rewritten; its
own consumers keep running from its own sealed image.

The base's code pin (4cb93b4) cannot run the amendment, so the amendment
binds its own code pin in the same sealed image (``docker/ladder-crop-distill``)
rebuilt at the amendment's commit: same base, same locked environments, new
source. ``sealed_runtime`` authenticates that image as the freeze does
(``prepare_nfi_holdout.preparation_runtime``,
``fieldtruth.verify_evaluation_source``); every consumer must match what the
prepare step recorded. The record is ``verify_runtime``'s, so
``fieldtruth.evaluation_runtime_identity`` and
``fieldtruth.verify_prediction_dump`` apply to amendment dumps unchanged.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from imint.crop_distill_runtime import snapshot_tree, tree_payload_sha256, verify_runtime
from imint.eval.fieldtruth import (
    NFI_KEY, NFI_SOURCE_FILES, evaluation_runtime_identity, load_frozen_holdout,
    sha256_file, verify_prediction_dump,
)

AMENDMENT_SCHEMA = "nfi-fieldtruth-amendment-v1"
AMENDMENT_PROTOCOL_ID = "nfi-freeze-2026-10-27-amendment-im023-v1"

#: The approved digest of the freeze this amendment extends (k8s/nfi-scoring-job.yaml).
BASE_MANIFEST_SHA256 = "daaa428754aa4acfd832d9cc0fe5a0acb60e86ade2b511699f1181732697ff2c"
#: The base cell whose checkpoint the amendment's r4 cell must be.
BASE_R4_CELL = "tessera_r4"

#: Cell -> number of classes of its hard head.
AMENDMENT_CELLS = {"im023_main": 29, "im023_control": 28, "r4": 28}
CELL_HEAD = "class_and_fraction"
FRACTION_COLUMNS = ["frac_tall", "frac_gran", "frac_trivial", "frac_adel"]

TALLSKOG, GRANSKOG, LOVSKOG, BLANDSKOG, BARRBLANDSKOG = 1, 2, 3, 4, 28

#: Fixed before any amendment result exists; the prepare step copies it into
#: the manifest and every consumer requires it verbatim.
AMENDMENT_PROTOCOL = {
    "truth": "frozen_nfi_forest_v1",
    "truth_dominant_fraction": 0.7,
    "head": CELL_HEAD,
    "fraction_order": ["tall", "gran", "trivial", "adel"],
    "primary_rule": "class_head_then_28_to_tallskog_if_tall_ge_gran_else_granskog",
    "share_rule_dominant_fraction": 0.7,
    "share_rule_species_fraction": 0.7,
    "forest_floor": 0.1,
    # Barrblandskog truth: conifer dominance is truth v1's own (class 1 or 2
    # at truth_dominant_fraction), then this split of the conifer volume.
    "barrblandskog_truth": "v1_conifer_then_species_split",
    "barrblandskog_species_fraction": 0.7,
    "contorta_sensitivity_fraction": 0.3,
    "bootstrap_seed": 20260818,
    "bootstrap_samples": 10000,
    "block_km": 50,
    "block_assignment": "50km_grid_at_frozen_tract_year_centroid",
    "sesoi": 0.02,
    # Bounds of the paired ΔF1 compared with -SESOI: the 5 % and 95 %
    # bootstrap quantiles, the one-sided α = 0.05 the NFI protocol uses for
    # its SESOI (equivalence) checks (Tobias 2026-11-07).
    "condition_bound_quantiles": [0.05, 0.95],
    # A bound left undefined by an undefined-F1 draw does not hold:
    # "annars behålls klassen" (Tobias 2026-11-07).
    "undefined_bound": "condition_not_met",
}
NUM_FRACTIONS = len(AMENDMENT_PROTOCOL["fraction_order"])

AMENDMENT_SOURCE_FILES = NFI_SOURCE_FILES | {
    "imint/eval/nfi_amendment.py", "scripts/prepare_nfi_amendment.py",
    "scripts/nfi_amendment_checkpoints.py",
    "scripts/infer_nfi_amendment.py", "scripts/score_nfi_amendment.py",
}

_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_IMAGE_REF = re.compile(r".+@sha256:[0-9a-f]{64}")


def _matches(pattern: re.Pattern, value) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def dump_name(cell: str) -> str:
    return f"nfi-amendment-per-plot-{cell}.parquet"


# ── rules ────────────────────────────────────────────────────────────────────

def fold_barrblandskog(pred, tall, gran) -> np.ndarray:
    """Primary rule: a predicted 28 becomes tallskog if tall >= gran, else granskog.

    The tie goes to tallskog, as in ``collapse_fractions_to_nfi_class``.
    Every other prediction is unchanged.
    """
    pred, tall, gran = (np.asarray(v) for v in (pred, tall, gran))
    if not pred.shape == tall.shape == gran.shape:
        raise ValueError("predictions and fraction shares differ in shape")
    folded = pred == BARRBLANDSKOG
    if not np.isfinite(tall[folded]).all() or not np.isfinite(gran[folded]).all():
        raise ValueError("a folded prediction needs finite tall and gran shares")
    return np.where(folded, np.where(tall >= gran, TALLSKOG, GRANSKOG), pred)


def _species_volumes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    names = ("VolPine", "VolContorta", "VolSpruce", "VolBirch", "VolOtherDec")
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"missing NFI species volumes: {sorted(missing)}")
    volumes = {name: frame[name].to_numpy(dtype=float) for name in names}
    stacked = np.vstack(list(volumes.values()))
    if not np.isfinite(stacked).all() or (stacked < 0).any():
        raise ValueError("NFI species volumes must be finite and nonnegative")
    return volumes


def total_volume(frame: pd.DataFrame) -> np.ndarray:
    return sum(_species_volumes(frame).values())


def derive_barrblandskog(frame: pd.DataFrame, *, species_fraction: float) -> np.ndarray:
    """Plots truth v1 calls conifer-dominated (tallskog or granskog) where
    neither pine (VolPine + VolContorta) nor spruce reaches
    ``species_fraction`` of the conifer volume.

    Dominance is read from ``nfi_forest``, never recomputed: a second sum of
    the same volumes can round to the other side of 0.7 and disagree with v1.
    """
    v = _species_volumes(frame)
    pine = v["VolPine"] + v["VolContorta"]
    conifer = pine + v["VolSpruce"]
    dominant = np.isin(frame["nfi_forest"].to_numpy(dtype=int), (TALLSKOG, GRANSKOG))
    with np.errstate(divide="ignore", invalid="ignore"):
        mixed = (pine / conifer < species_fraction) & (v["VolSpruce"] / conifer < species_fraction)
    return dominant & mixed


def contorta_share(frame: pd.DataFrame) -> np.ndarray:
    """VolContorta / (VolPine + VolContorta); 0 where the plot has no pine."""
    v = _species_volumes(frame)
    pine = v["VolPine"] + v["VolContorta"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pine > 0, v["VolContorta"] / pine, 0.0)


# ── manifest ─────────────────────────────────────────────────────────────────

def validate_amendment_structure(amendment: dict) -> None:
    """Reject an incomplete amendment even when its digest was approved."""
    base = amendment.get("base", {})
    cells = amendment.get("cells", {})
    if (amendment.get("schema") != AMENDMENT_SCHEMA
            or amendment.get("protocol_id") != AMENDMENT_PROTOCOL_ID
            or amendment.get("identity") != NFI_KEY
            or amendment.get("protocol") != AMENDMENT_PROTOCOL
            or base.get("manifest_sha256") != BASE_MANIFEST_SHA256
            or not isinstance(base.get("manifest_path"), str)
            or not _matches(_HEX64, base.get("holdout_sha256"))
            or type(base.get("holdout_observations")) is not int
            or set(cells) != set(AMENDMENT_CELLS)
            or not AMENDMENT_SOURCE_FILES <= set(amendment.get("source_sha256", {}))):
        raise ValueError("incomplete amendment identity, base, roster, source or protocol")
    for name, cell in cells.items():
        checkpoint = cell.get("checkpoint", {})
        if (cell.get("num_classes") != AMENDMENT_CELLS[name] or cell.get("head") != CELL_HEAD
                or not isinstance(cell.get("backbone"), str) or type(cell.get("img_size")) is not int
                or not _matches(_HEX64, checkpoint.get("sha256"))
                or type(checkpoint.get("bytes")) is not int
                or not isinstance(cell.get("input_contract"), dict)):
            raise ValueError(f"incomplete amendment cell: {name}")
    if len({cell["checkpoint"]["sha256"] for cell in cells.values()}) != len(cells):
        raise ValueError("amendment cells must pin distinct checkpoints")
    if any(not _matches(_HEX64, h) for h in amendment["source_sha256"].values()):
        raise ValueError("invalid amendment source digest")
    runtime = amendment.get("preparation_runtime", {})
    if (not _matches(_HEX40, amendment.get("git_sha"))
            or runtime.get("source", {}).get("git_sha") != amendment["git_sha"]
            or not _matches(_HEX64, runtime.get("source", {}).get("payload_sha256"))
            or not _matches(_IMAGE_REF, amendment.get("runtime_image"))
            or runtime.get("image", {}).get("ref") != amendment["runtime_image"]
            or not _matches(_HEX64, runtime.get("runtime_manifest", {}).get("sha256"))
            or not isinstance(runtime.get("runtime_manifest", {}).get("path"), str)):
        raise ValueError("incomplete amendment runtime identity")


def bind_base(amendment: dict) -> tuple[pd.DataFrame, dict]:
    """Load the frozen holdout through the original freeze's own verification."""
    record = amendment["base"]
    holdout, base = load_frozen_holdout(record["manifest_path"], record["manifest_sha256"])
    if (base["holdout"]["sha256"] != record["holdout_sha256"]
            or base["holdout"]["observations"] != record["holdout_observations"]
            or len(holdout) != record["holdout_observations"]):
        raise ValueError("amendment does not reference the frozen holdout")
    if (amendment["cells"]["r4"]["checkpoint"]["sha256"]
            != base["cells"][BASE_R4_CELL]["checkpoint"]["sha256"]):
        raise ValueError(f"amendment r4 is not the frozen {BASE_R4_CELL} checkpoint")
    return holdout, base


def load_amended_holdout(
    manifest_path: str | Path, expected_manifest_sha256: str,
) -> tuple[pd.DataFrame, dict, dict]:
    """(frozen holdout, amendment, base manifest), or an error."""
    payload = Path(manifest_path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if not _matches(_HEX64, expected_manifest_sha256) or digest != expected_manifest_sha256:
        raise ValueError("amendment manifest differs from the approved SHA256")
    amendment = json.loads(payload)
    amendment["_manifest_sha256"] = digest
    validate_amendment_structure(amendment)
    holdout, base = bind_base(amendment)
    return holdout, amendment, base


def verify_amendment_dump(path: Path, cell: str, amendment: dict) -> pd.DataFrame:
    """The frozen dump verifier, then the amendment's per-plot columns.

    The amendment itself is the manifest the dump binds to, so the verifier
    reads no manifest path.
    """
    frame = verify_prediction_dump(path, None, cell, manifest=amendment)
    metadata = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
    if (metadata.get("prediction_sha256") != frame.attrs["authenticated_sha256"]
            or metadata.get("amendment_protocol_id") != amendment["protocol_id"]
            or metadata.get("base_manifest_sha256") != amendment["base"]["manifest_sha256"]
            or metadata.get("observations") != len(frame)):
        raise ValueError(f"{cell}: dump metadata does not match the amendment")
    columns = NFI_KEY + ["tile_name", "nfi_forest", "model_pred"] + FRACTION_COLUMNS
    missing = set(columns) - set(frame.columns)
    if missing or frame[columns].isna().any().any():
        raise ValueError(f"{cell}: dump lacks complete columns {sorted(missing)}")
    pred = frame["model_pred"].to_numpy()
    if ((pred % 1 != 0).any() or (pred < 0).any()
            or (pred >= amendment["cells"][cell]["num_classes"]).any()):
        raise ValueError(f"{cell}: class prediction outside the cell's head")
    shares = frame[FRACTION_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(shares).all() or (shares < 0).any() or (shares > 1).any():
        raise ValueError(f"{cell}: fraction shares outside [0, 1]")
    return frame


# ── runtime ──────────────────────────────────────────────────────────────────

def sealed_runtime(root: Path, runtime_manifest: Path, *, source_git_sha: str,
                   image_ref: str, environment: str) -> dict:
    """Authenticate the sealed image's source and this step's interpreter, without Git.

    The checks of ``prepare_nfi_holdout.preparation_runtime`` (scoring) and
    ``fieldtruth.verify_evaluation_source`` (model), for either environment.
    The image digest cannot be observed from inside the Pod: it is the
    declared ``image_ref``, and the Pod's ``imageID`` is captured with the run
    evidence, as for the original freeze.
    """
    runtime = verify_runtime(runtime_manifest, source_git_sha=source_git_sha, image_ref=image_ref)
    if tree_payload_sha256(snapshot_tree(root)) != runtime["source"]["payload_sha256"]:
        raise ValueError("this step is not running from the verified source tree")
    expected_python = runtime["environments"][environment]["python"]["path"]
    # Do not resolve venv symlinks: both environments can share a base binary.
    if Path(sys.executable).absolute() != Path(expected_python).absolute():
        raise ValueError(f"this step requires the verified {environment} interpreter")
    return runtime


def verify_amendment_source(amendment: dict, *, environment: str) -> dict:
    """Reauthenticate the amendment's code and sealed runtime in this process."""
    validate_amendment_structure(amendment)
    root = Path(__file__).resolve().parents[2]
    for relative, expected in amendment["source_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"evaluation code differs from the amendment's source: {relative}")
    prepared = amendment["preparation_runtime"]
    runtime = sealed_runtime(root, Path(prepared["runtime_manifest"]["path"]),
                             source_git_sha=amendment["git_sha"],
                             image_ref=amendment["runtime_image"], environment=environment)
    if runtime != prepared:
        raise ValueError("execution runtime differs from the amendment's runtime")
    return evaluation_runtime_identity(amendment)
