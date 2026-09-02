"""Immutable protocol constants for the LUCAS crop-distill evidence jobs.

This module is part of the runtime-image payload.  Kubernetes manifests may
select one of the six model keys, but every behavioural value is resolved here
from the source SHA baked into that image.  Keeping the map in Commit A closes
the gap where a later manifest-only commit could otherwise change an image's
checkpoint, crop grid, modality requirements, or scoring protocol.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

SOURCE_ROOT = Path("/opt/imintengine")
BASE_PYTHON = Path("/usr/local/bin/python")
MODEL_PYTHON = Path("/opt/venvs/model/bin/python")
SCORING_PYTHON = Path("/opt/venvs/scoring/bin/python")
RUNTIME_MANIFEST = Path("/opt/provenance/runtime.json")
WORK_ROOT = Path("/work")

PROVENANCE_SCRIPT = SOURCE_ROOT / "scripts/crop_distill_provenance.py"
SPLIT_SCRIPT = SOURCE_ROOT / "scripts/build_lucas_crop_split.py"
EXTRACT_SCRIPT = SOURCE_ROOT / "scripts/extract_plot_features.py"
SCORE_SCRIPT = SOURCE_ROOT / "scripts/nfi_head_cv.py"

PVC_ROOT = Path("/cephfs")
DATA_DIR = PVC_ROOT / "unified_v2_512"
LUCAS_SOURCE_INDEX = Path("/cephfs/lucas/lucas_tile_index.parquet")
DISTILL_DIR = Path("/cephfs/distill/crop_split")
CROP_INDEX = DISTILL_DIR / "lucas_crop_distill_index.parquet"
CROP_SPLIT = DISTILL_DIR / "lucas_crop_split.json"
CROP_SPLIT_MANIFEST = DISTILL_DIR / "lucas_crop_split.MANIFEST.json"

# Storage prep creates the exact backing roots. Split and crop Pods then see
# only their dedicated read-only or narrowly writable subPath projections;
# neither workload mounts the full PVC root.
CROP_HEADS_DIR = Path("/cephfs/crop-heads")
CROP_RECORD_DIR = Path("/cephfs/crop-records")
SPLIT_RECORD_DIR = Path("/cephfs/ops/crop-distill")
CROP_HEADS_BACKING_ROOT = Path("/cephfs/distill/crop_heads")
CROP_RECORDS_BACKING_ROOT = Path("/cephfs/ops/crop-distill")
SPLIT_RECORD_BACKING_DIR = CROP_RECORDS_BACKING_ROOT / "split"

STORAGE_UID = 2000
STORAGE_GID = 2000
STORAGE_MODE = 0o3770
STORAGE_PARENT_MODE = 0o750
STORAGE_LEAF_MODE = 0o750
FROZEN_SPLIT_MODE = 0o550

TRUTH_COLUMN = "unified_class"
OOF_FOLDS = 5
OOF_HEADS = "mlp"
OOF_GROUP_COLUMN = "tile_name"
DEVICE = "cuda"


@dataclass(frozen=True, slots=True)
class CropModelProtocol:
    """All model-varying behaviour permitted in a crop-distill job."""

    img_size: int
    backbone: str
    required_npz_keys: tuple[str, ...]
    checkpoint_path: Path
    checkpoint_name: str
    checkpoint_size: int
    checkpoint_sha256: str


def _model(
    key: str,
    *,
    img_size: int,
    backbone: str,
    required_npz_keys: tuple[str, ...],
    checkpoint_size: int,
    checkpoint_sha256: str,
) -> CropModelProtocol:
    checkpoint_name = "best_model.pt"
    checkpoint_path = Path("/cephfs/checkpoints/ladder") / f"{key}_r2" / checkpoint_name
    return CropModelProtocol(
        img_size=img_size,
        backbone=backbone,
        required_npz_keys=required_npz_keys,
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_name,
        checkpoint_size=checkpoint_size,
        checkpoint_sha256=checkpoint_sha256,
    )


CROP_MODELS = {
    "clay": _model(
        "clay",
        img_size=504,
        backbone="clay_v1_5",
        required_npz_keys=(),
        checkpoint_size=2_601_012_332,
        checkpoint_sha256=(
            "0a37ebdbbae8ac61145424350ae8f2990225d2cb15a3e1c178c9d42134c226e2"
        ),
    ),
    "croma": _model(
        "croma",
        img_size=504,
        backbone="croma_base",
        required_npz_keys=("s1_vv_vh",),
        checkpoint_size=834_654_805,
        checkpoint_sha256=(
            "dbfc04cf9475ca6b604dd5133191854736e961deebeb992be855f211d152bd80"
        ),
    ),
    "prithvi300m": _model(
        "prithvi300m",
        img_size=496,
        backbone="prithvi_300m",
        required_npz_keys=(),
        checkpoint_size=1_285_893_675,
        checkpoint_sha256=(
            "a27dadd9caf1c9ccfba6ecbd76ac7815fcb7236978e9df807e1d1bf7a498cda0"
        ),
    ),
    "prithvi600m": _model(
        "prithvi600m",
        img_size=504,
        backbone="prithvi_600m",
        required_npz_keys=(),
        checkpoint_size=2_741_619_081,
        checkpoint_sha256=(
            "89d544c06fd353772722dec5600a4ba8696fd8971250f471b47f6b53828d1d46"
        ),
    ),
    "terramind": _model(
        "terramind",
        img_size=496,
        backbone="terramind_v1_base",
        required_npz_keys=("s1_vv_vh",),
        checkpoint_size=401_358_843,
        checkpoint_sha256=(
            "97316cf22612288072f0278f5c90e1a987a845a35acb1dcb431cc13432b4fc8f"
        ),
    ),
    "tessera": _model(
        "tessera",
        img_size=504,
        backbone="tessera_v1",
        required_npz_keys=("tessera",),
        checkpoint_size=1_596_322,
        checkpoint_sha256=(
            "9dd7cfcad09b26576d23c846c29c3fd540d463b97a72df0b7557f6558dbced04"
        ),
    ),
}

MODEL_KEYS = tuple(sorted(CROP_MODELS))

# Distinct UIDs stop one model column from rewriting another model's 0750
# publication directory. GID 2000 grants read/traverse on the root-owned 0750
# parents, but only the owning UID can create inside its pre-provisioned 0750
# leaf. This map is runtime payload, not a manifest-only choice. A seventh
# model needs a reviewed UID and an intentional range extension.
CROP_MODEL_UIDS = {
    "clay": 2001,
    "croma": 2002,
    "prithvi300m": 2003,
    "prithvi600m": 2004,
    "terramind": 2005,
    "tessera": 2006,
}


@dataclass(frozen=True, slots=True)
class StorageTarget:
    """One exact backing directory storage prep is allowed to create/change."""

    path: Path
    uid: int
    gid: int
    mode: int
    preserve_mode: int | None = None


def crop_head_backing_dir(model: str) -> Path:
    """Return the one PVC head directory owned by a model UID."""
    model_protocol(model)
    return CROP_HEADS_BACKING_ROOT / f"{model}_r2_crop_runs"


def crop_record_backing_dir(model: str) -> Path:
    """Return the one PVC evidence directory owned by a model UID."""
    model_protocol(model)
    return CROP_RECORDS_BACKING_ROOT / model


_SOURCE_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_RE = re.compile(
    r"^ghcr\.io/tobiasedman/imint-ladder-crop-distill@sha256:"
    r"[0-9a-f]{64}$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POD_UID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")
MISSING_IDENTITY_CLAIM = "<missing>"


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """The three manifest-supplied values allowed to reach an entrypoint."""

    source_git_sha: str
    image_ref: str
    pod_uid: str


def model_protocol(model: str) -> CropModelProtocol:
    """Resolve one exact model protocol; unknown keys fail closed."""
    try:
        return CROP_MODELS[model]
    except KeyError as exc:
        choices = ", ".join(MODEL_KEYS)
        raise ValueError(
            f"unknown crop-distill model {model!r}; choose {choices}"
        ) from exc


def validate_model_uid_map(mapping: Mapping[str, int]) -> None:
    """Validate the complete collision-free model process-identity map."""
    if set(mapping) != set(CROP_MODELS):
        raise ValueError("model UID map must declare exactly the crop model keys")
    values = tuple(mapping.values())
    if any(type(uid) is not int or not 2001 <= uid <= 2006 for uid in values):
        raise ValueError("model UIDs must be integers in the reviewed range 2001..2006")
    if len(set(values)) != len(values):
        raise ValueError("model UIDs must be unique")


def model_process_uid(model: str) -> int:
    """Return one model's reviewed UID after validating the whole map."""
    validate_model_uid_map(CROP_MODEL_UIDS)
    try:
        return CROP_MODEL_UIDS[model]
    except KeyError as exc:
        raise ValueError(f"no process UID for crop-distill model {model!r}") from exc


# Construct the immutable storage layout only after both model lookup and UID
# validation are defined.  Keeping this below ``model_process_uid`` also makes
# a malformed or colliding model/UID map fail at import rather than creating
# an ambiguous ownership plan for the privileged storage-prep entrypoint.
STORAGE_TARGETS = (
    StorageTarget(
        DISTILL_DIR,
        STORAGE_UID,
        STORAGE_GID,
        STORAGE_MODE,
        preserve_mode=FROZEN_SPLIT_MODE,
    ),
    StorageTarget(
        CROP_HEADS_BACKING_ROOT,
        0,
        STORAGE_GID,
        STORAGE_PARENT_MODE,
    ),
    StorageTarget(
        CROP_RECORDS_BACKING_ROOT,
        0,
        STORAGE_GID,
        STORAGE_PARENT_MODE,
    ),
    StorageTarget(
        SPLIT_RECORD_BACKING_DIR,
        STORAGE_UID,
        STORAGE_GID,
        STORAGE_LEAF_MODE,
    ),
    *(
        StorageTarget(
            crop_head_backing_dir(model),
            model_process_uid(model),
            STORAGE_GID,
            STORAGE_LEAF_MODE,
        )
        for model in MODEL_KEYS
    ),
    *(
        StorageTarget(
            crop_record_backing_dir(model),
            model_process_uid(model),
            STORAGE_GID,
            STORAGE_LEAF_MODE,
        )
        for model in MODEL_KEYS
    ),
)


def require_process_identity(
    expected_uid: int,
    *,
    expected_gid: int = STORAGE_GID,
    role: str,
) -> None:
    """Fail before PVC work if the applied Pod identity drifted from review."""
    actual_uid = os.geteuid()
    actual_gid = os.getegid()
    if (actual_uid, actual_gid) != (expected_uid, expected_gid):
        raise RuntimeError(
            f"{role} requires effective UID:GID {expected_uid}:{expected_gid}; "
            f"running as {actual_uid}:{actual_gid}"
        )


def runtime_identity(environ: Mapping[str, str]) -> RuntimeIdentity:
    """Read and validate the required identity/Downward-API environment."""
    identity = runtime_claims(environ)
    source_git_sha = identity.source_git_sha
    image_ref = identity.image_ref
    if _SOURCE_SHA_RE.fullmatch(source_git_sha) is None or source_git_sha == "0" * 40:
        raise ValueError(
            "CROP_DISTILL_SOURCE_GIT_SHA must be one nonzero lowercase 40-hex SHA"
        )
    if _IMAGE_RE.fullmatch(image_ref) is None or image_ref.endswith("0" * 64):
        raise ValueError(
            "CROP_DISTILL_IMAGE must be the immutable crop-distill image digest"
        )
    return identity


def runtime_claims(environ: Mapping[str, str]) -> RuntimeIdentity:
    """Preserve raw source/image claims once a safe record path is known.

    Failure provenance intentionally receives malformed or missing claims so
    the terminal record explains an identity bootstrap failure.  POD_UID is
    different: it selects the write-once record path and therefore must be
    present and path-safe before a job object can be constructed.
    """
    pod_uid = environ.get("POD_UID", "")
    if not pod_uid:
        raise ValueError("missing required environment variable: POD_UID")
    if _POD_UID_RE.fullmatch(pod_uid) is None:
        raise ValueError("POD_UID contains unsafe path characters")
    return RuntimeIdentity(
        environ.get("CROP_DISTILL_SOURCE_GIT_SHA") or MISSING_IDENTITY_CLAIM,
        environ.get("CROP_DISTILL_IMAGE") or MISSING_IDENTITY_CLAIM,
        pod_uid,
    )


def split_manifest_claim(environ: Mapping[str, str]) -> str:
    """Preserve the Git-reviewed split-manifest identity for diagnostics."""
    return environ.get("CROP_DISTILL_SPLIT_MANIFEST_SHA256") or MISSING_IDENTITY_CLAIM


def require_split_manifest_sha256(value: str) -> str:
    """Validate the off-PVC trust anchor supplied by the reviewed manifest."""
    if _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        raise ValueError(
            "CROP_DISTILL_SPLIT_MANIFEST_SHA256 must be one nonzero "
            "lowercase 64-hex digest"
        )
    return value
