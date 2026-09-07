"""Fail-closed runtime and terminal evidence for LUCAS crop distillation."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import crop_distill_provenance as provenance

SOURCE_GIT_SHA = "1" * 40
HISTORICAL_SPLIT_SOURCE_GIT_SHA = "a" * 40
IMAGE_REF = f"ghcr.io/example/crop@sha256:{'2' * 64}"
SOURCE_ACCESS_PLAN_SHA256 = "3" * 64
SOURCE_ACCESS_PLAN_POD_UID = "source-access-plan-pod"
SOURCE_ACCESS_COMPLETION_SHA256 = "4" * 64
SOURCE_ACCESS_COMPLETION_POD_UID = "source-access-completion-pod"
_CHECKPOINT_PAYLOAD = b"checkpoint"


@pytest.fixture(autouse=True)
def completion_authority(tmp_path: Path, monkeypatch):
    """Project production authority paths onto one isolated test root."""
    work_root = tmp_path / "work"
    distill_dir = tmp_path / "distill"
    heads_dir = tmp_path / "crop-heads"
    records_dir = tmp_path / "records"
    split_records_dir = tmp_path / "split-records"
    original_model_protocol = provenance.model_protocol

    def test_model_protocol(model: str):
        protocol = original_model_protocol(model)
        return replace(
            protocol,
            checkpoint_path=(
                tmp_path / "checkpoints" / model / protocol.checkpoint_name
            ),
            checkpoint_size=len(_CHECKPOINT_PAYLOAD),
            checkpoint_sha256=hashlib.sha256(
                _CHECKPOINT_PAYLOAD
            ).hexdigest(),
        )

    monkeypatch.setattr(provenance, "WORK_ROOT", work_root)
    monkeypatch.setattr(provenance, "DISTILL_DIR", distill_dir)
    monkeypatch.setattr(
        provenance,
        "CROP_INDEX",
        distill_dir / "lucas_crop_distill_index.parquet",
    )
    monkeypatch.setattr(
        provenance,
        "CROP_SPLIT",
        distill_dir / "lucas_crop_split.json",
    )
    monkeypatch.setattr(
        provenance,
        "CROP_SPLIT_MANIFEST",
        distill_dir / "lucas_crop_split.MANIFEST.json",
    )
    monkeypatch.setattr(provenance, "CROP_HEADS_DIR", heads_dir)
    monkeypatch.setattr(provenance, "CROP_RECORD_DIR", records_dir)
    monkeypatch.setattr(provenance, "SPLIT_RECORD_DIR", split_records_dir)
    monkeypatch.setattr(provenance, "STORAGE_UID", os.geteuid())
    monkeypatch.setattr(provenance, "STORAGE_GID", os.getegid())
    monkeypatch.setattr(
        provenance,
        "model_process_uid",
        lambda _model: os.geteuid(),
    )
    monkeypatch.setattr(provenance, "model_protocol", test_model_protocol)
    anchor = _split_bundle(tmp_path / "authority-anchor")
    monkeypatch.setenv(
        "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
        str(anchor["manifest_sha256"]),
    )


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_python(path: Path) -> tuple[Path, dict]:
    identity = {
        "implementation": "CPython",
        "path": str(path),
        "version": "3.11.14",
        "version_info": [3, 11, 14],
    }
    payload = json.dumps(identity, sort_keys=True)
    _write(path, f"#!/bin/sh\nprintf '%s\\n' '{payload}'\n".encode())
    path.chmod(0o755)
    return path, identity


def _tree_record(
    root: Path,
    manifest: Path,
    *,
    git_sha: str,
    archive_sha256: str,
) -> dict:
    entries = provenance.snapshot_tree(root)
    payload_sha256 = provenance.tree_payload_sha256(entries)
    manifest.write_bytes(provenance.canonical_json_bytes({
        "schema": provenance.TREE_SCHEMA,
        "entries": entries,
        "payload_sha256": payload_sha256,
    }))
    return {
        "git_sha": git_sha,
        "archive_sha256": archive_sha256,
        "root": str(root),
        "files_manifest": str(manifest),
        "files_manifest_sha256": _sha(manifest),
        "payload_sha256": payload_sha256,
    }


@pytest.fixture
def runtime(tmp_path: Path) -> dict[str, Path | str]:
    source = tmp_path / "source"
    croma = tmp_path / "croma"
    clay = tmp_path / "clay"
    _write(source / "scripts" / "worker.py", b"print('source')\n")
    _write(croma / "use_croma.py", b"class PretrainedCROMA: pass\n")
    _write(clay / "claymodel" / "module.py", b"class ClayMAEModule: pass\n")

    source_record = _tree_record(
        source,
        tmp_path / "source-files.json",
        git_sha=SOURCE_GIT_SHA,
        archive_sha256="3" * 64,
    )
    source_record["selection"] = provenance.SOURCE_ARCHIVE_SELECTION
    croma_record = _tree_record(
        croma,
        tmp_path / "croma-files.json",
        git_sha=provenance.CROMA_GIT_SHA,
        archive_sha256=provenance.CROMA_ARCHIVE_SHA256,
    )
    clay_record = _tree_record(
        clay,
        tmp_path / "clay-files.json",
        git_sha=provenance.CLAY_GIT_SHA,
        archive_sha256=provenance.CLAY_ARCHIVE_SHA256,
    )
    _, base_python = _fake_python(tmp_path / "base-python")
    _, model_python = _fake_python(tmp_path / "model-python")
    _, scoring_python = _fake_python(tmp_path / "scoring-python")
    model_lock = _write(
        tmp_path / "model-requirements.lock", b"numpy==2.2.6 --hash=x\n"
    )
    model_freeze = _write(
        tmp_path / "model-pip-freeze.txt", b"numpy==2.2.6\n"
    )
    scoring_lock = _write(
        tmp_path / "scoring-requirements.lock", b"numpy==1.26.4 --hash=x\n"
    )
    scoring_freeze = _write(
        tmp_path / "scoring-pip-freeze.txt", b"numpy==1.26.4\n"
    )

    def environment(
        python: dict,
        lock: Path,
        freeze: Path,
    ) -> dict:
        return {
            "python": python,
            "requirements_lock": {
                "path": str(lock),
                "size_bytes": lock.stat().st_size,
                "sha256": _sha(lock),
            },
            "pip_freeze": {
                "path": str(freeze),
                "size_bytes": freeze.stat().st_size,
                "sha256": _sha(freeze),
            },
        }

    manifest = tmp_path / "runtime.json"
    manifest.write_bytes(provenance.canonical_json_bytes({
        "schema": provenance.RUNTIME_SCHEMA,
        "base_image": provenance.BASE_IMAGE,
        "model_resolution": provenance.MODEL_RESOLUTION,
        "base_python": base_python,
        "source": source_record,
        "environments": {
            "model": environment(model_python, model_lock, model_freeze),
            "scoring": environment(
                scoring_python, scoring_lock, scoring_freeze
            ),
        },
        "external_sources": {"croma": croma_record, "clay": clay_record},
    }))
    return {
        "manifest": manifest,
        "source": source,
        "model_lock": model_lock,
        "model_freeze": model_freeze,
        "scoring_lock": scoring_lock,
        "scoring_freeze": scoring_freeze,
    }


def _split_bundle(
    tmp_path: Path,
    *,
    split_source_git_sha: str = SOURCE_GIT_SHA,
) -> dict[str, Path | str]:
    index = _write(tmp_path / "lucas_crop_distill_index.parquet", b"parquet")
    validator_holdout = _write(
        tmp_path / "lucas_crop_validator_holdout_index.parquet", b"holdout"
    )
    digests = {
        "qualified_keys_sha256": "4" * 64,
        "distill_keys_sha256": "5" * 64,
        "holdout_keys_sha256": "6" * 64,
        "partition_sha256": "7" * 64,
        "prior_test_tiles_sha256": "8" * 64,
        "prior_test_keys_sha256": "9" * 64,
        "source_index_sha256": "a" * 64,
        "forced_holdout_tiles_sha256": "b" * 64,
        "forced_holdout_keys_sha256": "c" * 64,
        "distill_input_data_sha256": "d" * 64,
        "validator_holdout_input_data_sha256": "e" * 64,
    }
    split = tmp_path / "lucas_crop_split.json"
    split.write_bytes(provenance.canonical_json_bytes({
        "plots": [],
        **digests,
    }))
    manifest = tmp_path / "lucas_crop_split.MANIFEST.json"
    manifest.write_bytes(provenance.canonical_json_bytes({
        "git_sha": split_source_git_sha,
        "artifacts": {
            index.name: _sha(index),
            validator_holdout.name: _sha(validator_holdout),
            split.name: _sha(split),
        },
        "n_qualified": 3,
        "n_distill": 2,
        "n_holdout": 1,
        **digests,
    }))
    return {
        "index": index,
        "validator_holdout": validator_holdout,
        "split": split,
        "manifest": manifest,
        "manifest_sha256": _sha(manifest),
    }


def _crop_args(
    tmp_path: Path,
    runtime: dict[str, Path | str],
    *,
    pod_uid: str = "pod-123",
    split_source_git_sha: str = SOURCE_GIT_SHA,
) -> tuple[object, dict[str, Path | str]]:
    model = "croma"
    bundle = _split_bundle(
        provenance.WORK_ROOT / pod_uid / "split",
        split_source_git_sha=split_source_git_sha,
    )
    checkpoint = _write(
        provenance.model_protocol(model).checkpoint_path,
        _CHECKPOINT_PAYLOAD,
    )
    features = _write(
        provenance.CROP_HEADS_DIR
        / f"{pod_uid}--{model}_r2_crop_features.parquet",
        b"features",
    )
    oof = _write(
        provenance.CROP_HEADS_DIR
        / f"{pod_uid}--{model}_r2_crop_distillability.json",
        b'{"score": 0.5}\n',
    )
    args = provenance.build_parser().parse_args([
        "finalize",
        "--model", model,
        "--record-dir", str(provenance.CROP_RECORD_DIR),
        "--run-id", pod_uid,
        "--job", "ladder-crop-distill-croma",
        "--pod-uid", pod_uid,
        "--status", "completed",
        "--source-git-sha", SOURCE_GIT_SHA,
        "--image-ref", IMAGE_REF,
        "--runtime-manifest", str(runtime["manifest"]),
        "--split-manifest", str(bundle["manifest"]),
        "--split-sha256", str(bundle["manifest_sha256"]),
        "--split-source-git-sha", split_source_git_sha,
        "--checkpoint", str(checkpoint),
        "--checkpoint-sha256", _sha(checkpoint),
        "--checkpoint-size", str(checkpoint.stat().st_size),
        "--artifact", f"features={features}",
        "--artifact-size", f"features={features.stat().st_size}",
        "--artifact-sha256", f"features={_sha(features)}",
        "--artifact", f"oof={oof}",
        "--artifact-size", f"oof={oof.stat().st_size}",
        "--artifact-sha256", f"oof={_sha(oof)}",
    ])
    return args, {
        **bundle,
        "checkpoint": checkpoint,
        "features": features,
        "oof": oof,
    }


def _failed_args(
    tmp_path: Path,
    runtime_manifest: Path,
    *,
    pod_uid: str,
    source_git_sha: str = SOURCE_GIT_SHA,
    split_sha256: str = "f" * 64,
):
    return provenance.build_parser().parse_args([
        "finalize",
        "--kind", "crop",
        "--record-dir", str(tmp_path / "records"),
        "--run-id", "crop-run-failed",
        "--job", "ladder-crop-distill-clay",
        "--pod-uid", pod_uid,
        "--status", "failed",
        "--exit-code", "17",
        "--failure-stage", "verify-runtime",
        "--source-git-sha", source_git_sha,
        "--image-ref", IMAGE_REF,
        "--runtime-manifest", str(runtime_manifest),
        "--split-sha256", split_sha256,
    ])


def _split_args(
    runtime: dict[str, Path | str],
    *,
    pod_uid: str,
    split_source_git_sha: str = SOURCE_GIT_SHA,
) -> tuple[object, dict[str, Path | str]]:
    bundle = _split_bundle(
        provenance.DISTILL_DIR,
        split_source_git_sha=split_source_git_sha,
    )
    args = provenance.build_parser().parse_args([
        "finalize",
        "--kind", "split",
        "--record-dir", str(provenance.SPLIT_RECORD_DIR),
        "--run-id", pod_uid,
        "--job", "ladder-lucas-crop-split",
        "--pod-uid", pod_uid,
        "--status", "completed",
        "--source-git-sha", SOURCE_GIT_SHA,
        "--image-ref", IMAGE_REF,
        "--runtime-manifest", str(runtime["manifest"]),
        "--split-manifest", str(bundle["manifest"]),
        "--split-sha256", str(bundle["manifest_sha256"]),
        "--split-source-git-sha", split_source_git_sha,
        "--source-access-plan-sha256", SOURCE_ACCESS_PLAN_SHA256,
        "--source-access-plan-pod-uid", SOURCE_ACCESS_PLAN_POD_UID,
        "--source-access-completion-sha256", SOURCE_ACCESS_COMPLETION_SHA256,
        "--source-access-completion-pod-uid", SOURCE_ACCESS_COMPLETION_POD_UID,
        "--artifact", f"index={bundle['index']}",
        "--artifact", f"validator_holdout={bundle['validator_holdout']}",
        "--artifact", f"split={bundle['split']}",
        "--artifact", f"manifest={bundle['manifest']}",
    ])
    return args, bundle


def test_verify_runtime_accepts_sealed_exact_payload(runtime):
    result = provenance.verify_runtime(
        runtime["manifest"],
        source_git_sha=SOURCE_GIT_SHA,
        image_ref=IMAGE_REF,
    )
    assert result["image"]["digest"] == "2" * 64
    assert result["source"]["git_sha"] == SOURCE_GIT_SHA
    assert set(result["environments"]) == {"model", "scoring"}
    assert result["model_resolution"] == provenance.MODEL_RESOLUTION
    assert result["external_sources"]["croma"]["git_sha"] == (
        provenance.CROMA_GIT_SHA
    )


@pytest.mark.parametrize("image_ref", [
    "ghcr.io/example/crop:latest",
    f"ghcr.io/example/crop@sha256:{'A' * 64}",
    "ghcr.io/example/crop@sha256:short",
])
def test_verify_runtime_rejects_mutable_or_malformed_image(runtime, image_ref):
    with pytest.raises(provenance.ProvenanceError, match="image-ref"):
        provenance.verify_runtime(
            runtime["manifest"],
            source_git_sha=SOURCE_GIT_SHA,
            image_ref=image_ref,
        )


def test_verify_runtime_rejects_missing_dependency(runtime):
    runtime["model_lock"].unlink()
    with pytest.raises(provenance.ProvenanceError, match="missing dependency"):
        provenance.verify_runtime(
            runtime["manifest"],
            source_git_sha=SOURCE_GIT_SHA,
            image_ref=IMAGE_REF,
        )


def test_verify_runtime_rejects_missing_environment(runtime):
    value = json.loads(runtime["manifest"].read_text())
    del value["environments"]["scoring"]
    runtime["manifest"].write_bytes(provenance.canonical_json_bytes(value))
    with pytest.raises(provenance.ProvenanceError, match="exactly model and scoring"):
        provenance.verify_runtime(
            runtime["manifest"],
            source_git_sha=SOURCE_GIT_SHA,
            image_ref=IMAGE_REF,
        )


def test_verify_runtime_rejects_python_identity_mismatch(runtime):
    value = json.loads(runtime["manifest"].read_text())
    value["base_python"]["version"] = "3.11.13"
    value["base_python"]["version_info"] = [3, 11, 13]
    runtime["manifest"].write_bytes(provenance.canonical_json_bytes(value))
    with pytest.raises(provenance.ProvenanceError, match="running interpreter"):
        provenance.verify_runtime(
            runtime["manifest"],
            source_git_sha=SOURCE_GIT_SHA,
            image_ref=IMAGE_REF,
        )


def test_verify_runtime_rejects_mutated_source_tree(runtime):
    _write(runtime["source"] / "scripts" / "worker.py", b"tampered\n")
    with pytest.raises(provenance.ProvenanceError, match="sealed tree differs"):
        provenance.verify_runtime(
            runtime["manifest"],
            source_git_sha=SOURCE_GIT_SHA,
            image_ref=IMAGE_REF,
        )


def test_verify_runtime_rejects_source_sha_mismatch(runtime):
    with pytest.raises(provenance.ProvenanceError, match="source git SHA mismatch"):
        provenance.verify_runtime(
            runtime["manifest"],
            source_git_sha="8" * 40,
            image_ref=IMAGE_REF,
        )


def test_completed_crop_record_is_byte_stable_and_write_once(tmp_path, runtime):
    args, _ = _crop_args(tmp_path, runtime)
    first = provenance.finalize(args)
    target = Path(first["record"])
    first_bytes = target.read_bytes()

    second = provenance.finalize(args)
    assert second["record_sha256"] == first["record_sha256"]
    assert first["process_identity"] == {
        "effective_uid": os.geteuid(),
        "effective_gid": os.getegid(),
    }
    assert target.read_bytes() == first_bytes
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o750

    with pytest.raises(provenance.ProvenanceError, match="refusing to overwrite"):
        provenance.write_once_bytes(target, b"different terminal record\n")
    assert target.read_bytes() == first_bytes


def test_completed_crop_binds_extractor_authenticated_checkpoint_without_reopen(
    tmp_path, runtime
):
    args, files = _crop_args(tmp_path, runtime)
    args.checkpoint_sha256 = "not-a-hash"
    with pytest.raises(provenance.ProvenanceError, match="64 lowercase hex"):
        provenance.finalize(args)

    args.checkpoint_sha256 = _sha(files["checkpoint"])
    expected = {
        "path": str(files["checkpoint"]),
        "size_bytes": files["checkpoint"].stat().st_size,
        "sha256": args.checkpoint_sha256,
        "verification": "extractor-authenticated-private-snapshot",
    }
    files["checkpoint"].unlink()
    result = provenance.finalize(args)
    assert result["checkpoint"] == expected


def test_completed_crop_rejects_missing_output(tmp_path, runtime):
    args, files = _crop_args(tmp_path, runtime)
    files["oof"].unlink()
    with pytest.raises(provenance.ProvenanceError, match="missing output artifact oof"):
        provenance.finalize(args)


def test_completed_crop_rejects_output_substitution_after_publication(
    tmp_path, runtime
):
    args, files = _crop_args(
        tmp_path, runtime, pod_uid="pod-substituted-output"
    )
    files["features"].write_bytes(b"attacker replacement")

    with pytest.raises(provenance.ProvenanceError, match="(size|SHA256) mismatch"):
        provenance.finalize(args)


def test_completed_crop_refuses_missing_runtime(tmp_path, runtime):
    args, _ = _crop_args(tmp_path, runtime, pod_uid="pod-no-runtime-complete")
    args.runtime_manifest = tmp_path / "missing-runtime.json"
    with pytest.raises(provenance.ProvenanceError, match="missing runtime manifest"):
        provenance.finalize(args)
    assert not (
        tmp_path / "records" / "pod-no-runtime-complete" / "completion.json"
    ).exists()


def test_split_manifest_hash_and_source_are_verified(
    tmp_path,
    runtime,
    monkeypatch,
):
    args, files = _crop_args(tmp_path, runtime)
    args.split_sha256 = "9" * 64
    with pytest.raises(provenance.ProvenanceError, match="SHA256 mismatch"):
        provenance.finalize(args)

    manifest = files["manifest"]
    value = json.loads(manifest.read_text())
    value["git_sha"] = "a" * 40
    manifest.write_bytes(provenance.canonical_json_bytes(value))
    args.split_sha256 = _sha(manifest)
    monkeypatch.setenv(
        "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
        args.split_sha256,
    )
    with pytest.raises(provenance.ProvenanceError, match="does not match"):
        provenance.finalize(args)


@pytest.mark.parametrize("kind", ["crop", "split"])
def test_completion_accepts_historical_split_source(
    tmp_path,
    runtime,
    monkeypatch,
    kind,
):
    if kind == "crop":
        args, _ = _crop_args(
            tmp_path,
            runtime,
            split_source_git_sha=HISTORICAL_SPLIT_SOURCE_GIT_SHA,
        )
        monkeypatch.setenv(
            "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
            args.split_sha256,
        )
    else:
        args, _ = _split_args(
            runtime,
            pod_uid="pod-historical-split",
            split_source_git_sha=HISTORICAL_SPLIT_SOURCE_GIT_SHA,
        )

    result = provenance.finalize(args)

    assert result["runtime"]["source"]["git_sha"] == SOURCE_GIT_SHA
    assert result["split_manifest"]["git_sha"] == (
        HISTORICAL_SPLIT_SOURCE_GIT_SHA
    )


def test_completion_rejects_wrong_historical_split_source(
    tmp_path,
    runtime,
    monkeypatch,
):
    args, _ = _crop_args(
        tmp_path,
        runtime,
        split_source_git_sha=HISTORICAL_SPLIT_SOURCE_GIT_SHA,
    )
    monkeypatch.setenv(
        "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
        args.split_sha256,
    )
    args.split_source_git_sha = "b" * 40

    with pytest.raises(provenance.ProvenanceError, match="split source"):
        provenance.finalize(args)


def test_split_manifest_and_document_are_parsed_from_authenticated_bytes(
    tmp_path, runtime, monkeypatch
):
    args, files = _crop_args(
        tmp_path, runtime, pod_uid="pod-private-snapshot-json"
    )
    original = provenance._regular_file_payload_identity

    def mutate_path_after_authenticated_read(path, label, **expected):
        result = original(path, label, **expected)
        if label in {
            "frozen split manifest",
            f"frozen split artifact {files['split'].name}",
        }:
            path.write_bytes(b"{}\n")
        return result

    monkeypatch.setattr(
        provenance,
        "_regular_file_payload_identity",
        mutate_path_after_authenticated_read,
    )
    result = provenance.finalize(args)
    assert result["split_manifest"]["sha256"] == files["manifest_sha256"]


def test_completed_crop_binds_all_immutable_split_digests(tmp_path, runtime):
    args, _ = _crop_args(tmp_path, runtime)
    result = provenance.finalize(args)
    digests = result["split_manifest"]["immutable_digests"]
    assert set(digests) == provenance._SPLIT_DIGEST_FIELDS
    assert digests["distill_input_data_sha256"] == "d" * 64
    assert digests["validator_holdout_input_data_sha256"] == "e" * 64


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing-model", "requires --model"),
        ("cross-model-job", "job must be exactly"),
        ("run-id", "run-id equal to pod-uid"),
        ("record-root", "record-dir must be exactly"),
        ("checkpoint-path", "checkpoint must be exactly"),
        ("checkpoint-size", "checkpoint size must equal"),
        ("checkpoint-sha", "checkpoint SHA256 must equal"),
        ("artifact-path", "exact Pod/model-owned paths"),
    ],
)
def test_completed_crop_rejects_confused_deputy_claims(
    tmp_path,
    runtime,
    mutation,
    match,
):
    args, _ = _crop_args(
        tmp_path,
        runtime,
        pod_uid=f"pod-authority-{mutation}",
    )
    if mutation == "missing-model":
        args.model = None
    elif mutation == "cross-model-job":
        args.job = "ladder-crop-distill-clay"
    elif mutation == "run-id":
        args.run_id = "different-safe-run"
    elif mutation == "record-root":
        args.record_dir = tmp_path / "attacker-records"
    elif mutation == "checkpoint-path":
        args.checkpoint = tmp_path / "attacker-checkpoint.pt"
    elif mutation == "checkpoint-size":
        args.checkpoint_size += 1
    elif mutation == "checkpoint-sha":
        args.checkpoint_sha256 = "f" * 64
    else:
        args.artifact[0] = f"features={tmp_path / 'attacker.parquet'}"

    with pytest.raises(provenance.ProvenanceError, match=match):
        provenance.finalize(args)


def test_completed_crop_rejects_wrong_process_identity(
    tmp_path,
    runtime,
    monkeypatch,
):
    args, _ = _crop_args(tmp_path, runtime, pod_uid="pod-wrong-uid")
    monkeypatch.setattr(
        provenance,
        "model_process_uid",
        lambda _model: os.geteuid() + 1,
    )
    with pytest.raises(provenance.ProvenanceError, match="process identity"):
        provenance.finalize(args)


def test_completed_crop_rejects_shared_manifest_instead_of_consumed_snapshot(
    tmp_path,
    runtime,
):
    args, _ = _crop_args(tmp_path, runtime, pod_uid="pod-shared-manifest")
    shared = _split_bundle(provenance.DISTILL_DIR)
    args.split_manifest = shared["manifest"]
    args.split_sha256 = shared["manifest_sha256"]

    with pytest.raises(
        provenance.ProvenanceError,
        match="consumed split-manifest must be exactly",
    ):
        provenance.finalize(args)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "distill_input_data_sha256 must be 64 lowercase hex"),
        (
            "malformed",
            "validator_holdout_input_data_sha256 must be 64 lowercase hex",
        ),
        ("mismatch", "disagrees between manifest and split"),
    ],
)
def test_completed_crop_rejects_invalid_split_digest_bindings(
    tmp_path,
    runtime,
    monkeypatch,
    mutation,
    match,
):
    args, files = _crop_args(tmp_path, runtime)
    manifest = files["manifest"]
    value = json.loads(manifest.read_text())
    if mutation == "missing":
        del value["distill_input_data_sha256"]
    elif mutation == "malformed":
        value["validator_holdout_input_data_sha256"] = "not-a-hash"
    else:
        value["distill_input_data_sha256"] = "f" * 64
    manifest.write_bytes(provenance.canonical_json_bytes(value))
    args.split_sha256 = _sha(manifest)
    monkeypatch.setenv(
        "CROP_DISTILL_SPLIT_MANIFEST_SHA256",
        args.split_sha256,
    )
    with pytest.raises(provenance.ProvenanceError, match=match):
        provenance.finalize(args)


def test_completed_split_records_all_four_outputs(tmp_path, runtime):
    pod_uid = "pod-split"
    args, _ = _split_args(runtime, pod_uid=pod_uid)
    result = provenance.finalize(args)
    assert result["kind"] == "split"
    assert set(result["artifacts"]) == {
        "index", "validator_holdout", "split", "manifest",
    }
    assert result["checkpoint"] is None
    assert result["source_access"] == {
        "plan": {
            "sha256": SOURCE_ACCESS_PLAN_SHA256,
            "pod_uid": SOURCE_ACCESS_PLAN_POD_UID,
        },
        "completion": {
            "sha256": SOURCE_ACCESS_COMPLETION_SHA256,
            "pod_uid": SOURCE_ACCESS_COMPLETION_POD_UID,
        },
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_access_plan_sha256", None, "requires the PLAN and APPLY"),
        ("source_access_plan_sha256", "0" * 64, "must be nonzero"),
        ("source_access_plan_pod_uid", "../unsafe", "must contain only"),
        ("source_access_completion_sha256", "not-a-hash", "64 lowercase hex"),
        ("source_access_completion_pod_uid", "", "must contain only"),
    ],
)
def test_completed_split_rejects_invalid_source_access_authority(
    runtime,
    field,
    value,
    match,
):
    args, _ = _split_args(runtime, pod_uid=f"pod-invalid-{field}")
    setattr(args, field, value)

    with pytest.raises(provenance.ProvenanceError, match=match):
        provenance.finalize(args)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("model", "must not specify --model"),
        ("run-id", "run-id equal to pod-uid"),
        ("job", "job must be exactly"),
        ("record-root", "record-dir must be exactly"),
        ("manifest", "split-manifest must be exactly"),
        ("artifact", "exact protocol paths"),
    ],
)
def test_completed_split_rejects_confused_deputy_claims(
    tmp_path,
    runtime,
    mutation,
    match,
):
    args, _ = _split_args(runtime, pod_uid=f"pod-split-{mutation}")
    if mutation == "model":
        args.model = "croma"
    elif mutation == "run-id":
        args.run_id = "different-safe-run"
    elif mutation == "job":
        args.job = "ladder-lucas-crop-split-attacker"
    elif mutation == "record-root":
        args.record_dir = tmp_path / "attacker-records"
    elif mutation == "manifest":
        args.split_manifest = (
            tmp_path / provenance.CROP_SPLIT_MANIFEST.name
        )
        args.artifact[-1] = f"manifest={args.split_manifest}"
    else:
        args.artifact[0] = (
            f"index={tmp_path / provenance.CROP_INDEX.name}"
        )

    with pytest.raises(provenance.ProvenanceError, match=match):
        provenance.finalize(args)


def test_completed_split_rejects_wrong_process_identity(
    runtime,
    monkeypatch,
):
    args, _ = _split_args(runtime, pod_uid="pod-split-wrong-uid")
    monkeypatch.setattr(provenance, "STORAGE_UID", os.geteuid() + 1)
    with pytest.raises(provenance.ProvenanceError, match="process identity"):
        provenance.finalize(args)


def test_crop_never_accesses_validator_holdout(tmp_path, runtime):
    args, bundle = _crop_args(tmp_path, runtime)
    validator_holdout = bundle["validator_holdout"]
    validator_holdout.unlink()
    result = provenance.finalize(args)
    declaration = result["split_manifest"]["declared_artifacts"][
        validator_holdout.name
    ]
    assert declaration == {
        "path": str(validator_holdout),
        "sha256": hashlib.sha256(b"holdout").hexdigest(),
        "verification": "declaration-only",
    }


def test_split_completion_reads_validator_holdout(
    tmp_path, runtime
):
    pod_uid = "pod-split-guard"
    args, bundle = _split_args(runtime, pod_uid=pod_uid)
    validator_holdout = bundle["validator_holdout"]
    result = provenance.finalize(args)
    assert result["artifacts"]["validator_holdout"] == {
        "path": str(validator_holdout),
        "size_bytes": validator_holdout.stat().st_size,
        "sha256": _sha(validator_holdout),
    }


def test_failed_record_requires_nonzero_but_not_pipeline_artifacts(tmp_path, runtime):
    parser = provenance.build_parser()
    argv = [
        "finalize",
        "--kind", "crop",
        "--record-dir", str(tmp_path / "records"),
        "--run-id", "crop-run-failed",
        "--job", "ladder-crop-distill-clay",
        "--pod-uid", "pod-failed",
        "--status", "failed",
        "--failure-stage", "extract-features",
        "--source-git-sha", SOURCE_GIT_SHA,
        "--image-ref", IMAGE_REF,
        "--runtime-manifest", str(runtime["manifest"]),
    ]
    with pytest.raises(provenance.ProvenanceError, match="positive exit-code"):
        provenance.finalize(parser.parse_args(argv))

    result = provenance.finalize(parser.parse_args(argv + ["--exit-code", "17"]))
    assert result["terminal"] == {
        "status": "failed",
        "exit_code": 17,
        "failure_stage": "extract-features",
    }
    assert result["split_manifest"] == {
        "verification": "not-dereferenced",
        "claimed_sha256": "<missing>",
    }
    assert result["checkpoint"] is None
    assert result["artifacts"] == {}


def test_failed_missing_runtime_is_immutable_diagnostic(tmp_path, runtime):
    missing = tmp_path / "missing-runtime.json"
    args = _failed_args(
        tmp_path, missing, pod_uid="pod-failed-missing-runtime"
    )
    first = provenance.finalize(args)
    assert first["runtime"] == {
        "verification": "not-dereferenced",
        "claimed": {
            "image_ref": IMAGE_REF,
            "source_git_sha": SOURCE_GIT_SHA,
            "runtime_manifest": str(missing),
        },
    }
    assert first["split_manifest"]["claimed_sha256"] == "f" * 64

    second = provenance.finalize(args)
    assert second["record_sha256"] == first["record_sha256"]
    args.failure_stage = "retry-with-different-stage"
    with pytest.raises(provenance.ProvenanceError, match="refusing to overwrite"):
        provenance.finalize(args)


def test_failed_record_does_not_require_completion_authority(
    tmp_path,
    runtime,
    monkeypatch,
):
    args = _failed_args(
        tmp_path,
        runtime["manifest"],
        pod_uid="pod-diagnostic-only",
    )
    args.record_dir = tmp_path / "diagnostic-records"
    args.run_id = "diagnostic-run-not-pod"
    args.job = "diagnostic-job"
    monkeypatch.delenv("CROP_DISTILL_SPLIT_MANIFEST_SHA256")
    monkeypatch.setattr(
        provenance,
        "model_process_uid",
        lambda _model: pytest.fail("failure record checked model authority"),
    )

    result = provenance.finalize(args)
    assert result["terminal"]["status"] == "failed"
    assert result["model"] is None
    assert Path(result["record"]).is_file()


def test_failed_corrupt_runtime_is_published_as_untrusted(tmp_path, runtime):
    corrupt = _write(tmp_path / "corrupt-runtime.json", b"{not-json\n")
    args = _failed_args(
        tmp_path, corrupt, pod_uid="pod-failed-corrupt-runtime"
    )
    result = provenance.finalize(args)
    assert result["runtime"]["verification"] == "not-dereferenced"
    assert result["runtime"]["claimed"]["runtime_manifest"] == str(corrupt)


def test_failed_source_mismatch_is_published_as_untrusted(tmp_path, runtime):
    args = _failed_args(
        tmp_path,
        runtime["manifest"],
        pod_uid="pod-failed-source-mismatch",
        source_git_sha="8" * 40,
    )
    result = provenance.finalize(args)
    assert result["runtime"]["verification"] == "not-dereferenced"
    assert result["runtime"]["claimed"]["source_git_sha"] == "8" * 40


def test_failed_malformed_runtime_claims_are_safely_recorded(tmp_path, runtime):
    args = _failed_args(
        tmp_path,
        runtime["manifest"],
        pod_uid="pod-failed-malformed-claims",
        source_git_sha="not-a-sha\nwith-control",
    )
    args.image_ref = "mutable:latest\nwith-control"
    args.split_sha256 = "not-a-hash\nwith-control"
    result = provenance.finalize(args)
    assert result["runtime"]["verification"] == "not-dereferenced"
    assert result["runtime"]["claimed"] == {
        "image_ref": "mutable:latest with-control",
        "source_git_sha": "not-a-sha with-control",
        "runtime_manifest": str(runtime["manifest"]),
    }
    assert result["split_manifest"] == {
        "verification": "not-dereferenced",
        "claimed_sha256": "not-a-hash with-control",
    }


def test_failed_record_never_reads_pipeline_evidence(
    tmp_path, runtime, monkeypatch
):
    args, bundle = _crop_args(
        tmp_path, runtime, pod_uid="pod-failed-no-pipeline-read"
    )
    args.status = "failed"
    args.exit_code = 9
    args.failure_stage = "extract-features"
    # EXIT traps may inherit malformed partial flags. A failed record ignores
    # them entirely instead of parsing or dereferencing pipeline evidence.
    args.artifact = ["not-even-name-equals-path"]
    args.artifact_size = ["malformed"]
    args.artifact_sha256 = ["malformed"]
    args.split_sha256 = "malformed"
    args.checkpoint_sha256 = "malformed"
    args.checkpoint_size = -1
    forbidden = {
        runtime["manifest"],
        bundle["manifest"],
        bundle["index"],
        bundle["validator_holdout"],
        bundle["split"],
        bundle["checkpoint"],
        bundle["features"],
        bundle["oof"],
    }
    original_lstat = Path.lstat
    original_open = Path.open
    original_os_open = provenance.os.open

    def guarded_lstat(path, *values, **keywords):
        if path in forbidden:
            raise AssertionError(f"failed finalization accessed {path}")
        return original_lstat(path, *values, **keywords)

    def guarded_open(path, *values, **keywords):
        if path in forbidden:
            raise AssertionError(f"failed finalization opened {path}")
        return original_open(path, *values, **keywords)

    forbidden_names = {Path(path).name for path in forbidden}

    def guarded_os_open(path, *values, **keywords):
        if str(path) in forbidden_names:
            raise AssertionError(f"failed finalization opened {path}")
        return original_os_open(path, *values, **keywords)

    def forbidden_runtime_verification(*_args, **_kwargs):
        raise AssertionError("failed finalization verified the runtime")

    monkeypatch.setattr(Path, "lstat", guarded_lstat)
    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(provenance.os, "open", guarded_os_open)
    monkeypatch.setattr(
        provenance, "verify_runtime", forbidden_runtime_verification
    )
    result = provenance.finalize(args)
    assert result["runtime"]["verification"] == "not-dereferenced"
    assert result["split_manifest"] == {
        "verification": "not-dereferenced",
        "claimed_sha256": "malformed",
    }
    assert result["checkpoint"] is None
    assert result["artifacts"] == {}


def test_terminal_record_rejects_symlinked_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(
        provenance.ProvenanceError, match="symlink or non-directory"
    ):
        provenance.write_once_bytes(alias / "completion.json", b"evidence\n")
    assert not (real / "completion.json").exists()


def test_terminal_record_rejects_writable_shared_parent(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    shared.chmod(0o777)

    with pytest.raises(
        provenance.ProvenanceError, match="group/other write access"
    ):
        provenance.write_once_bytes(shared / "completion.json", b"evidence\n")


def test_terminal_record_rejects_preexisting_symlink(tmp_path):
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    victim = tmp_path / "victim"
    victim.write_bytes(b"victim\n")
    target = record_dir / "completion.json"
    target.symlink_to(victim)

    with pytest.raises(provenance.ProvenanceError, match="cannot verify raced"):
        provenance.write_once_bytes(target, b"evidence\n")
    assert victim.read_bytes() == b"victim\n"


def test_terminal_record_rejects_preexisting_hardlink(tmp_path):
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    victim = tmp_path / "victim"
    victim.write_bytes(b"evidence\n")
    target = record_dir / "completion.json"
    target.hardlink_to(victim)

    with pytest.raises(provenance.ProvenanceError, match="exactly one hard link"):
        provenance.write_once_bytes(target, b"evidence\n")
    assert victim.read_bytes() == b"evidence\n"


def test_terminal_record_recovers_crash_after_link_before_temp_unlink(
    tmp_path,
    monkeypatch,
):
    record_dir = tmp_path / "records"
    target = record_dir / "completion.json"
    payload = b"immutable terminal evidence\n"
    real_unlink = provenance.os.unlink

    def interrupt_temp_unlink(path, *args, **kwargs):
        if str(path).startswith(".completion.json.") and str(path).endswith(
            ".create"
        ):
            raise OSError("simulated process death after publication")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(provenance.os, "unlink", interrupt_temp_unlink)
    with pytest.raises(OSError, match="simulated process death"):
        provenance.write_once_bytes(target, payload)

    assert target.read_bytes() == payload
    assert target.stat().st_nlink == 2
    stale = list(record_dir.glob(".completion.json.*.create"))
    assert len(stale) == 1

    monkeypatch.setattr(provenance.os, "unlink", real_unlink)
    provenance.write_once_bytes(target, payload)

    assert target.read_bytes() == payload
    assert target.stat().st_nlink == 1
    assert list(record_dir.glob(".completion.json.*.create")) == []
    with pytest.raises(provenance.ProvenanceError, match="refusing to overwrite"):
        provenance.write_once_bytes(target, b"different\n")


def test_terminal_evidence_round_trips_exact_on_disk_bytes(
    tmp_path,
    runtime,
):
    args, _ = _crop_args(tmp_path, runtime, pod_uid="pod-terminal-evidence")
    result = provenance.finalize(args)
    line = provenance.terminal_evidence_line(result)
    digest, payload, record = provenance.parse_terminal_evidence_line(line)
    target = Path(result["record"])

    assert line.startswith(f"{provenance.TERMINAL_EVIDENCE_PREFIX} ")
    assert digest == result["record_sha256"]
    assert payload == target.read_bytes()
    assert record["model"] == "croma"
    assert hashlib.sha256(payload).hexdigest() == digest


def test_finalize_main_emits_exactly_one_terminal_evidence_marker(
    tmp_path,
    runtime,
    monkeypatch,
    capsys,
):
    args, _ = _crop_args(tmp_path, runtime, pod_uid="pod-main-evidence")
    result = provenance.finalize(args)

    class Parser:
        @staticmethod
        def parse_args(_argv):
            return args

    monkeypatch.setattr(provenance, "build_parser", lambda: Parser())
    monkeypatch.setattr(provenance, "finalize", lambda _args: result)
    provenance.main([])

    markers = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(provenance.TERMINAL_EVIDENCE_PREFIX)
    ]
    assert len(markers) == 1
    _, payload, _ = provenance.parse_terminal_evidence_line(markers[0])
    assert payload == Path(result["record"]).read_bytes()


def test_terminal_evidence_parser_rejects_malformed_or_noncanonical_lines(
    tmp_path,
    runtime,
):
    args, _ = _crop_args(tmp_path, runtime, pod_uid="pod-invalid-evidence")
    result = provenance.finalize(args)
    valid = provenance.terminal_evidence_line(result)
    _, _, record = provenance.parse_terminal_evidence_line(valid)
    prefix, digest, encoded = valid.split(" ")
    noncanonical = json.dumps(record, sort_keys=True).encode("utf-8")
    noncanonical_line = (
        f"{prefix} {hashlib.sha256(noncanonical).hexdigest()} "
        f"{base64.b64encode(noncanonical).decode('ascii')}"
    )
    malformed = [
        valid.replace(prefix, "WRONG_EVIDENCE_PREFIX", 1),
        f"{prefix} {'A' * 64} {encoded}",
        f"{prefix} {'0' * 64} {encoded}",
        f"{prefix} {digest} !!!!",
        f"{prefix}  {digest} {encoded}",
        valid + "\nextra",
        noncanonical_line,
    ]

    for line in malformed:
        with pytest.raises(provenance.ProvenanceError):
            provenance.parse_terminal_evidence_line(line)


def test_completed_output_rejects_symlink(tmp_path, runtime):
    args, files = _crop_args(tmp_path, runtime, pod_uid="pod-output-symlink")
    victim = tmp_path / "victim-output"
    victim.write_bytes(b"attacker\n")
    files["features"].unlink()
    files["features"].symlink_to(victim)

    with pytest.raises(provenance.ProvenanceError, match="missing output artifact"):
        provenance.finalize(args)
    assert victim.read_bytes() == b"attacker\n"
