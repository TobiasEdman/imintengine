"""Least-privilege storage bootstrap for crop-distill Jobs."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts import crop_distill_protocol as protocol
from scripts import crop_distill_provenance as provenance
from scripts import prepare_crop_distill_storage as storage_prep
from scripts.crop_distill_protocol import RuntimeIdentity, StorageTarget

SOURCE_SHA = "a" * 40
IMAGE_REF = "ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:" + "b" * 64
POD_UID = "storage-prep-pod"


def _local_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exercise real fd chmod/chown while retaining the current host owner."""
    real_gid = os.getgid()
    monkeypatch.setattr(storage_prep, "STORAGE_GID", real_gid)
    monkeypatch.setattr(storage_prep, "DATASET_LOCK_UID", os.getuid())
    monkeypatch.setattr(storage_prep, "FROZEN_SPLIT_MODE", 0o550)
    lock_root = tmp_path / "source-access-lock"
    lock_root.mkdir()
    monkeypatch.setattr(
        storage_prep,
        "SOURCE_ACCESS_LOCK_BACKING_FILE",
        lock_root / "dataset.lock",
    )
    monkeypatch.setattr(storage_prep.os, "geteuid", lambda: 0)
    monkeypatch.setattr(storage_prep.os, "getegid", lambda: real_gid)
    monkeypatch.setattr(
        storage_prep,
        "runtime_identity",
        lambda _env: RuntimeIdentity(
            SOURCE_SHA,
            IMAGE_REF,
            POD_UID,
        ),
    )
    monkeypatch.setattr(
        storage_prep,
        "verify_runtime",
        lambda *_args, **_kwargs: {"verification": "verified"},
    )


def _local_target(
    path: Path,
    *,
    mode: int = 0o750,
    preserve_mode: int | None = None,
) -> StorageTarget:
    return StorageTarget(
        path=path,
        uid=os.getuid(),
        gid=os.getgid(),
        mode=mode,
        preserve_mode=preserve_mode,
    )


def test_prepare_storage_changes_only_baked_roots(monkeypatch, tmp_path):
    _local_identity(monkeypatch, tmp_path)
    distill = tmp_path / "distill"
    ops = tmp_path / "ops"
    distill.mkdir()
    ops.mkdir()
    crop_split = distill / "crop_split"
    heads = distill / "crop_heads"
    records = ops / "crop-distill"
    untouched = distill / "unrelated"
    untouched.mkdir(mode=0o755)
    monkeypatch.setattr(
        storage_prep,
        "STORAGE_TARGETS",
        tuple(_local_target(path) for path in (crop_split, heads, records)),
    )

    result = storage_prep.prepare_storage()

    assert result["schema"] == storage_prep.STORAGE_PREP_COMPLETION_SCHEMA
    assert result["pod_uid"] == POD_UID
    assert result["status"] == "completed"
    assert result["process_identity"] == {
        "effective_uid": 0,
        "effective_gid": os.getgid(),
    }
    assert result["runtime"] == {"verification": "verified"}
    assert result["dataset_lock"] == {
        "path": str(storage_prep.SOURCE_ACCESS_LOCK_BACKING_FILE),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "mode": "0660",
        "device": result["dataset_lock"]["device"],
        "inode": result["dataset_lock"]["inode"],
        "size_bytes": 0,
        "nlink": 1,
        "state": "ready",
    }
    assert (
        stat.S_IMODE(storage_prep.SOURCE_ACCESS_LOCK_BACKING_FILE.stat().st_mode)
        == 0o660
    )
    assert {item["path"] for item in result["targets"]} == {
        str(crop_split),
        str(heads),
        str(records),
    }
    for path in (crop_split, heads, records):
        identity = path.lstat()
        assert stat.S_ISDIR(identity.st_mode)
        assert stat.S_IMODE(identity.st_mode) == 0o750
    assert stat.S_IMODE(untouched.stat().st_mode) == 0o755


def test_storage_prep_publishes_marker_for_exact_immutable_record(
    monkeypatch,
    tmp_path,
    capsys,
):
    record_root = tmp_path / "storage-prep-records"
    monkeypatch.setattr(storage_prep, "STORAGE_PREP_RECORD_DIR", record_root)
    record = {
        "schema": storage_prep.STORAGE_PREP_COMPLETION_SCHEMA,
        "pod_uid": POD_UID,
        "status": "completed",
        "process_identity": {
            "effective_uid": 0,
            "effective_gid": protocol.STORAGE_GID,
        },
        "preserved_frozen_mode": "0550",
        "runtime": {"verification": "verified"},
        "targets": [],
        "dataset_lock": {
            "path": str(protocol.SOURCE_ACCESS_LOCK_BACKING_FILE),
            "uid": 0,
            "gid": protocol.STORAGE_GID,
            "mode": "0660",
            "device": 1,
            "inode": 1,
            "size_bytes": 0,
            "nlink": 1,
            "state": "ready",
        },
    }
    monkeypatch.setattr(storage_prep, "prepare_storage", lambda _env: record)

    assert storage_prep.main([], environ={}) == 0

    marker_line, summary_line = capsys.readouterr().out.splitlines()
    prefix, digest, encoded = marker_line.split(" ")
    payload = base64.b64decode(encoded, validate=True)
    record_path = record_root / POD_UID / "completion.json"
    summary = json.loads(summary_line)
    assert prefix == storage_prep.STORAGE_PREP_COMPLETION_MARKER
    assert payload == record_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == digest
    assert summary["record"] == str(record_path)
    assert summary["record_sha256"] == digest
    assert json.loads(payload) == record
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o444

    with pytest.raises(provenance.ProvenanceError, match="refusing to overwrite"):
        storage_prep.publish_completion(
            {**json.loads(payload), "status": "different"}
        )


def test_prepare_storage_is_idempotent(monkeypatch, tmp_path):
    _local_identity(monkeypatch, tmp_path)
    parent = tmp_path / "distill"
    parent.mkdir()
    target = parent / "crop_split"
    monkeypatch.setattr(
        storage_prep, "STORAGE_TARGETS", (_local_target(target),)
    )

    first = storage_prep.prepare_storage()
    second = storage_prep.prepare_storage()

    assert first == second


def test_prepare_storage_preserves_locked_frozen_split(monkeypatch, tmp_path):
    _local_identity(monkeypatch, tmp_path)
    parent = tmp_path / "distill"
    parent.mkdir()
    target = parent / "crop_split"
    target.mkdir(mode=0o750)
    monkeypatch.setattr(
        storage_prep,
        "STORAGE_TARGETS",
        (_local_target(target, mode=0o770, preserve_mode=0o550),),
    )
    storage_prep.prepare_storage()
    target.chmod(0o550)

    replay = storage_prep.prepare_storage()

    assert stat.S_IMODE(target.stat().st_mode) == 0o550
    assert replay["targets"][0]["state"] == "preserved-frozen"


def test_prepare_storage_rejects_symlink_target(monkeypatch, tmp_path):
    _local_identity(monkeypatch, tmp_path)
    parent = tmp_path / "distill"
    parent.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    target = parent / "crop_split"
    target.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(
        storage_prep, "STORAGE_TARGETS", (_local_target(target),)
    )

    with pytest.raises(storage_prep.StoragePrepError, match="not a real directory"):
        storage_prep.prepare_storage()

    assert stat.S_IMODE(real.stat().st_mode) == 0o755


@pytest.mark.parametrize("attack", ["symlink", "hardlink", "nonempty"])
def test_prepare_storage_rejects_aliased_or_nonempty_dataset_lock(
    monkeypatch,
    tmp_path,
    attack,
):
    _local_identity(monkeypatch, tmp_path)
    monkeypatch.setattr(storage_prep, "STORAGE_TARGETS", ())
    lock = storage_prep.SOURCE_ACCESS_LOCK_BACKING_FILE
    victim = tmp_path / "victim-lock"
    victim.write_bytes(b"x" if attack == "nonempty" else b"")
    if attack == "symlink":
        lock.symlink_to(victim)
    elif attack == "hardlink":
        lock.hardlink_to(victim)
    else:
        lock.write_bytes(b"unexpected")

    with pytest.raises(
        storage_prep.StoragePrepError,
        match="securely create/open|empty, unaliased",
    ):
        storage_prep.prepare_storage()
    assert victim.read_bytes() == (b"x" if attack == "nonempty" else b"")


def test_storage_prep_accepts_no_path_or_identity_arguments():
    with pytest.raises(SystemExit):
        storage_prep.build_parser().parse_args(["--path", "/tmp/other"])


def test_storage_prep_verifies_runtime_before_mutating(monkeypatch, tmp_path):
    parent = tmp_path / "distill"
    parent.mkdir()
    target = parent / "crop_split"
    monkeypatch.setattr(
        storage_prep, "STORAGE_TARGETS", (_local_target(target),)
    )
    monkeypatch.setattr(storage_prep.os, "geteuid", lambda: 0)
    monkeypatch.setattr(storage_prep.os, "getegid", lambda: storage_prep.STORAGE_GID)
    monkeypatch.setattr(
        storage_prep,
        "runtime_identity",
        lambda _env: RuntimeIdentity(SOURCE_SHA, IMAGE_REF, POD_UID),
    )

    def reject(*_args, **_kwargs):
        raise ValueError("runtime mismatch")

    monkeypatch.setattr(storage_prep, "verify_runtime", reject)
    with pytest.raises(ValueError, match="runtime mismatch"):
        storage_prep.prepare_storage({})
    assert not target.exists()


def test_production_split_storage_identity_is_dedicated_setgid_and_sticky():
    assert protocol.STORAGE_UID == 2000
    assert protocol.STORAGE_GID == 2000
    assert protocol.STORAGE_MODE == 0o3770
    assert protocol.FROZEN_SPLIT_MODE == 0o550
    assert protocol.SOURCE_ACCESS_LOCK_MODE == 0o660
    assert (
        protocol.SOURCE_ACCESS_LOCK_BACKING_FILE
        == protocol.SOURCE_ACCESS_LOCK_BACKING_DIR / "dataset.lock"
    )
    assert protocol.SOURCE_ACCESS_LOCK_FILE == Path(
        "/cephfs/source-access-lock/dataset.lock"
    )


def test_production_storage_layout_preowns_isolated_model_directories():
    targets = {target.path: target for target in protocol.STORAGE_TARGETS}
    expected_paths = {
        protocol.DISTILL_DIR,
        protocol.CROP_HEADS_BACKING_ROOT,
        protocol.CROP_RECORDS_BACKING_ROOT,
        protocol.SPLIT_RECORD_BACKING_DIR,
        protocol.SOURCE_ACCESS_BACKING_ROOT,
        protocol.SOURCE_ACCESS_PLAN_BACKING_DIR,
        protocol.SOURCE_ACCESS_APPLY_BACKING_DIR,
        protocol.SOURCE_ACCESS_LOCK_BACKING_DIR,
    }
    expected_paths.update(
        protocol.crop_head_backing_dir(model) for model in protocol.MODEL_KEYS
    )
    expected_paths.update(
        protocol.crop_record_backing_dir(model) for model in protocol.MODEL_KEYS
    )
    assert len(protocol.STORAGE_TARGETS) == len(expected_paths) == 20
    assert set(targets) == expected_paths

    split = targets[protocol.DISTILL_DIR]
    assert (split.uid, split.gid, split.mode, split.preserve_mode) == (
        2000,
        2000,
        0o3770,
        0o550,
    )
    for root in (
        protocol.CROP_HEADS_BACKING_ROOT,
        protocol.CROP_RECORDS_BACKING_ROOT,
    ):
        target = targets[root]
        assert (target.uid, target.gid, target.mode) == (0, 2000, 0o750)

    split_record = targets[protocol.SPLIT_RECORD_BACKING_DIR]
    assert (split_record.uid, split_record.gid, split_record.mode) == (
        2000,
        2000,
        0o750,
    )
    for path in (
        protocol.SOURCE_ACCESS_BACKING_ROOT,
        protocol.SOURCE_ACCESS_PLAN_BACKING_DIR,
        protocol.SOURCE_ACCESS_APPLY_BACKING_DIR,
        protocol.SOURCE_ACCESS_LOCK_BACKING_DIR,
    ):
        target = targets[path]
        assert (target.uid, target.gid, target.mode) == (0, 2000, 0o750)
    for model, uid in protocol.CROP_MODEL_UIDS.items():
        head = targets[protocol.crop_head_backing_dir(model)]
        record = targets[protocol.crop_record_backing_dir(model)]
        assert (head.uid, head.gid, head.mode) == (uid, 2000, 0o750)
        assert (record.uid, record.gid, record.mode) == (uid, 2000, 0o750)


def test_storage_prep_rejects_name_squatted_child(monkeypatch, tmp_path):
    _local_identity(monkeypatch, tmp_path)
    parent = tmp_path / "crop_heads"
    parent.mkdir()
    squatted = parent / "clay_r2_crop_runs"
    squatted.mkdir()
    wrong_uid = os.getuid() + 10_000
    monkeypatch.setattr(
        storage_prep,
        "STORAGE_TARGETS",
        (
            StorageTarget(
                squatted,
                uid=wrong_uid,
                gid=os.getgid(),
                mode=0o750,
            ),
        ),
    )

    with pytest.raises(storage_prep.StoragePrepError, match="unexpected owner"):
        storage_prep.prepare_storage()
