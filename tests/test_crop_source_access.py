"""Fail-closed PLAN/APPLY contract for LUCAS crop source access."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import crop_source_access as access
from scripts.crop_distill_protocol import RuntimeIdentity

IDENTITY = RuntimeIdentity(
    "a" * 40,
    "ghcr.io/tobiasedman/imint-ladder-crop-distill@sha256:" + "b" * 64,
    "plan-pod-uid",
)
VERIFIED_RUNTIME = {
    "runtime_manifest": {"sha256": "c" * 64},
    "source": {"payload_sha256": "d" * 64},
}


def _index_bytes(path: Path, rows: list[dict[str, object]]) -> bytes:
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path.read_bytes()


def _row(tile_name: str, *, row: int, col: int, point_id: int = 1) -> dict[str, object]:
    return {
        "tile_name": tile_name,
        "row": row,
        "col": col,
        "unified_class": 11,
        "point_id": point_id,
    }


def _write_lease(
    path: Path,
    *,
    phase: str = "apply",
    status: str = "held",
    heartbeat_ns: int = 1_000_000_000,
    valid_until_ns: int = 181_000_000_000,
) -> None:
    path.write_bytes(access.canonical_json_bytes({
        "schema": access.FREEZE_LEASE_SCHEMA,
        "run_id": "crop-repair-attempt-3",
        "phase": phase,
        "status": status,
        "sequence": 7,
        "heartbeat_unix_ns": heartbeat_ns,
        "valid_until_unix_ns": valid_until_ns,
        "snapshot_sha256": "a" * 64,
        "controller_snapshot_sha256": "b" * 64,
    }))


def test_freeze_lease_is_phase_bound_and_fail_closed_on_watchdog_loss(tmp_path):
    lease = tmp_path / "lease.json"
    _write_lease(lease)

    accepted = access.require_fresh_freeze_lease(
        lease,
        expected_phase="apply",
        now_ns=2_000_000_000,
    )
    assert accepted["sequence"] == 7

    with pytest.raises(access.SourceAccessError, match="not held for phase split"):
        access.require_fresh_freeze_lease(
            lease,
            expected_phase="split",
            now_ns=2_000_000_000,
        )

    with pytest.raises(access.SourceAccessError, match="watchdog lease is stale"):
        access.require_fresh_freeze_lease(
            lease,
            expected_phase="apply",
            now_ns=181_000_000_000,
        )

    _write_lease(lease, status="failed")
    with pytest.raises(access.SourceAccessError, match="not held for phase apply"):
        access.require_fresh_freeze_lease(
            lease,
            expected_phase="apply",
            now_ns=2_000_000_000,
        )


def test_candidates_apply_crop_window_before_unique_tile_inventory(tmp_path):
    payload = _index_bytes(
        tmp_path / "index.parquet",
        [
            _row("inside", row=8, col=503),
            _row("border-only", row=7, col=100, point_id=2),
            _row("inside", row=100, col=100, point_id=3),
            {
                **_row("not-a-crop", row=100, col=100, point_id=4),
                "unified_class": 8,
            },
        ],
    )

    names, crop_rows, window = access.derive_crop_window_candidates(payload)

    assert window == [8, 504]
    assert crop_rows == 2
    assert names == ["inside"]
    assert "border-only" not in names


def test_candidates_refuse_source_tile_path_authority(tmp_path):
    row = {**_row("inside", row=100, col=100), "tile_path": "/attacker/tile.npz"}
    payload = _index_bytes(tmp_path / "index.parquet", [row])

    with pytest.raises(access.SourceAccessError, match="must not supply tile_path"):
        access.derive_crop_window_candidates(payload)


def test_plan_inventories_canonical_files_and_actions(tmp_path, monkeypatch):
    index = tmp_path / "lucas_tile_index.parquet"
    _index_bytes(index, [_row("tile-b", row=10, col=10), _row("tile-a", row=20, col=20)])
    data = tmp_path / "tiles"
    data.mkdir()
    for name in ("tile-a", "tile-b"):
        (data / f"{name}.npz").write_bytes(f"payload-{name}".encode())
        (data / f"{name}.npz").chmod(0o600)

    monkeypatch.setattr(access, "_classify", lambda _identity: access.ACTION_REPAIR)
    index_payload = index.read_bytes()
    plan = access.build_plan_record(
        identity=IDENTITY,
        verified_runtime=VERIFIED_RUNTIME,
        source_index=index,
        data_dir=data,
        expected_index_sha256=hashlib.sha256(index_payload).hexdigest(),
        expected_index_size=len(index_payload),
        enforce_production_counts=False,
    )

    assert plan["crop_window"] == [8, 504]
    assert plan["crop_rows"] == 2
    assert plan["summary"] == {
        "candidates": 2,
        "repairs": 2,
        "accepted_0644": 0,
        "already_correct": 0,
    }
    files = plan["files"]
    assert [item["tile_name"] for item in files] == ["tile-a", "tile-b"]
    for item in files:
        assert item["nlink"] == 1
        assert item["mode"] == "0600"
        assert item["size"] > 0
        assert len(item["sha256"]) == 64
        assert all(key in item for key in ("dev", "inode", "mtime_ns", "uid", "gid"))


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_plan_refuses_aliased_candidate(tmp_path, monkeypatch, kind):
    index = tmp_path / "lucas_tile_index.parquet"
    _index_bytes(index, [_row("tile", row=10, col=10)])
    data = tmp_path / "tiles"
    data.mkdir()
    target = data / "target.npz"
    target.write_bytes(b"payload")
    candidate = data / "tile.npz"
    if kind == "symlink":
        candidate.symlink_to(target)
    else:
        os.link(target, candidate)
    monkeypatch.setattr(access, "_classify", lambda _identity: access.ACTION_REPAIR)
    payload = index.read_bytes()

    with pytest.raises((access.SourceAccessError, OSError)):
        access.build_plan_record(
            identity=IDENTITY,
            verified_runtime=VERIFIED_RUNTIME,
            source_index=index,
            data_dir=data,
            expected_index_sha256=hashlib.sha256(payload).hexdigest(),
            expected_index_size=len(payload),
            enforce_production_counts=False,
        )


def test_apply_changes_only_metadata_and_is_idempotent(tmp_path, monkeypatch):
    data = tmp_path / "tiles"
    data.mkdir()
    tile = data / "tile.npz"
    np.savez_compressed(tile, value=np.arange(8))
    tile.chmod(0o600)
    data_fd = access._open_directory_tree(data)
    fd = access._open_regular_at(data_fd, tile.name)
    try:
        before = access._identity(fd)
    finally:
        os.close(fd)
        os.close(data_fd)
    planned = {
        "tile_name": "tile",
        "file_name": tile.name,
        "path": str(tile),
        **before,
        "action": access.ACTION_REPAIR,
    }
    plan = {"files": [planned]}

    monkeypatch.setattr(access, "SOURCE_ACCESS_TARGET_UID", os.geteuid())
    monkeypatch.setattr(access, "SOURCE_ACCESS_TARGET_GID", os.getegid())

    def local_classify(identity):
        if identity["mode"] == "0600":
            return access.ACTION_REPAIR
        if identity["mode"] == "0640":
            return access.ACTION_ALREADY_CORRECT
        raise access.SourceAccessError("unexpected local test mode")

    monkeypatch.setattr(access, "_classify", local_classify)
    calls: list[str] = []
    real_fchmod = os.fchmod
    real_fsync = os.fsync

    def tracked_fchown(_fd: int, uid: int, gid: int) -> None:
        calls.append(f"fchown:{uid}:{gid}")

    def tracked_fchmod(file_fd: int, mode: int) -> None:
        calls.append(f"fchmod:{mode:o}")
        real_fchmod(file_fd, mode)

    def tracked_fsync(file_fd: int) -> None:
        calls.append("fsync")
        real_fsync(file_fd)

    monkeypatch.setattr(access.os, "fchown", tracked_fchown)
    monkeypatch.setattr(access.os, "fchmod", tracked_fchmod)
    monkeypatch.setattr(access.os, "fsync", tracked_fsync)

    first = access.apply_plan_record(plan, data_dir=data)
    second = access.apply_plan_record(plan, data_dir=data)

    assert calls == [
        f"fchown:-1:{os.getegid()}",
        "fchmod:640",
        "fsync",
        "fsync",
    ]
    assert first[0]["applied_action"] == "repaired"
    assert first[0]["sha256_unchanged"] is True
    assert first[0]["size_unchanged"] is True
    assert first[0]["mtime_unchanged"] is True
    assert first[0]["inode_unchanged"] is True
    assert first[0]["after"]["uid"] == first[0]["before"]["uid"]
    assert first[0]["ctime_changed"] is True
    assert second[0]["applied_action"] == "already-repaired"
    assert stat.S_IMODE(tile.stat().st_mode) == 0o640
    with np.load(tile) as payload:
        np.testing.assert_array_equal(payload["value"], np.arange(8))


def test_apply_refuses_bytes_changed_after_plan(tmp_path, monkeypatch):
    data = tmp_path / "tiles"
    data.mkdir()
    tile = data / "tile.npz"
    tile.write_bytes(b"before")
    tile.chmod(0o600)
    directory_fd = access._open_directory_tree(data)
    fd = access._open_regular_at(directory_fd, tile.name)
    try:
        before = access._identity(fd)
    finally:
        os.close(fd)
        os.close(directory_fd)
    plan = {"files": [{
        "tile_name": "tile",
        "file_name": tile.name,
        "path": str(tile),
        **before,
        "action": access.ACTION_REPAIR,
    }]}
    tile.write_bytes(b"after")
    monkeypatch.setattr(access, "_classify", lambda _identity: access.ACTION_REPAIR)

    with pytest.raises(access.SourceAccessError, match="differs from plan"):
        access.apply_plan_record(plan, data_dir=data)


def test_apply_preflights_all_files_before_first_metadata_change(
    tmp_path, monkeypatch,
):
    data = tmp_path / "tiles"
    data.mkdir()
    planned_files = []
    for name in ("a", "b"):
        tile = data / f"{name}.npz"
        tile.write_bytes(f"payload-{name}".encode())
        tile.chmod(0o600)
        directory_fd = access._open_directory_tree(data)
        fd = access._open_regular_at(directory_fd, tile.name)
        try:
            identity = access._identity(fd)
        finally:
            os.close(fd)
            os.close(directory_fd)
        planned_files.append({
            "tile_name": name,
            "file_name": tile.name,
            "path": str(tile),
            **identity,
            "action": access.ACTION_REPAIR,
        })
    (data / "b.npz").write_bytes(b"changed-after-plan")
    monkeypatch.setattr(
        access,
        "_classify",
        lambda _identity: access.ACTION_REPAIR,
    )
    monkeypatch.setattr(
        access.os,
        "fchown",
        lambda *_args: pytest.fail("mutation occurred before full preflight"),
    )

    with pytest.raises(access.SourceAccessError, match="differs from plan"):
        access.apply_plan_record({"files": planned_files}, data_dir=data)

    assert stat.S_IMODE((data / "a.npz").stat().st_mode) == 0o600


def test_final_full_rescan_refuses_early_tile_replaced_during_later_work(
    tmp_path, monkeypatch,
):
    data = tmp_path / "tiles"
    data.mkdir()
    planned_files = []
    for name in ("a", "b"):
        tile = data / f"{name}.npz"
        tile.write_bytes(f"payload-{name}".encode())
        tile.chmod(0o600)
        directory_fd = access._open_directory_tree(data)
        fd = access._open_regular_at(directory_fd, tile.name)
        try:
            identity = access._identity(fd)
        finally:
            os.close(fd)
            os.close(directory_fd)
        planned_files.append({
            "tile_name": name,
            "file_name": tile.name,
            "path": str(tile),
            **identity,
            "action": access.ACTION_REPAIR,
        })

    monkeypatch.setattr(access, "SOURCE_ACCESS_TARGET_UID", os.geteuid())
    monkeypatch.setattr(access, "SOURCE_ACCESS_TARGET_GID", os.getegid())
    monkeypatch.setattr(access, "SOURCE_ACCESS_EXPECTED_CANDIDATES", 2)

    def local_classify(identity):
        if identity["mode"] == "0600":
            return access.ACTION_REPAIR
        if identity["mode"] == "0640":
            return access.ACTION_ALREADY_CORRECT
        raise access.SourceAccessError("unexpected local test mode")

    monkeypatch.setattr(access, "_classify", local_classify)
    real_fchmod = os.fchmod
    repairs = 0

    def replace_first_while_second_is_repaired(fd: int, mode: int) -> None:
        nonlocal repairs
        real_fchmod(fd, mode)
        repairs += 1
        if repairs == 2:
            replacement = data / "replacement.npz"
            replacement.write_bytes(b"concurrent replacement")
            os.replace(replacement, data / "a.npz")

    monkeypatch.setattr(access.os, "fchmod", replace_first_while_second_is_repaired)

    results = access.apply_plan_record({"files": planned_files}, data_dir=data)
    assert [item["tile_name"] for item in results] == ["a", "b"]

    completion_path = tmp_path / "completion.json"
    with pytest.raises(
        access.SourceAccessError,
        match="differs from completion after-state: a",
    ):
        access.publish_completion(
            {"files": results},
            completion_path,
            data_dir=data,
        )
    assert not completion_path.exists()


def test_watchdog_loss_stops_apply_before_completion_and_leaves_resumable_state(
    tmp_path, monkeypatch,
):
    data = tmp_path / "tiles"
    data.mkdir()
    planned_files = []
    for name in ("a", "b"):
        tile = data / f"{name}.npz"
        tile.write_bytes(f"payload-{name}".encode())
        tile.chmod(0o600)
        directory_fd = access._open_directory_tree(data)
        fd = access._open_regular_at(directory_fd, tile.name)
        try:
            identity = access._identity(fd)
        finally:
            os.close(fd)
            os.close(directory_fd)
        planned_files.append({
            "tile_name": name,
            "file_name": tile.name,
            "path": str(tile),
            **identity,
            "action": access.ACTION_REPAIR,
        })

    monkeypatch.setattr(access, "SOURCE_ACCESS_TARGET_UID", os.geteuid())
    monkeypatch.setattr(access, "SOURCE_ACCESS_TARGET_GID", os.getegid())

    def local_classify(identity):
        if identity["mode"] == "0600":
            return access.ACTION_REPAIR
        if identity["mode"] == "0640":
            return access.ACTION_ALREADY_CORRECT
        raise access.SourceAccessError("unexpected local test mode")

    monkeypatch.setattr(access, "_classify", local_classify)
    checks = 0

    def lease_check() -> None:
        nonlocal checks
        checks += 1
        if checks == 5:
            raise access.SourceAccessError("watchdog lease is stale")

    with pytest.raises(access.SourceAccessError, match="watchdog lease is stale"):
        access.apply_plan_record(
            {"files": planned_files},
            data_dir=data,
            lease_check=lease_check,
        )

    assert stat.S_IMODE((data / "a.npz").stat().st_mode) == 0o640
    assert stat.S_IMODE((data / "b.npz").stat().st_mode) == 0o600


def test_apply_resumes_after_fchown_succeeds_and_fchmod_fails(
    tmp_path, monkeypatch,
):
    data = tmp_path / "tiles"
    data.mkdir()
    tile = data / "tile.npz"
    tile.write_bytes(b"payload")
    tile.chmod(0o600)
    directory_fd = access._open_directory_tree(data)
    fd = access._open_regular_at(directory_fd, tile.name)
    try:
        before = access._identity(fd)
    finally:
        os.close(fd)
        os.close(directory_fd)
    plan = {"files": [{
        "tile_name": "tile",
        "file_name": tile.name,
        "path": str(tile),
        **before,
        "action": access.ACTION_REPAIR,
    }]}
    monkeypatch.setattr(access, "SOURCE_ACCESS_TARGET_UID", os.geteuid())
    monkeypatch.setattr(access, "SOURCE_ACCESS_TARGET_GID", os.getegid())
    real_fchmod = os.fchmod
    calls = 0

    def fail_once(fd: int, mode: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("simulated fchmod failure")
        real_fchmod(fd, mode)

    monkeypatch.setattr(access.os, "fchmod", fail_once)
    with pytest.raises(PermissionError, match="fchmod"):
        access.apply_plan_record(plan, data_dir=data)

    result = access.apply_plan_record(plan, data_dir=data)
    assert result[0]["applied_action"] == "repaired"
    assert stat.S_IMODE(tile.stat().st_mode) == 0o640


def test_exclusive_dataset_lock_refuses_symlink(tmp_path):
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    target = tmp_path / "target"
    target.write_text("not a lock")
    lock = lock_root / "dataset.lock"
    lock.symlink_to(target)

    with pytest.raises(OSError):
        with access.exclusive_dataset_lock(lock):
            pytest.fail("symlink lock must never be acquired")


def test_strict_dataset_lock_requires_precreated_exact_identity(tmp_path):
    lock = tmp_path / "dataset.lock"
    lock.touch(mode=0o660)
    lock.chmod(0o660)

    with access.exclusive_dataset_lock(
        lock,
        create=False,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
        expected_mode=0o660,
    ):
        assert stat.S_IMODE(lock.stat().st_mode) == 0o660


def test_general_dataset_lock_creation_settles_mode_0660(tmp_path):
    lock = tmp_path / "nested" / "dataset.lock"

    with access.exclusive_dataset_lock(lock):
        identity = lock.stat()
        assert stat.S_ISREG(identity.st_mode)
        assert identity.st_nlink == 1
        assert identity.st_size == 0
        assert stat.S_IMODE(identity.st_mode) == 0o660


def test_strict_dataset_lock_refuses_missing_mode_and_hardlink(tmp_path):
    missing = tmp_path / "missing.lock"
    with pytest.raises(FileNotFoundError):
        with access.exclusive_dataset_lock(
            missing,
            create=False,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
            expected_mode=0o660,
        ):
            pytest.fail("strict workflow must not create its lock")
    assert not missing.exists()

    lock = tmp_path / "dataset.lock"
    lock.touch(mode=0o660)
    lock.chmod(0o640)
    with pytest.raises(RuntimeError, match="lock mode"):
        with access.exclusive_dataset_lock(
            lock,
            create=False,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
            expected_mode=0o660,
        ):
            pytest.fail("wrong-mode lock must not be acquired")

    lock.chmod(0o660)
    alias = tmp_path / "alias.lock"
    os.link(lock, alias)
    with pytest.raises(RuntimeError, match="unaliased"):
        with access.exclusive_dataset_lock(
            lock,
            create=False,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
            expected_mode=0o660,
        ):
            pytest.fail("hard-linked lock must not be acquired")


def test_strict_dataset_lock_refuses_nonempty_and_wrong_ownership(tmp_path):
    lock = tmp_path / "dataset.lock"
    lock.write_bytes(b"not-an-empty-lock")
    lock.chmod(0o660)
    expected = {
        "create": False,
        "expected_uid": os.geteuid(),
        "expected_gid": os.getegid(),
        "expected_mode": 0o660,
    }

    with pytest.raises(RuntimeError, match="empty"):
        with access.exclusive_dataset_lock(lock, **expected):
            pytest.fail("nonempty lock must not be acquired")

    lock.write_bytes(b"")
    with pytest.raises(RuntimeError, match="lock UID"):
        with access.exclusive_dataset_lock(
            lock,
            **{**expected, "expected_uid": os.geteuid() + 1},
        ):
            pytest.fail("wrong-owner lock must not be acquired")
    with pytest.raises(RuntimeError, match="lock GID"):
        with access.exclusive_dataset_lock(
            lock,
            **{**expected, "expected_gid": os.getegid() + 1},
        ):
            pytest.fail("wrong-group lock must not be acquired")


def test_completion_verifier_rejects_forged_unchanged_summary(
    tmp_path, monkeypatch,
):
    identity_before = {
        "dev": 1,
        "inode": 2,
        "size": 7,
        "mtime_ns": 3,
        "ctime_ns": 4,
        "uid": 0,
        "gid": 0,
        "mode": "0600",
        "nlink": 1,
        "sha256": "a" * 64,
    }
    identity_after = {
        **identity_before,
        "ctime_ns": 5,
        "gid": 2000,
        "mode": "0640",
        "sha256": "b" * 64,
    }
    record = {
        "schema": access.COMPLETION_SCHEMA,
        "pod_uid": "apply-pod",
        "status": "completed",
        "runtime": {
            "source_git_sha": IDENTITY.source_git_sha,
            "image_ref": IDENTITY.image_ref,
            "runtime_manifest_sha256": "c" * 64,
            "source_payload_sha256": "d" * 64,
        },
        "process_identity": {"effective_uid": 0, "effective_gid": 2000},
        "plan": {"pod_uid": "plan-pod", "sha256": "e" * 64},
        "source_index": {
            "path": "/index.parquet",
            **{**identity_before, "size": 10, "sha256": "f" * 64},
        },
        "summary": {
            "files": 1,
            "repaired": 1,
            "already_repaired": 0,
            "no_op": 0,
            "content_unchanged": True,
            "ctime_policy": "changed-on-repair; unchanged permitted on idempotent no-op",
        },
        "files": [{
            "tile_name": "tile",
            "planned_action": access.ACTION_REPAIR,
            "applied_action": "repaired",
            "before": identity_before,
            "after": identity_after,
            "sha256_unchanged": True,
            "size_unchanged": True,
            "mtime_unchanged": True,
            "inode_unchanged": True,
            "ctime_changed": True,
        }],
    }
    path = tmp_path / "completion.json"
    payload = access.canonical_json_bytes(record)
    path.write_bytes(payload)
    monkeypatch.setattr(access, "SOURCE_ACCESS_INDEX_INPUT", Path("/index.parquet"))
    monkeypatch.setattr(access, "SOURCE_ACCESS_INDEX_SHA256", "f" * 64)
    monkeypatch.setattr(access, "SOURCE_ACCESS_INDEX_SIZE", 10)
    monkeypatch.setattr(access, "SOURCE_ACCESS_EXPECTED_CANDIDATES", 1)
    monkeypatch.setattr(access, "SOURCE_ACCESS_EXPECTED_REPAIRS", 1)
    monkeypatch.setattr(access, "SOURCE_ACCESS_EXPECTED_NOOPS", 0)

    with pytest.raises(access.SourceAccessError, match="runtime binding mismatch"):
        access.verify_completion(
            path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_source_git_sha=IDENTITY.source_git_sha,
            expected_image_ref=IDENTITY.image_ref,
            expected_completion_pod_uid="apply-pod",
            expected_plan_sha256="e" * 64,
            expected_runtime_binding={
                **record["runtime"],
                "runtime_manifest_sha256": "0" * 64,
            },
        )

    with pytest.raises(access.SourceAccessError, match="proof is inconsistent"):
        access.verify_completion(
            path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_source_git_sha=IDENTITY.source_git_sha,
            expected_image_ref=IDENTITY.image_ref,
            expected_completion_pod_uid="apply-pod",
            expected_plan_sha256="e" * 64,
            expected_runtime_binding=record["runtime"],
        )
