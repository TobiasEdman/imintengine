#!/usr/bin/env python3
"""Hold and continuously verify the external LUCAS source-data freeze.

This is an operator-side protocol.  It suspends the known Pod-producing
CronJobs with resourceVersion compare-and-swap, inventories every standard
namespaced Pod producer, and refreshes a short-lived ConfigMap lease only
after a clean structural scan.  PLAN, APPLY, and split Pods mount that lease
read-only and refuse stale, failed, or wrong-phase records without receiving a
service-account token or network access.

The tool is intentionally not an admission controller.  A namespace admin can
create or mutate a workload between watchdog polls; every such interval is a
documented residual race, bounded by the lease lifetime and the payload's
repeated lease checks.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import secrets
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

if __package__:
    from .crop_distill_provenance import (
        canonical_json_bytes as workload_canonical_json_bytes,
    )
else:
    from crop_distill_provenance import (
        canonical_json_bytes as workload_canonical_json_bytes,
    )

FREEZE_SCHEMA = "imint-crop-source-freeze-state-v1"
LEASE_SCHEMA = "imint-crop-source-freeze-lease-v1"
PHASE_STATE_SCHEMA = "imint-crop-source-freeze-phase-v1"
LEASE_CONFIGMAP = "crop-source-freeze-lease"
PVC_CLAIM = "training-data-cephfs"
DATASET_SUBPATH = PurePosixPath("unified_v2_512")
NAMESPACE = "prithvi-training-default"
CONTEXT = "icekube"
LEASE_SECONDS = 180
DEFAULT_INTERVAL_SECONDS = 15.0
KUBECTL_READ_ATTEMPTS = 8
KUBECTL_READ_RETRY_BASE_SECONDS = 1.0
# PLAN and APPLY have activeDeadlineSeconds=7200 (split is 3600). A gate
# request lives just long enough for the longest deadline plus the maximum
# projected-ConfigMap lease lag, never indefinitely if the gate process or
# operator host dies after arming a phase.
PHASE_REQUEST_SECONDS = 7500
LEASE_FIELDS = frozenset({
    "schema",
    "run_id",
    "phase",
    "status",
    "sequence",
    "heartbeat_unix_ns",
    "valid_until_unix_ns",
    "snapshot_sha256",
    "controller_snapshot_sha256",
})

SUSPEND_CONTROLLERS = ("ladder-queue", "gpu-reaper")
OBSERVE_CONTROLLERS = ("campaign-orchestrator",)
ALL_CONTROLLERS = SUSPEND_CONTROLLERS + OBSERVE_CONTROLLERS
RESOURCE_TYPES = (
    "pods",
    "jobs",
    "cronjobs",
    "deployments",
    "statefulsets",
    "daemonsets",
    "replicasets",
    "replicationcontrollers",
)
PHASE_JOB = {
    "idle": None,
    "plan": ("ladder-crop-source-access-plan", "ladder-crop-source-access-plan"),
    "apply": ("ladder-crop-source-access-apply", "ladder-crop-source-access-apply"),
    "split": ("ladder-lucas-crop-split", "ladder-crop-distill"),
}
APPLY_ALLOWED_OVERLAP = {
    "container_field": "containers",
    "container": "source-access-apply",
    "volume": "training-data-cephfs",
    "access": "rw-mount",
    "sub_path": "unified_v2_512",
}


class FreezeError(RuntimeError):
    """The external freeze cannot safely be acquired or retained."""


class CoordinationBusy(FreezeError):
    """Another local protocol actor currently owns a nonblocking lock."""


_TRANSIENT_KUBECTL_READ_ERRORS = (
    "connection reset by peer",
    "network is unreachable",
    "unable to connect to the server",
    "unexpected error when reading response body",
    "tls handshake timeout",
    "i/o timeout",
)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _new_phase_state(*, run_id: str, phase: str) -> dict[str, object]:
    if phase not in PHASE_JOB:
        raise FreezeError(f"unknown freeze phase: {phase}")
    return {
        "schema": PHASE_STATE_SCHEMA,
        "run_id": run_id,
        "phase": phase,
        "request_id": secrets.token_hex(32),
        "valid_until_unix_ns": (
            0
            if phase == "idle"
            else time.time_ns() + PHASE_REQUEST_SECONDS * 1_000_000_000
        ),
    }


def _validate_phase_state(
    value: Mapping[str, object],
    *,
    run_id: str,
    now_ns: int | None = None,
    allow_expired: bool = False,
) -> str:
    if set(value) != {
        "schema",
        "run_id",
        "phase",
        "request_id",
        "valid_until_unix_ns",
    } or (
        value.get("schema") != PHASE_STATE_SCHEMA
        or value.get("run_id") != run_id
    ):
        raise FreezeError("watchdog phase request is malformed")
    phase = value.get("phase")
    request_id = value.get("request_id")
    valid_until = value.get("valid_until_unix_ns")
    if (
        not isinstance(phase, str)
        or phase not in PHASE_JOB
        or not isinstance(request_id, str)
        or len(request_id) != 64
        or any(ch not in "0123456789abcdef" for ch in request_id)
        or type(valid_until) is not int
    ):
        raise FreezeError("watchdog phase request values are malformed")
    current = time.time_ns() if now_ns is None else now_ns
    if phase == "idle":
        if valid_until != 0:
            raise FreezeError("idle watchdog phase must not carry an expiry")
    elif valid_until <= current and not allow_expired:
        raise FreezeError("watchdog phase request expired")
    return phase


def _phase_state_expired(
    value: Mapping[str, object],
    *,
    run_id: str,
    now_ns: int | None = None,
) -> bool:
    """Validate a phase record and report a terminal non-idle expiry."""
    current = time.time_ns() if now_ns is None else now_ns
    phase = _validate_phase_state(
        value,
        run_id=run_id,
        now_ns=current,
        allow_expired=True,
    )
    return phase != "idle" and int(value["valid_until_unix_ns"]) <= current


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once_json(path: Path, value: object) -> str:
    """Atomically publish one mode-0600 canonical record without overwrite."""
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    temporary_name = f".{path.name}.{secrets.token_hex(12)}.create"
    fd: int | None = None
    try:
        fd = os.open(
            temporary_name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(fd, 0o600)
            os.fsync(fd)
        # Hard-link publication is the portable no-replace primitive. A crash
        # can leave a complete private temporary link, never a partial final
        # record and never an overwrite of an existing immutable record.
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_name = ""
        os.fsync(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)
    return hashlib.sha256(payload).hexdigest()


def replace_json(path: Path, value: object) -> None:
    """Durably replace mutable local watchdog coordination state."""
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    directory_fd: int | None = None
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(fd)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        os.replace(
            temporary.name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(fd)
    _fsync_directory(path.parent)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"cannot read canonical state {path}: {exc}") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != path.read_bytes():
        raise FreezeError(f"state is not canonical JSON: {path}")
    return value


def _coordination_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_uid),
        int(value.st_gid),
        stat.S_IMODE(value.st_mode),
        int(value.st_nlink),
    )


@contextmanager
def _coordination_owner(
    run_dir: Path,
    *,
    name: str,
    conflict: str,
    nonblocking: bool = True,
):
    """Hold one strict, unaliased local coordination inode without waiting."""
    path = run_dir / name
    flags = (
        os.O_RDWR
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    created = False
    try:
        fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        fd = os.open(path, flags)
    locked = False
    try:
        if created:
            os.fchmod(fd, 0o600)
            os.fsync(fd)
            _fsync_directory(run_dir)
        identity = os.fstat(fd)
        path_identity = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(identity.st_mode)
            or identity.st_nlink != 1
            or identity.st_size != 0
            or identity.st_uid != os.geteuid()
            or stat.S_IMODE(identity.st_mode) != 0o600
            or _coordination_identity(path_identity)
            != _coordination_identity(identity)
        ):
            raise FreezeError("coordination lock identity is unsafe")
        try:
            operation = fcntl.LOCK_EX
            if nonblocking:
                operation |= fcntl.LOCK_NB
            fcntl.flock(fd, operation)
        except BlockingIOError as exc:
            raise CoordinationBusy(conflict) from exc
        locked = True
        after = os.fstat(fd)
        path_after = os.stat(path, follow_symlinks=False)
        if (
            _coordination_identity(after) != _coordination_identity(identity)
            or _coordination_identity(path_after) != _coordination_identity(after)
        ):
            raise FreezeError("coordination lock changed during acquisition")
        yield
    finally:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def _watchdog_owner(run_dir: Path):
    """Hold the one local watchdog/restore ownership lock for this run."""
    with _coordination_owner(
        run_dir,
        name="watchdog-owner.lock",
        conflict="another watchdog or restore owns this run",
    ):
        yield


@contextmanager
def _phase_owner(run_dir: Path):
    """Serialize phase gates and exact restore for one run."""
    with _coordination_owner(
        run_dir,
        name="phase-owner.lock",
        conflict="another phase gate or restore owns this run",
    ):
        yield


@contextmanager
def _phase_publication_owner(run_dir: Path):
    """Serialize short phase-state and live-lease publication commits."""
    with _coordination_owner(
        run_dir,
        name="phase-publication.lock",
        conflict="phase publication lock is unavailable",
        nonblocking=False,
    ):
        yield


@contextmanager
def _hold_owner(run_dir: Path):
    """Exclude exact restore until hold has either completed or failed."""
    with _coordination_owner(
        run_dir,
        name="hold-owner.lock",
        conflict="another hold or restore owns this run",
    ):
        yield


def _begin_restore(run_dir: Path) -> None:
    """Atomically exclude both a live and every future watchdog invocation."""
    marker = run_dir / "restore-in-progress.json"
    expected = {
        "schema": FREEZE_SCHEMA,
        "run_id": run_dir.name,
        "status": "restore-in-progress",
    }
    with _watchdog_owner(run_dir):
        if marker.exists():
            if _read_json(marker) != expected:
                raise FreezeError("restore ownership marker is malformed")
        else:
            write_once_json(marker, expected)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_hold_record_hashes(run_dir: Path) -> dict[str, Any]:
    complete = _read_json(run_dir / "hold-complete.json")
    if set(complete) != {
        "schema",
        "run_id",
        "controllers_before_sha256",
        "controllers_held_sha256",
        "initial_snapshot_sha256",
    } or (
        complete.get("schema") != FREEZE_SCHEMA
        or complete.get("run_id") != run_dir.name
    ):
        raise FreezeError("hold-complete record is malformed")
    authorities = {
        "controllers_before_sha256": run_dir / "controllers-before.json",
        "controllers_held_sha256": run_dir / "controllers-held.json",
        "initial_snapshot_sha256": run_dir / "snapshots/00000000.json",
    }
    for field, path in authorities.items():
        try:
            actual = _sha256_path(path)
        except OSError as exc:
            raise FreezeError(f"hold authority is unavailable: {path}") from exc
        if complete.get(field) != actual:
            raise FreezeError(f"hold authority hash mismatch: {path}")
    return complete


def _verify_clean_idle_snapshot(
    run_dir: Path,
    *,
    sequence: int,
    expected_sha256: object,
) -> dict[str, Any]:
    snapshot_path = run_dir / "snapshots" / f"{sequence:08d}.json"
    if (
        not isinstance(expected_sha256, str)
        or _sha256_path(snapshot_path) != expected_sha256
    ):
        raise FreezeError("clean idle snapshot hash differs")
    snapshot = _read_json(snapshot_path)
    if set(snapshot) != {
        "schema",
        "run_id",
        "phase",
        "sequence",
        "captured_unix_ns",
        "controllers",
        "inventory",
        "violations",
    } or (
        snapshot.get("schema") != FREEZE_SCHEMA
        or snapshot.get("run_id") != run_dir.name
        or snapshot.get("phase") != "idle"
        or snapshot.get("sequence") != sequence
        or type(snapshot.get("captured_unix_ns")) is not int
        or not isinstance(snapshot.get("controllers"), list)
        or not isinstance(snapshot.get("inventory"), list)
        or snapshot.get("violations") != []
    ):
        raise FreezeError("clean idle snapshot is malformed")
    return snapshot


def _verify_expired_phase_stop(
    run_dir: Path,
    lease: Mapping[str, object],
) -> None:
    """Verify the clean idle scan that permits restore after phase expiry."""
    stopped = _read_json(run_dir / "watchdog-stopped.json")
    expected_fields = {
        "schema",
        "run_id",
        "sequence",
        "reason",
        "expired_phase",
        "phase_request_id",
        "phase_valid_until_unix_ns",
        "snapshot_sha256",
    }
    if set(stopped) != expected_fields or (
        stopped.get("schema") != FREEZE_SCHEMA
        or stopped.get("run_id") != run_dir.name
        or stopped.get("reason") != "phase-expired-clean-idle-scan"
        or type(stopped.get("sequence")) is not int
        or not isinstance(stopped.get("expired_phase"), str)
        or not isinstance(stopped.get("phase_request_id"), str)
        or type(stopped.get("phase_valid_until_unix_ns")) is not int
        or not isinstance(stopped.get("snapshot_sha256"), str)
    ):
        raise FreezeError("expired-phase watchdog stop record is malformed")

    phase_state = _read_json(run_dir / "phase.json")
    if not _phase_state_expired(phase_state, run_id=run_dir.name):
        raise FreezeError("expired-phase watchdog stop is not currently expired")
    if (
        phase_state.get("phase") != stopped["expired_phase"]
        or phase_state.get("request_id") != stopped["phase_request_id"]
        or phase_state.get("valid_until_unix_ns")
        != stopped["phase_valid_until_unix_ns"]
    ):
        raise FreezeError("expired-phase watchdog stop authority differs")

    sequence = int(stopped["sequence"])
    _verify_clean_idle_snapshot(
        run_dir,
        sequence=sequence,
        expected_sha256=stopped["snapshot_sha256"],
    )

    if (
        lease.get("status") != "failed"
        or lease.get("phase") != stopped["expired_phase"]
        or lease.get("sequence") != sequence
        or lease.get("snapshot_sha256") != stopped["snapshot_sha256"]
        or lease.get("valid_until_unix_ns") != lease.get("heartbeat_unix_ns")
    ):
        raise FreezeError("expired-phase failed lease differs from clean stop")


def _verify_restore_requested_stop(
    run_dir: Path,
    lease: Mapping[str, object],
) -> None:
    """Verify the exact idle heartbeat used for a restore-coordinated stop."""
    stopped = _read_json(run_dir / "watchdog-stopped.json")
    if set(stopped) != {
        "schema",
        "run_id",
        "sequence",
        "reason",
        "phase_request_id",
        "snapshot_sha256",
    } or (
        stopped.get("schema") != FREEZE_SCHEMA
        or stopped.get("run_id") != run_dir.name
        or stopped.get("reason") != "restore-requested-clean-idle-scan"
        or type(stopped.get("sequence")) is not int
        or not isinstance(stopped.get("phase_request_id"), str)
        or not isinstance(stopped.get("snapshot_sha256"), str)
    ):
        raise FreezeError("restore-requested watchdog stop record is malformed")

    stop_request = _read_json(run_dir / "stop-requested.json")
    if stop_request != {"schema": FREEZE_SCHEMA, "run_id": run_dir.name}:
        raise FreezeError("watchdog stop request is malformed")
    phase_state = _read_json(run_dir / "phase.json")
    if (
        _validate_phase_state(phase_state, run_id=run_dir.name) != "idle"
        or phase_state.get("request_id") != stopped["phase_request_id"]
    ):
        raise FreezeError("restore-requested stop phase authority differs")

    sequence = int(stopped["sequence"])
    _verify_clean_idle_snapshot(
        run_dir,
        sequence=sequence,
        expected_sha256=stopped["snapshot_sha256"],
    )
    if (
        lease.get("status") != "closed"
        or lease.get("phase") != "idle"
        or lease.get("sequence") != sequence
        or lease.get("snapshot_sha256") != stopped["snapshot_sha256"]
        or lease.get("controller_snapshot_sha256")
        != _sha256_path(run_dir / "controllers-held.json")
        or lease.get("valid_until_unix_ns") != lease.get("heartbeat_unix_ns")
    ):
        raise FreezeError("restore-requested closed lease differs from clean stop")


class Kubectl:
    """Small JSON-only kubectl adapter; all writes retain resourceVersion CAS."""

    def __init__(self, *, context: str, namespace: str) -> None:
        self.context = context
        self.namespace = namespace

    def _run(self, args: Sequence[str], *, stdin: bytes | None = None) -> dict[str, Any]:
        command = [
            "kubectl",
            "--context",
            self.context,
            "-n",
            self.namespace,
            *args,
        ]
        attempts = KUBECTL_READ_ATTEMPTS if args and args[0] == "get" else 1
        for attempt in range(attempts):
            result = subprocess.run(
                command,
                input=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            stderr = result.stderr.decode("utf-8", "replace")
            retryable = any(
                marker in stderr.lower()
                for marker in _TRANSIENT_KUBECTL_READ_ERRORS
            )
            if result.returncode == 0 or not retryable or attempt + 1 == attempts:
                break
            time.sleep(
                min(KUBECTL_READ_RETRY_BASE_SECONDS * (2**attempt), 5.0)
            )
        if result.returncode != 0:
            raise FreezeError(
                f"kubectl {' '.join(args)} failed: "
                f"{stderr.strip()}"
            )
        try:
            value = json.loads(result.stdout)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise FreezeError("kubectl returned non-JSON output") from exc
        if not isinstance(value, dict):
            raise FreezeError("kubectl returned a non-object JSON value")
        return value

    def get(self, resource: str, name: str | None = None) -> dict[str, Any]:
        args = ["get", resource]
        if name is not None:
            args.append(name)
        args.extend(["-o", "json"])
        return self._run(args)

    def create(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return self._run(
            ["create", "--validate=false", "-f", "-", "-o", "json"],
            stdin=canonical_json_bytes(value),
        )

    def replace(self, value: Mapping[str, Any]) -> dict[str, Any]:
        # The caller passes the exact object returned by GET, including its
        # resourceVersion.  Kubernetes rejects an intervening update.
        return self._run(
            ["replace", "--validate=false", "-f", "-", "-o", "json"],
            stdin=canonical_json_bytes(value),
        )

    def inventory(self) -> list[dict[str, Any]]:
        result = self.get(",".join(RESOURCE_TYPES))
        items = result.get("items")
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise FreezeError("cluster inventory is not a Kubernetes List")
        return items


def _metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = value.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _kind(value: Mapping[str, Any]) -> str:
    return str(value.get("kind", ""))


def _terminal_pod(value: Mapping[str, Any]) -> bool:
    status = value.get("status")
    return isinstance(status, Mapping) and status.get("phase") in {"Succeeded", "Failed"}


def _terminal_job(value: Mapping[str, Any]) -> bool:
    status = value.get("status")
    conditions = status.get("conditions", []) if isinstance(status, Mapping) else []
    return any(
        isinstance(condition, Mapping)
        and condition.get("type") in {"Complete", "Failed"}
        and condition.get("status") == "True"
        for condition in conditions
    )


def _pod_spec(value: Mapping[str, Any]) -> Mapping[str, Any] | None:
    spec = value.get("spec")
    if not isinstance(spec, Mapping):
        return None
    kind = _kind(value)
    if kind == "Pod":
        return spec
    if kind == "CronJob":
        job_template = spec.get("jobTemplate")
        job_spec = job_template.get("spec") if isinstance(job_template, Mapping) else None
        template = job_spec.get("template") if isinstance(job_spec, Mapping) else None
    else:
        template = spec.get("template")
    pod_spec = template.get("spec") if isinstance(template, Mapping) else None
    return pod_spec if isinstance(pod_spec, Mapping) else None


def _normalise_subpath(value: object) -> PurePosixPath | None:
    if not isinstance(value, str) or not value:
        return PurePosixPath(".")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    parts = tuple(part for part in candidate.parts if part not in {"", "."})
    return PurePosixPath(*parts) if parts else PurePosixPath(".")


def _paths_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    if left == PurePosixPath(".") or right == PurePosixPath("."):
        return True
    left_parts = left.parts
    right_parts = right.parts
    shortest = min(len(left_parts), len(right_parts))
    return left_parts[:shortest] == right_parts[:shortest]


def pod_spec_rw_overlaps(pod_spec: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return every effective RW mount/device overlapping the source dataset."""
    volumes = pod_spec.get("volumes", [])
    claim_volumes: dict[str, bool] = {}
    if isinstance(volumes, list):
        for volume in volumes:
            if not isinstance(volume, Mapping):
                continue
            pvc = volume.get("persistentVolumeClaim")
            if isinstance(pvc, Mapping) and pvc.get("claimName") == PVC_CLAIM:
                name = volume.get("name")
                if isinstance(name, str):
                    claim_volumes[name] = bool(pvc.get("readOnly", False))

    overlaps: list[dict[str, str]] = []
    for container_field in ("initContainers", "containers", "ephemeralContainers"):
        containers = pod_spec.get(container_field, [])
        if not isinstance(containers, list):
            continue
        for container in containers:
            if not isinstance(container, Mapping):
                continue
            container_name = str(container.get("name", "<unnamed>"))
            mounts = container.get("volumeMounts", [])
            if isinstance(mounts, list):
                for mount in mounts:
                    if not isinstance(mount, Mapping):
                        continue
                    volume_name = mount.get("name")
                    if volume_name not in claim_volumes:
                        continue
                    if claim_volumes[str(volume_name)] or bool(mount.get("readOnly", False)):
                        continue
                    if "subPathExpr" in mount:
                        overlaps.append({
                            "container_field": container_field,
                            "container": container_name,
                            "volume": str(volume_name),
                            "access": "rw-unresolved-subPathExpr",
                            "sub_path": str(mount.get("subPathExpr")),
                        })
                        continue
                    subpath = _normalise_subpath(mount.get("subPath", ""))
                    if subpath is None or _paths_overlap(subpath, DATASET_SUBPATH):
                        overlaps.append({
                            "container_field": container_field,
                            "container": container_name,
                            "volume": str(volume_name),
                            "access": "rw-mount",
                            "sub_path": str(mount.get("subPath", "")),
                        })
            devices = container.get("volumeDevices", [])
            if isinstance(devices, list):
                for device in devices:
                    if not isinstance(device, Mapping):
                        continue
                    volume_name = device.get("name")
                    if volume_name in claim_volumes and not claim_volumes[str(volume_name)]:
                        overlaps.append({
                            "container_field": container_field,
                            "container": container_name,
                            "volume": str(volume_name),
                            "access": "rw-volume-device",
                            "sub_path": "<block-device>",
                        })
    return overlaps


def _owner_job_uid(pod: Mapping[str, Any]) -> tuple[str, str] | None:
    refs = _metadata(pod).get("ownerReferences", [])
    if not isinstance(refs, list):
        return None
    controlling = [
        ref
        for ref in refs
        if isinstance(ref, Mapping)
        and ref.get("kind") == "Job"
        and ref.get("controller") is True
    ]
    if len(controlling) != 1:
        return None
    name, uid = controlling[0].get("name"), controlling[0].get("uid")
    if not isinstance(name, str) or not isinstance(uid, str):
        return None
    return name, uid


def _is_exact_apply_overlap(overlaps: Sequence[Mapping[str, str]]) -> bool:
    """Permit only APPLY's one reviewed dataset mount, never a broader alias."""
    return len(overlaps) == 1 and dict(overlaps[0]) == APPLY_ALLOWED_OVERLAP


def find_rw_overlap_violations(
    items: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    held_controllers: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    """Scan nonterminal workloads and every standard Pod template."""
    if phase not in PHASE_JOB:
        raise FreezeError(f"unknown freeze phase: {phase}")
    allowed = PHASE_JOB[phase]
    jobs_by_name = {
        str(_metadata(item).get("name")): item
        for item in items
        if _kind(item) == "Job"
    }
    allowed_job_uid: str | None = None
    # PLAN and split need no source-data RW overlap and therefore receive no
    # exception at all. APPLY is exceptional only if its live Job template has
    # exactly the reviewed container/volume/subPath overlap shape.
    if phase == "apply" and allowed is not None:
        allowed_job = jobs_by_name.get(allowed[0])
        allowed_job_spec = (
            _pod_spec(allowed_job) if allowed_job is not None else None
        )
        allowed_job_labels = (
            _metadata(allowed_job).get("labels")
            if allowed_job is not None
            else None
        )
        if (
            allowed_job is not None
            and not _terminal_job(allowed_job)
            and isinstance(allowed_job_spec, Mapping)
            and isinstance(allowed_job_labels, Mapping)
            and allowed_job_labels.get("purpose") == allowed[1]
            and _is_exact_apply_overlap(
                pod_spec_rw_overlaps(allowed_job_spec)
            )
        ):
            uid = _metadata(allowed_job).get("uid")
            allowed_job_uid = str(uid) if isinstance(uid, str) else None

    violations: list[dict[str, object]] = []
    for item in items:
        kind = _kind(item)
        metadata = _metadata(item)
        name = str(metadata.get("name", "<unnamed>"))
        if kind == "Pod" and _terminal_pod(item):
            continue
        if kind == "Job" and _terminal_job(item):
            continue
        pod_spec = _pod_spec(item)
        if pod_spec is None:
            continue
        overlaps = pod_spec_rw_overlaps(pod_spec)
        if not overlaps:
            continue

        labels = metadata.get("labels")
        purpose = labels.get("purpose") if isinstance(labels, Mapping) else None
        is_allowed = False
        if phase == "apply" and allowed is not None and kind == "Job":
            is_allowed = (
                name == allowed[0]
                and purpose == allowed[1]
                and _is_exact_apply_overlap(overlaps)
            )
        elif (
            phase == "apply"
            and allowed is not None
            and kind == "Pod"
            and allowed_job_uid is not None
        ):
            is_allowed = (
                _owner_job_uid(item) == (allowed[0], allowed_job_uid)
                and purpose == allowed[1]
                and _is_exact_apply_overlap(overlaps)
            )
        elif kind == "CronJob" and name in held_controllers:
            held = held_controllers[name]
            is_allowed = (
                metadata.get("uid") == _metadata(held).get("uid")
                and isinstance(item.get("spec"), Mapping)
                and item["spec"].get("suspend") is True
                and item.get("spec") == held.get("spec")
            )

        if not is_allowed:
            violations.append({
                "kind": kind,
                "name": name,
                "uid": str(metadata.get("uid", "")),
                "resource_version": str(metadata.get("resourceVersion", "")),
                "overlaps": overlaps,
            })
    return sorted(violations, key=lambda value: (str(value["kind"]), str(value["name"])))


def _controller_record(value: Mapping[str, Any]) -> dict[str, object]:
    metadata = _metadata(value)
    spec = value.get("spec")
    if not isinstance(spec, Mapping):
        raise FreezeError("CronJob lacks spec")
    if "suspend" in spec and type(spec.get("suspend")) is not bool:
        raise FreezeError("CronJob spec.suspend is not boolean")
    return {
        "name": str(metadata.get("name", "")),
        "uid": str(metadata.get("uid", "")),
        "resource_version": str(metadata.get("resourceVersion", "")),
        "prior_suspend_present": "suspend" in spec,
        "prior_suspend": spec.get("suspend") if "suspend" in spec else None,
        "object": value,
    }


def _held_map(run_dir: Path) -> dict[str, Mapping[str, Any]]:
    state = _read_json(run_dir / "controllers-held.json")
    controllers = state.get("controllers")
    if not isinstance(controllers, list):
        raise FreezeError("held-controller record is malformed")
    result: dict[str, Mapping[str, Any]] = {}
    for entry in controllers:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("object"), Mapping):
            raise FreezeError("held-controller entry is malformed")
        result[str(entry.get("name"))] = entry["object"]
    if set(result) != set(ALL_CONTROLLERS):
        raise FreezeError("held-controller set is incomplete")
    return result


def _validate_held_controllers(
    client: Kubectl, held: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    current: list[dict[str, Any]] = []
    for name in ALL_CONTROLLERS:
        live = client.get("cronjob", name)
        expected = held[name]
        if _metadata(live).get("uid") != _metadata(expected).get("uid"):
            raise FreezeError(f"CronJob UID changed while frozen: {name}")
        if live.get("spec") != expected.get("spec"):
            raise FreezeError(f"CronJob spec drifted while frozen: {name}")
        if name in SUSPEND_CONTROLLERS and live.get("spec", {}).get("suspend") is not True:
            raise FreezeError(f"CronJob is no longer suspended: {name}")
        current.append(live)
    return current


def _lease_record(
    *,
    run_id: str,
    phase: str,
    sequence: int,
    snapshot_sha256: str,
    controller_snapshot_sha256: str,
    status: str = "held",
    now_ns: int | None = None,
) -> dict[str, object]:
    now = time.time_ns() if now_ns is None else now_ns
    return {
        "schema": LEASE_SCHEMA,
        "run_id": run_id,
        "phase": phase,
        "status": status,
        "sequence": sequence,
        "heartbeat_unix_ns": now,
        "valid_until_unix_ns": now + LEASE_SECONDS * 1_000_000_000,
        "snapshot_sha256": snapshot_sha256,
        "controller_snapshot_sha256": controller_snapshot_sha256,
    }


def _lease_configmap(
    lease: Mapping[str, object], *, namespace: str = NAMESPACE
) -> dict[str, object]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": LEASE_CONFIGMAP,
            "namespace": namespace,
            "labels": {
                "app": "unified-training",
                "purpose": "crop-source-freeze-lease",
            },
        },
        "data": {
            "lease.json": workload_canonical_json_bytes(lease).decode("utf-8")
        },
    }


def _replace_lease(client: Kubectl, lease: Mapping[str, object]) -> dict[str, Any]:
    live = client.get("configmap", LEASE_CONFIGMAP)
    try:
        prior = json.loads(live.get("data", {}).get("lease.json", ""))
    except (TypeError, json.JSONDecodeError) as exc:
        raise FreezeError("live freeze lease is malformed") from exc
    if not isinstance(prior, dict) or prior.get("run_id") != lease.get("run_id"):
        raise FreezeError("live freeze lease belongs to another run")
    replacement = _lease_configmap(lease, namespace=client.namespace)
    replacement["metadata"]["resourceVersion"] = _metadata(live).get(
        "resourceVersion"
    )
    return client.replace(replacement)


def _install_initial_lease(
    client: Kubectl, lease: Mapping[str, object]
) -> dict[str, Any]:
    """Create the fixed lease, or CAS-reuse only a cleanly released one."""
    try:
        live = client.get("configmap", LEASE_CONFIGMAP)
    except FreezeError as exc:
        if "not found" not in str(exc).lower() and "(notfound)" not in str(exc).lower():
            raise
        return client.create(_lease_configmap(lease, namespace=client.namespace))
    try:
        prior = json.loads(live.get("data", {}).get("lease.json", ""))
    except (TypeError, json.JSONDecodeError) as exc:
        raise FreezeError("existing freeze lease is malformed") from exc
    if not isinstance(prior, dict) or prior.get("status") != "released":
        raise FreezeError(
            "existing freeze lease was not cleanly released; recover its run first"
        )
    replacement = _lease_configmap(lease, namespace=client.namespace)
    replacement["metadata"]["resourceVersion"] = _metadata(live).get(
        "resourceVersion"
    )
    return client.replace(replacement)


def _preflight_lease_slot(client: Kubectl) -> None:
    """Refuse controller mutation while another run owns the fixed lease."""
    try:
        live = client.get("configmap", LEASE_CONFIGMAP)
    except FreezeError as exc:
        if "not found" in str(exc).lower() or "(notfound)" in str(exc).lower():
            return
        raise
    try:
        prior = json.loads(live.get("data", {}).get("lease.json", ""))
    except (TypeError, json.JSONDecodeError) as exc:
        raise FreezeError("existing freeze lease is malformed") from exc
    if not isinstance(prior, dict) or prior.get("status") != "released":
        raise FreezeError(
            "another or incompletely restored freeze run owns the live lease"
        )


def _snapshot(
    *,
    run_id: str,
    phase: str,
    sequence: int,
    controllers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    violations: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    return {
        "schema": FREEZE_SCHEMA,
        "run_id": run_id,
        "phase": phase,
        "sequence": sequence,
        "captured_unix_ns": time.time_ns(),
        "controllers": list(controllers),
        "inventory": list(items),
        "violations": list(violations),
    }


def hold(client: Kubectl, *, state_dir: Path, run_id: str) -> Path:
    if not run_id or not all(ch.isalnum() or ch in "-." for ch in run_id):
        raise FreezeError("run-id must contain only alphanumeric, dot, or dash")
    _preflight_lease_slot(client)
    run_dir = state_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    _fsync_directory(run_dir.parent)
    hold_ownership = ExitStack()
    hold_ownership.enter_context(_hold_owner(run_dir))

    lease_installed = False
    before_sha = "0" * 64
    try:
        before = [
            _controller_record(client.get("cronjob", name))
            for name in ALL_CONTROLLERS
        ]
        campaign = next(
            entry for entry in before if entry["name"] == "campaign-orchestrator"
        )
        if campaign["prior_suspend"] is not True:
            raise FreezeError("campaign-orchestrator must already be suspended")
        before_sha = write_once_json(
            run_dir / "controllers-before.json",
            {"schema": FREEZE_SCHEMA, "run_id": run_id, "controllers": before},
        )

        # Claim the single live lease slot before the first controller CAS.
        # The initializing status is deliberately never accepted by data Jobs.
        # If this process dies mid-hold, `restore` can replay the immutable
        # before record and converge a partially suspended controller set.
        initializing = _lease_record(
            run_id=run_id,
            phase="idle",
            sequence=0,
            snapshot_sha256="0" * 64,
            controller_snapshot_sha256=before_sha,
            status="initializing",
        )
        initializing["valid_until_unix_ns"] = initializing["heartbeat_unix_ns"]
        _install_initial_lease(client, initializing)
        lease_installed = True

        for entry in before:
            if entry["name"] not in SUSPEND_CONTROLLERS:
                continue
            replacement = copy.deepcopy(entry["object"])
            replacement["spec"]["suspend"] = True
            client.replace(replacement)

        held_records = [
            _controller_record(client.get("cronjob", name))
            for name in ALL_CONTROLLERS
        ]
        by_name = {str(entry["name"]): entry for entry in held_records}
        for prior in before:
            current = by_name[str(prior["name"])]
            if current["uid"] != prior["uid"]:
                raise FreezeError(
                    f"CronJob UID changed during hold: {prior['name']}"
                )
            if prior["name"] in SUSPEND_CONTROLLERS:
                expected_spec = copy.deepcopy(prior["object"]["spec"])
                expected_spec["suspend"] = True
                if current["resource_version"] == prior["resource_version"]:
                    raise FreezeError(
                        f"CronJob resourceVersion did not advance: {prior['name']}"
                    )
                if current["object"].get("spec") != expected_spec:
                    raise FreezeError(
                        f"CronJob changed beyond exact suspension: {prior['name']}"
                    )
            elif current["object"].get("spec") != prior["object"].get("spec"):
                # Observed controllers are part of the freeze boundary too.
                # They are not mutated by this tool, so any spec difference
                # between the initial capture and the held re-read is drift.
                raise FreezeError(
                    f"CronJob changed beyond exact suspension: {prior['name']}"
                )
        held_sha = write_once_json(
            run_dir / "controllers-held.json",
            {"schema": FREEZE_SCHEMA, "run_id": run_id, "controllers": held_records},
        )
        held = {str(entry["name"]): entry["object"] for entry in held_records}
        items = client.inventory()
        violations = find_rw_overlap_violations(
            items,
            phase="idle",
            held_controllers=held,
        )
        snapshot = _snapshot(
            run_id=run_id,
            phase="idle",
            sequence=0,
            controllers=[entry["object"] for entry in held_records],
            items=items,
            violations=violations,
        )
        snapshot_sha = write_once_json(
            run_dir / "snapshots/00000000.json",
            snapshot,
        )
        if violations:
            raise FreezeError(
                f"hold found {len(violations)} overlapping workload(s); "
                "see restricted snapshot"
            )
        replace_json(
            run_dir / "phase.json",
            _new_phase_state(run_id=run_id, phase="idle"),
        )
        lease = _lease_record(
            run_id=run_id,
            phase="idle",
            sequence=0,
            snapshot_sha256=snapshot_sha,
            controller_snapshot_sha256=held_sha,
        )
        _replace_lease(client, lease)
        write_once_json(
            run_dir / "hold-complete.json",
            {
                "schema": FREEZE_SCHEMA,
                "run_id": run_id,
                "controllers_before_sha256": before_sha,
                "controllers_held_sha256": held_sha,
                "initial_snapshot_sha256": snapshot_sha,
            },
        )
    except BaseException as exc:
        if lease_installed:
            failed = _lease_record(
                run_id=run_id,
                phase="idle",
                sequence=0,
                snapshot_sha256="0" * 64,
                controller_snapshot_sha256=before_sha,
                status="failed",
            )
            failed["valid_until_unix_ns"] = failed["heartbeat_unix_ns"]
            try:
                _replace_lease(client, failed)
            except Exception:
                pass
            try:
                write_once_json(
                    run_dir / f"hold-failed-{time.time_ns()}.json",
                    {
                        "schema": FREEZE_SCHEMA,
                        "run_id": run_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            except Exception:
                pass
        raise
    finally:
        hold_ownership.close()
    print(f"freeze held: run={run_id} state={run_dir}")
    return run_dir


def _watch_owned(
    client: Kubectl,
    *,
    run_dir: Path,
    interval_seconds: float,
    once: bool = False,
) -> int:
    _verify_hold_record_hashes(run_dir)
    held = _held_map(run_dir)
    run_id = run_dir.name
    controller_sha = hashlib.sha256((run_dir / "controllers-held.json").read_bytes()).hexdigest()
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    existing_sequences = [
        int(path.stem)
        for path in (run_dir / "snapshots").glob("[0-9]*.json")
        if path.stem.isdigit()
    ]
    sequence = max(existing_sequences, default=0)
    write_once_json(
        run_dir / "watchdogs" / f"{time.time_ns()}-{os.getpid()}.json",
        {"schema": FREEZE_SCHEMA, "run_id": run_id, "pid": os.getpid()},
    )
    try:
        while not stop_requested:
            stop_path = run_dir / "stop-requested.json"
            restore_stop_requested = stop_path.exists()
            if restore_stop_requested and _read_json(stop_path) != {
                "schema": FREEZE_SCHEMA,
                "run_id": run_id,
            }:
                raise FreezeError("watchdog stop request is malformed")
            phase_state = _read_json(run_dir / "phase.json")
            phase = _validate_phase_state(
                phase_state,
                run_id=run_id,
                allow_expired=True,
            )
            phase_expired = _phase_state_expired(
                phase_state,
                run_id=run_id,
            )
            # Once the bounded authorization expires, its formerly allowed
            # Job is no longer exempt. A clean scan must therefore use idle
            # policy before this watchdog can publish terminal stop evidence.
            scan_phase = (
                "idle" if restore_stop_requested or phase_expired else phase
            )
            phase_request_id = str(phase_state["request_id"])
            controllers = _validate_held_controllers(client, held)
            items = client.inventory()
            violations = find_rw_overlap_violations(
                items,
                phase=scan_phase,
                held_controllers=held,
            )
            sequence += 1
            snapshot = _snapshot(
                run_id=run_id,
                phase=scan_phase,
                sequence=sequence,
                controllers=controllers,
                items=items,
                violations=violations,
            )
            snapshot_path = run_dir / "snapshots" / f"{sequence:08d}.json"
            snapshot_sha = write_once_json(snapshot_path, snapshot)
            if violations:
                raise FreezeError(
                    f"watchdog found {len(violations)} overlapping workload(s)"
                )
            if restore_stop_requested:
                # Restore already owns the long phase lock. This fresh scan
                # deliberately uses idle policy, then the short publication
                # lock binds its exact phase, closed lease, and terminal marker
                # so no ordinary gate can masquerade as restore authority.
                with _phase_publication_owner(run_dir):
                    terminal_phase = _read_json(run_dir / "phase.json")
                    if terminal_phase != phase_state:
                        continue
                    if (
                        _validate_phase_state(terminal_phase, run_id=run_id)
                        != "idle"
                    ):
                        raise FreezeError(
                            "restore stop requires an exact idle phase"
                        )
                    closed = _lease_record(
                        run_id=run_id,
                        phase="idle",
                        sequence=sequence,
                        snapshot_sha256=snapshot_sha,
                        controller_snapshot_sha256=controller_sha,
                        status="closed",
                    )
                    closed["valid_until_unix_ns"] = closed[
                        "heartbeat_unix_ns"
                    ]
                    _replace_lease(client, closed)
                    write_once_json(
                        run_dir / "watchdog-stopped.json",
                        {
                            "schema": FREEZE_SCHEMA,
                            "run_id": run_id,
                            "sequence": sequence,
                            "reason": "restore-requested-clean-idle-scan",
                            "phase_request_id": phase_request_id,
                            "snapshot_sha256": snapshot_sha,
                        },
                    )
                return 0
            if phase_expired:
                try:
                    # Terminal expiry must exclude both a gate and restore.
                    # The short publication lock then makes its final phase
                    # equality check, failed lease, and stopped record one
                    # local commit with respect to heartbeat publication.
                    with _phase_owner(run_dir):
                        with _phase_publication_owner(run_dir):
                            terminal_phase = _read_json(run_dir / "phase.json")
                            if terminal_phase != phase_state or not (
                                _phase_state_expired(
                                    terminal_phase,
                                    run_id=run_id,
                                )
                            ):
                                continue
                            failed = _lease_record(
                                run_id=run_id,
                                phase=phase,
                                sequence=sequence,
                                snapshot_sha256=snapshot_sha,
                                controller_snapshot_sha256=controller_sha,
                                status="failed",
                            )
                            failed["valid_until_unix_ns"] = failed[
                                "heartbeat_unix_ns"
                            ]
                            _replace_lease(client, failed)
                            write_once_json(
                                run_dir / "watchdog-stopped.json",
                                {
                                    "schema": FREEZE_SCHEMA,
                                    "run_id": run_id,
                                    "sequence": sequence,
                                    "reason": "phase-expired-clean-idle-scan",
                                    "expired_phase": phase,
                                    "phase_request_id": phase_request_id,
                                    "phase_valid_until_unix_ns": phase_state[
                                        "valid_until_unix_ns"
                                    ],
                                    "snapshot_sha256": snapshot_sha,
                                },
                            )
                except CoordinationBusy:
                    # A gate/restore owns phase state. It will either publish
                    # a new request or stop this watchdog; rescan instead of
                    # manufacturing a generic terminal failure.
                    continue
                raise FreezeError("watchdog phase request expired")
            with _phase_publication_owner(run_dir):
                current_phase_state = _read_json(run_dir / "phase.json")
                if current_phase_state != phase_state:
                    # A gate changed phase while this potentially slow scan
                    # was in flight. Keep the restricted snapshot, but never
                    # issue a lease for its stale request.
                    continue
                if _phase_state_expired(current_phase_state, run_id=run_id):
                    # Expiry crossed during a scan that used the old phase's
                    # Job allowance. Fail immediately, then rescan idle.
                    failed = _lease_record(
                        run_id=run_id,
                        phase=phase,
                        sequence=sequence,
                        snapshot_sha256="0" * 64,
                        controller_snapshot_sha256=controller_sha,
                        status="failed",
                    )
                    failed["valid_until_unix_ns"] = failed[
                        "heartbeat_unix_ns"
                    ]
                    _replace_lease(client, failed)
                    continue
                if once:
                    # Smoke scans never publish a held heartbeat that a
                    # concurrent gate could mistake for sustained liveness.
                    closed = _lease_record(
                        run_id=run_id,
                        phase=phase,
                        sequence=sequence,
                        snapshot_sha256=snapshot_sha,
                        controller_snapshot_sha256=controller_sha,
                        status="closed",
                    )
                    closed["valid_until_unix_ns"] = closed[
                        "heartbeat_unix_ns"
                    ]
                    _replace_lease(client, closed)
                    write_once_json(
                        run_dir / "watchdog-interrupted" / f"{time.time_ns()}.json",
                        {
                            "schema": FREEZE_SCHEMA,
                            "run_id": run_id,
                            "sequence": sequence,
                            "reason": "once",
                        },
                    )
                    return 0
                lease = _lease_record(
                    run_id=run_id,
                    phase=phase,
                    sequence=sequence,
                    snapshot_sha256=snapshot_sha,
                    controller_snapshot_sha256=controller_sha,
                )
                _replace_lease(client, lease)
                replace_json(
                    run_dir / "heartbeat.json",
                    {
                        **lease,
                        "phase_request_id": phase_request_id,
                        "watchdog_pid": os.getpid(),
                    },
                )
            time.sleep(interval_seconds)
    except BaseException as exc:
        if not (run_dir / "watchdog-stopped.json").exists():
            failed = _lease_record(
                run_id=run_id,
                phase="idle",
                sequence=sequence,
                snapshot_sha256="0" * 64,
                controller_snapshot_sha256=controller_sha,
                status="failed",
            )
            failed["valid_until_unix_ns"] = failed["heartbeat_unix_ns"]
            try:
                _replace_lease(client, failed)
            except Exception:
                pass
        write_once_json(
            run_dir / f"watchdog-failed-{time.time_ns()}.json",
            {
                "schema": FREEZE_SCHEMA,
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    # --once and unilateral signals are smoke/interruption paths, never
    # restoration authority. Close the live lease under the publication lock
    # but deliberately omit watchdog-stopped.json so restore requires a live
    # watchdog and a new clean idle scan.
    with _phase_publication_owner(run_dir):
        exit_phase_state = _read_json(run_dir / "phase.json")
        exit_phase = _validate_phase_state(
            exit_phase_state,
            run_id=run_id,
            allow_expired=True,
        )
        closed = _lease_record(
            run_id=run_id,
            phase=exit_phase,
            sequence=sequence,
            snapshot_sha256="0" * 64,
            controller_snapshot_sha256=controller_sha,
            status="closed",
        )
        closed["valid_until_unix_ns"] = closed["heartbeat_unix_ns"]
        _replace_lease(client, closed)
    write_once_json(
        run_dir / "watchdog-interrupted" / f"{time.time_ns()}.json",
        {
            "schema": FREEZE_SCHEMA,
            "run_id": run_id,
            "sequence": sequence,
            "reason": "signal",
        },
    )
    return 0


def watch(
    client: Kubectl,
    *,
    run_dir: Path,
    interval_seconds: float,
    once: bool = False,
) -> int:
    """Run exactly one watchdog and refuse after restoration has begun."""
    with _watchdog_owner(run_dir):
        if (run_dir / "restore-in-progress.json").exists():
            raise FreezeError("restore has begun; watchdog cannot restart")
        if (run_dir / "watchdog-stopped.json").exists():
            raise FreezeError("watchdog was already stopped for this run")
        return _watch_owned(
            client,
            run_dir=run_dir,
            interval_seconds=interval_seconds,
            once=once,
        )


def _gate_phase_owned(
    client: Kubectl,
    *,
    run_dir: Path,
    phase: str,
    timeout_seconds: float,
) -> None:
    if phase not in PHASE_JOB:
        raise FreezeError(f"unsupported phase: {phase}")
    heartbeat_path = run_dir / "heartbeat.json"
    before = _read_json(heartbeat_path) if heartbeat_path.exists() else {}
    previous_phase = _read_json(run_dir / "phase.json")
    _validate_phase_state(previous_phase, run_id=run_dir.name)
    requested_phase = _new_phase_state(run_id=run_dir.name, phase=phase)
    with _phase_publication_owner(run_dir):
        if _read_json(run_dir / "phase.json") != previous_phase:
            raise FreezeError("watchdog phase changed before gate publication")
        replace_json(run_dir / "phase.json", requested_phase)
    deadline = time.monotonic() + timeout_seconds
    authorized = False
    try:
        while time.monotonic() < deadline:
            if heartbeat_path.exists():
                heartbeat = _read_json(heartbeat_path)
                if (
                    heartbeat.get("run_id") == run_dir.name
                    and heartbeat.get("phase") == phase
                    and heartbeat.get("phase_request_id")
                    == requested_phase["request_id"]
                    and heartbeat.get("status") == "held"
                    and int(heartbeat.get("sequence", -1))
                    > int(before.get("sequence", -1))
                    and int(heartbeat.get("valid_until_unix_ns", 0))
                    > time.time_ns()
                ):
                    live = client.get("configmap", LEASE_CONFIGMAP)
                    try:
                        lease = json.loads(live["data"]["lease.json"])
                    except (KeyError, TypeError, json.JSONDecodeError) as exc:
                        raise FreezeError("live watchdog lease is malformed") from exc
                    heartbeat_lease = {
                        key: value
                        for key, value in heartbeat.items()
                        if key not in {"phase_request_id", "watchdog_pid"}
                    }
                    if (
                        set(lease) != LEASE_FIELDS
                        or set(heartbeat_lease) != LEASE_FIELDS
                        or lease != heartbeat_lease
                    ):
                        raise FreezeError("local and live watchdog leases differ")
                    authorization = {
                        "schema": FREEZE_SCHEMA,
                        "run_id": run_dir.name,
                        "phase": phase,
                        "phase_request_id": requested_phase["request_id"],
                        "sequence": heartbeat["sequence"],
                        "snapshot_sha256": heartbeat["snapshot_sha256"],
                        "valid_until_unix_ns": heartbeat["valid_until_unix_ns"],
                    }
                    write_once_json(
                        run_dir
                        / "authorizations"
                        / f"{heartbeat['sequence']:08d}-{phase}.json",
                        authorization,
                    )
                    authorized = True
                    print(
                        f"phase authorized while watchdog is live: {phase} "
                        f"sequence={heartbeat['sequence']}"
                    )
                    return
            time.sleep(0.25)
        raise FreezeError("watchdog did not publish a fresh clean phase lease")
    finally:
        if not authorized:
            # Never leave a timed-out request armed for a later watchdog
            # restart. If a late heartbeat already exposed that phase, expire
            # it as well; the next clean scan may renew only the prior phase.
            with _phase_publication_owner(run_dir):
                current_phase = _read_json(run_dir / "phase.json")
                if current_phase == requested_phase:
                    replace_json(run_dir / "phase.json", previous_phase)
                elif current_phase != previous_phase:
                    raise FreezeError("watchdog phase drifted during gate rollback")
                try:
                    live_lease = _live_lease(client, run_id=run_dir.name)
                except FreezeError:
                    # Missing or malformed is already unusable by data Jobs.
                    live_lease = None
                if live_lease is not None and (
                    live_lease.get("phase") == phase
                    and live_lease.get("status") == "held"
                ):
                    failed = dict(live_lease)
                    failed["status"] = "failed"
                    failed["valid_until_unix_ns"] = failed[
                        "heartbeat_unix_ns"
                    ]
                    # Do not swallow a CAS failure: a valid held lease for a
                    # denied request must either be expired here or surface as
                    # a terminal operator error until its short TTL elapses.
                    _replace_lease(client, failed)


def gate_phase(
    client: Kubectl,
    *,
    run_dir: Path,
    phase: str,
    timeout_seconds: float,
) -> None:
    """Serialize and authorize one phase while the watchdog stays live."""
    with _phase_owner(run_dir):
        if (
            (run_dir / "restore-in-progress.json").exists()
            or (run_dir / "stop-requested.json").exists()
            or (run_dir / "watchdog-stopped.json").exists()
        ):
            raise FreezeError("freeze shutdown has begun; phase gate refused")
        _gate_phase_owned(
            client,
            run_dir=run_dir,
            phase=phase,
            timeout_seconds=timeout_seconds,
        )


def _live_lease(client: Kubectl, *, run_id: str) -> dict[str, Any]:
    live = client.get("configmap", LEASE_CONFIGMAP)
    try:
        lease = json.loads(live["data"]["lease.json"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FreezeError("live watchdog lease is malformed") from exc
    if not isinstance(lease, dict) or set(lease) != LEASE_FIELDS or (
        lease.get("schema") != LEASE_SCHEMA
    ):
        raise FreezeError("live watchdog lease schema is malformed")
    if lease.get("run_id") != run_id:
        raise FreezeError("live watchdog lease belongs to another run")
    return lease


def _expected_held_spec(
    before: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    source = before.get("object")
    if not isinstance(source, Mapping) or not isinstance(source.get("spec"), Mapping):
        raise FreezeError(f"controller-before record is malformed: {name}")
    expected = copy.deepcopy(dict(source["spec"]))
    if name in SUSPEND_CONTROLLERS:
        expected["suspend"] = True
    return expected


def _restore_owned(
    client: Kubectl,
    *,
    run_dir: Path,
    timeout_seconds: float,
) -> None:
    before_state = _read_json(run_dir / "controllers-before.json")
    before_entries = before_state.get("controllers")
    if not isinstance(before_entries, list):
        raise FreezeError("controller-before record is malformed")
    before = {
        str(entry["name"]): entry
        for entry in before_entries
        if isinstance(entry, Mapping) and isinstance(entry.get("name"), str)
    }
    if set(before) != set(ALL_CONTROLLERS):
        raise FreezeError("controller-before set is incomplete")

    completed_hold = (run_dir / "hold-complete.json").exists()
    if completed_hold:
        _verify_hold_record_hashes(run_dir)
    stopped_path = run_dir / "watchdog-stopped.json"
    if completed_hold and not stopped_path.exists():
        _gate_phase_owned(
            client,
            run_dir=run_dir,
            phase="idle",
            timeout_seconds=timeout_seconds,
        )
        stop_path = run_dir / "stop-requested.json"
        if not stop_path.exists():
            write_once_json(
                stop_path,
                {"schema": FREEZE_SCHEMA, "run_id": run_dir.name},
            )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline and not stopped_path.exists():
            time.sleep(0.25)
        if not stopped_path.exists():
            raise FreezeError("watchdog did not fail-close its lease before restore")

    lease = _live_lease(client, run_id=run_dir.name)
    if completed_hold and lease.get("status") == "failed":
        # A bounded phase may end without a graceful operator stop. Restore is
        # safe only when the sole watchdog recorded a fresh zero-overlap scan
        # under idle policy and bound that exact snapshot into the failed
        # live lease after the phase's full Job deadline elapsed.
        _verify_expired_phase_stop(run_dir, lease)
    elif completed_hold and lease.get("status") == "closed":
        _verify_restore_requested_stop(run_dir, lease)
    elif completed_hold and lease.get("status") != "released":
        raise FreezeError("watchdog lease is not closed before restore")
    if not completed_hold and lease.get("status") not in {
        "initializing",
        "failed",
        "closed",
        "held",
        "released",
    }:
        raise FreezeError("incomplete hold has an unsafe live lease state")
    if not completed_hold:
        controller_hashes = {
            _sha256_path(run_dir / "controllers-before.json")
        }
        incomplete_held_path = run_dir / "controllers-held.json"
        if incomplete_held_path.exists():
            controller_hashes.add(_sha256_path(incomplete_held_path))
        if lease.get("controller_snapshot_sha256") not in controller_hashes:
            raise FreezeError("incomplete hold controller-record hash mismatch")

    # The sole watchdog has stopped (or an incomplete hold never started it).
    # Acquire the same OS lock and publish a permanent restore marker while
    # holding it, closing the stopped-marker/new-watchdog race. Retries reuse
    # this marker; future watchdog invocations fail before any lease update.
    _begin_restore(run_dir)

    held: dict[str, Mapping[str, Any]] = {}
    if (run_dir / "controllers-held.json").exists():
        held = _held_map(run_dir)
        for name in ALL_CONTROLLERS:
            if _metadata(held[name]).get("uid") != before[name].get("uid"):
                raise FreezeError(f"held-controller UID differs from before: {name}")
            if held[name].get("spec") != _expected_held_spec(before[name], name=name):
                raise FreezeError(f"held-controller spec differs from exact hold: {name}")

    # Full preflight accepts only the exact held or exact prior state. This
    # makes restoration resumable after a crash or a later CAS failure: an
    # already restored controller is a no-op, never an invitation to overwrite
    # arbitrary drift. Each replace still carries the just-read resourceVersion.
    current: dict[str, dict[str, Any]] = {}
    for name in ALL_CONTROLLERS:
        live = client.get("cronjob", name)
        if _metadata(live).get("uid") != before[name].get("uid"):
            raise FreezeError(f"CronJob UID changed before restore: {name}")
        prior_spec = before[name]["object"].get("spec")
        expected_held = _expected_held_spec(before[name], name=name)
        if live.get("spec") != prior_spec and live.get("spec") != expected_held:
            raise FreezeError(f"CronJob spec drifted before restore: {name}")
        current[name] = live

    restored: list[dict[str, object]] = []
    for name in SUSPEND_CONTROLLERS:
        prior_spec = before[name]["object"].get("spec")
        if current[name].get("spec") == prior_spec:
            result = current[name]
        else:
            replacement = copy.deepcopy(current[name])
            replacement["spec"] = copy.deepcopy(prior_spec)
            result = client.replace(replacement)
        restored_spec = result.get("spec", {})
        suspend_matches = (
            (
                before[name]["prior_suspend_present"]
                and restored_spec.get("suspend") == before[name]["prior_suspend"]
            )
            or (
                not before[name]["prior_suspend_present"]
                and "suspend" not in restored_spec
            )
        )
        if (
            _metadata(result).get("uid") != before[name]["uid"]
            or not suspend_matches
            or result.get("spec") != before[name]["object"].get("spec")
        ):
            raise FreezeError(f"CronJob exact suspend restoration failed: {name}")
        restored.append(_controller_record(result))

    # Re-read the entire controller cohort after the last CAS. This catches an
    # already-prior controller drifting after preflight and an earlier restored
    # controller drifting while a later one is restored. These final returned
    # objects, not cached preflight objects, become restoration evidence.
    restored = []
    for name in ALL_CONTROLLERS:
        final = client.get("cronjob", name)
        if (
            _metadata(final).get("uid") != before[name]["uid"]
            or final.get("spec") != before[name]["object"].get("spec")
        ):
            raise FreezeError(f"CronJob drifted after exact restore: {name}")
        restored.append(_controller_record(final))
    restored_record = {
        "schema": FREEZE_SCHEMA,
        "run_id": run_dir.name,
        "controllers": restored,
    }
    restored_path = run_dir / "controllers-restored.json"
    if restored_path.exists():
        existing = _read_json(restored_path)
        existing_entries = existing.get("controllers")
        if (
            existing.get("schema") != FREEZE_SCHEMA
            or existing.get("run_id") != run_dir.name
            or not isinstance(existing_entries, list)
        ):
            raise FreezeError("restored-controller record is malformed")
        existing_by_name = {
            str(entry.get("name")): entry
            for entry in existing_entries
            if isinstance(entry, Mapping)
        }
        if set(existing_by_name) != set(ALL_CONTROLLERS):
            raise FreezeError("restored-controller record is incomplete")
        for name in ALL_CONTROLLERS:
            entry = existing_by_name[name]
            recorded_object = entry.get("object")
            if (
                entry.get("uid") != before[name].get("uid")
                or not isinstance(recorded_object, Mapping)
                or recorded_object.get("spec")
                != before[name]["object"].get("spec")
            ):
                raise FreezeError(
                    f"restored-controller evidence differs from prior state: {name}"
                )
    else:
        write_once_json(restored_path, restored_record)
    controller_record_path = run_dir / "controllers-held.json"
    if not controller_record_path.exists():
        controller_record_path = run_dir / "controllers-before.json"
    released = _lease_record(
        run_id=run_dir.name,
        phase="idle",
        sequence=0,
        snapshot_sha256="0" * 64,
        controller_snapshot_sha256=hashlib.sha256(
            controller_record_path.read_bytes()
        ).hexdigest(),
        status="released",
    )
    released["valid_until_unix_ns"] = released["heartbeat_unix_ns"]
    if lease.get("status") != "released":
        _replace_lease(client, released)
    print(f"freeze restored exactly: run={run_dir.name}")


def restore(client: Kubectl, *, run_dir: Path, timeout_seconds: float) -> None:
    """Serialize exact restore against every phase gate and other restore."""
    with _hold_owner(run_dir):
        with _phase_owner(run_dir):
            _restore_owned(
                client,
                run_dir=run_dir,
                timeout_seconds=timeout_seconds,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default=CONTEXT)
    parser.add_argument("--namespace", default=NAMESPACE)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hold_parser = subparsers.add_parser("hold")
    hold_parser.add_argument("--state-dir", type=Path, required=True)
    hold_parser.add_argument("--run-id", required=True)

    watch_parser = subparsers.add_parser("watch")
    watch_parser.add_argument("--run-dir", type=Path, required=True)
    watch_parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    watch_parser.add_argument("--once", action="store_true", help="one scan for tests/operator smoke only")

    gate_parser = subparsers.add_parser("gate")
    gate_parser.add_argument("--run-dir", type=Path, required=True)
    gate_parser.add_argument("--phase", choices=tuple(PHASE_JOB), required=True)
    gate_parser.add_argument("--timeout-seconds", type=float, default=60.0)

    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("--run-dir", type=Path, required=True)
    restore_parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = Kubectl(context=args.context, namespace=args.namespace)
    try:
        if args.command == "hold":
            hold(client, state_dir=args.state_dir, run_id=args.run_id)
        elif args.command == "watch":
            watch(
                client,
                run_dir=args.run_dir,
                interval_seconds=args.interval_seconds,
                once=args.once,
            )
        elif args.command == "gate":
            gate_phase(
                client,
                run_dir=args.run_dir,
                phase=args.phase,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            restore(client, run_dir=args.run_dir, timeout_seconds=args.timeout_seconds)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        print(f"crop source freeze refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
