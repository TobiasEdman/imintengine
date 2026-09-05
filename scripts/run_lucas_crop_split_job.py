#!/usr/bin/env python3
"""Build and verify the fixed LUCAS crop split from the baked protocol.

This entrypoint intentionally accepts no behavioural arguments.  Kubernetes
supplies only the source/image identity and Pod UID through environment
variables; all paths and protocol values are part of the runtime-image SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from atomic_npz import exclusive_dataset_lock
from crop_distill_protocol import (
    BASE_PYTHON,
    DATA_DIR,
    DISTILL_DIR,
    LUCAS_SOURCE_INDEX,
    PROVENANCE_SCRIPT,
    PVC_ROOT,
    RUNTIME_MANIFEST,
    SCORING_PYTHON,
    SOURCE_ROOT,
    SOURCE_ACCESS_COMPLETION_INPUT,
    SOURCE_ACCESS_INDEX_SHA256,
    SOURCE_ACCESS_INDEX_SIZE,
    SOURCE_ACCESS_LOCK_FILE,
    SOURCE_ACCESS_LOCK_MODE,
    STORAGE_GID,
    STORAGE_UID,
    SPLIT_RECORD_DIR,
    SPLIT_SCRIPT,
    WORK_ROOT,
    RuntimeIdentity,
    require_process_identity,
    require_source_access_run_id,
    require_source_access_sha256,
    runtime_claims,
    runtime_identity,
)
from crop_distill_provenance import verify_runtime
from crop_source_access import (
    FREEZE_LEASE_PATH,
    require_fresh_freeze_lease,
    runtime_binding,
    verify_completion as verify_source_access_completion,
    verify_live_completion_cohort,
)

DISTILL_INDEX = DISTILL_DIR / "lucas_crop_distill_index.parquet"
VALIDATOR_INDEX = DISTILL_DIR / "lucas_crop_validator_holdout_index.parquet"
SPLIT = DISTILL_DIR / "lucas_crop_split.json"
SPLIT_MANIFEST = DISTILL_DIR / "lucas_crop_split.MANIFEST.json"
_MAX_DIAGNOSTIC_LENGTH = 1024


class JobArgumentError(ValueError):
    """Raised instead of exiting before failure provenance can be written."""


class JobArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise JobArgumentError(message)


def _bounded_claim(value: str, label: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        return f"<empty {label}>"
    return normalized[:_MAX_DIAGNOSTIC_LENGTH]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LucasCropSplitJob:
    """One fixed split freeze with fail-closed terminal provenance."""

    job_name = "ladder-lucas-crop-split"

    def __init__(self, identity: RuntimeIdentity) -> None:
        self.identity = identity
        self.failure_stage = "bootstrap"
        self.source_access_plan_sha256 = "<missing>"
        self.source_access_plan_pod_uid = "<missing>"
        self.source_access_completion_sha256 = "<missing>"
        self.source_access_completion_pod_uid = "<missing>"
        self.freeze_lease_path = FREEZE_LEASE_PATH

    def _run(self, stage: str, command: Sequence[object]) -> None:
        self.failure_stage = stage
        argv = [str(value) for value in command]
        print(f"[{stage}] {shlex.join(argv)}", flush=True)
        subprocess.run(argv, cwd=SOURCE_ROOT, check=True)

    def _runtime_args(self, *, diagnostic: bool = False) -> list[str]:
        if diagnostic:
            source_git_sha = _bounded_claim(
                self.identity.source_git_sha, "source-git-sha"
            )
            image_ref = _bounded_claim(self.identity.image_ref, "image-ref")
            return [
                f"--source-git-sha={source_git_sha}",
                f"--image-ref={image_ref}",
                f"--runtime-manifest={RUNTIME_MANIFEST}",
            ]
        return [
            "--source-git-sha",
            self.identity.source_git_sha,
            "--image-ref",
            self.identity.image_ref,
            "--runtime-manifest",
            str(RUNTIME_MANIFEST),
        ]

    def _provenance_base(self, *, status: str, exit_code: int) -> list[str]:
        diagnostic = status == "failed"
        command = [
            str(BASE_PYTHON),
            str(PROVENANCE_SCRIPT),
            "finalize",
            "--kind",
            "split",
            "--record-dir",
            str(SPLIT_RECORD_DIR),
            "--run-id",
            self.identity.pod_uid,
            "--job",
            self.job_name,
            "--pod-uid",
            self.identity.pod_uid,
            "--status",
            status,
            "--exit-code",
            str(exit_code),
            *self._runtime_args(diagnostic=diagnostic),
        ]
        if status == "completed":
            command.extend(
                [
                    "--source-access-plan-sha256",
                    self.source_access_plan_sha256,
                    "--source-access-plan-pod-uid",
                    self.source_access_plan_pod_uid,
                    "--source-access-completion-sha256",
                    self.source_access_completion_sha256,
                    "--source-access-completion-pod-uid",
                    self.source_access_completion_pod_uid,
                ]
            )
        return command

    def _prepare_directories(self) -> None:
        self.failure_stage = "prepare-output"
        required = (
            DATA_DIR,
            LUCAS_SOURCE_INDEX.parent,
            DISTILL_DIR,
            SPLIT_RECORD_DIR,
        )
        for path in required:
            _ensure_real_directory(path, create=False)
        _prepare_private_work()

    def execute(self) -> None:
        self._prepare_directories()
        self.failure_stage = "acquire-source-access-lock"
        with exclusive_dataset_lock(
            SOURCE_ACCESS_LOCK_FILE,
            create=False,
            expected_uid=0,
            expected_gid=STORAGE_GID,
            expected_mode=SOURCE_ACCESS_LOCK_MODE,
        ):
            self._execute_locked()

    def _execute_locked(self) -> None:
        """Hold the cooperating-producer exclusion lock through publication."""
        self._run(
            "verify-runtime",
            [
                BASE_PYTHON,
                PROVENANCE_SCRIPT,
                "verify-runtime",
                *self._runtime_args(),
            ],
        )
        verified_runtime = verify_runtime(
            RUNTIME_MANIFEST,
            source_git_sha=self.identity.source_git_sha,
            image_ref=self.identity.image_ref,
        )
        self.failure_stage = "require-freeze-lease-before-source-read"
        require_fresh_freeze_lease(
            self.freeze_lease_path,
            expected_phase="split",
        )
        self.failure_stage = "verify-source-access-completion"
        completion = verify_source_access_completion(
            SOURCE_ACCESS_COMPLETION_INPUT,
            expected_sha256=self.source_access_completion_sha256,
            expected_source_git_sha=self.identity.source_git_sha,
            expected_image_ref=self.identity.image_ref,
            expected_completion_pod_uid=self.source_access_completion_pod_uid,
            expected_plan_sha256=self.source_access_plan_sha256,
            expected_runtime_binding=runtime_binding(
                self.identity, verified_runtime
            ),
        )
        plan = completion["plan"]
        if plan != {
            "pod_uid": self.source_access_plan_pod_uid,
            "sha256": self.source_access_plan_sha256,
        }:
            raise RuntimeError(
                "verified source-access completion differs from the Git-pinned "
                "PLAN SHA256/Pod UID authority"
            )
        print(
            "[verify-source-access-completion] "
            f"files={completion['summary']['files']} "
            f"sha256={self.source_access_completion_sha256}",
            flush=True,
        )
        self.failure_stage = "verify-live-source-cohort-before-freeze"
        verify_live_completion_cohort(
            completion,
            data_dir=DATA_DIR,
            cross_pod=True,
        )
        self.failure_stage = "refresh-freeze-lease-before-freeze"
        require_fresh_freeze_lease(
            self.freeze_lease_path,
            expected_phase="split",
        )
        self._run(
            "freeze-split",
            [
                SCORING_PYTHON,
                SPLIT_SCRIPT,
                "--lucas-index",
                LUCAS_SOURCE_INDEX,
                "--data-dir",
                DATA_DIR,
                "--out-dir",
                DISTILL_DIR,
                "--git-sha",
                self.identity.source_git_sha,
                "--expected-source-index-sha256",
                SOURCE_ACCESS_INDEX_SHA256,
                "--expected-source-index-size",
                SOURCE_ACCESS_INDEX_SIZE,
            ],
        )
        self.failure_stage = "refresh-freeze-lease-after-freeze"
        require_fresh_freeze_lease(
            self.freeze_lease_path,
            expected_phase="split",
        )
        self._run(
            "verify-split",
            [
                SCORING_PYTHON,
                SPLIT_SCRIPT,
                "--verify",
                "--out-dir",
                DISTILL_DIR,
                "--expected-git-sha",
                self.identity.source_git_sha,
            ],
        )
        consumer_dir = DISTILL_DIR / "crop_consumer"
        _ensure_real_directory(consumer_dir, create=False)
        _set_directory_mode(consumer_dir, 0o550)
        _set_directory_mode(DISTILL_DIR, 0o550)
        self.failure_stage = "verify-live-source-cohort-before-completion"
        verify_live_completion_cohort(
            completion,
            data_dir=DATA_DIR,
            cross_pod=True,
        )
        self.failure_stage = "require-freeze-lease-before-completion"
        require_fresh_freeze_lease(
            self.freeze_lease_path,
            expected_phase="split",
        )
        self.failure_stage = "bind-split-manifest"
        manifest_sha = sha256_file(SPLIT_MANIFEST)
        self._run(
            "publish-completion",
            [
                *self._provenance_base(status="completed", exit_code=0),
                "--split-manifest",
                SPLIT_MANIFEST,
                "--split-sha256",
                manifest_sha,
                "--artifact",
                f"index={DISTILL_INDEX}",
                "--artifact",
                f"validator_holdout={VALIDATOR_INDEX}",
                "--artifact",
                f"split={SPLIT}",
                "--artifact",
                f"manifest={SPLIT_MANIFEST}",
            ],
        )

    def publish_failure(self, exit_code: int) -> None:
        """Publish only diagnostic identity; never touch split artifacts."""
        _ensure_real_directory(SPLIT_RECORD_DIR, create=False)
        command = [
            *self._provenance_base(status="failed", exit_code=exit_code),
            "--failure-stage",
            self.failure_stage,
        ]
        argv = [str(value) for value in command]
        print(f"[publish-failure] {shlex.join(argv)}", file=sys.stderr, flush=True)
        subprocess.run(argv, cwd=SOURCE_ROOT, check=True)


def _failure_exit_code(exc: BaseException) -> int:
    if isinstance(exc, KeyboardInterrupt):
        return 130
    if isinstance(exc, JobArgumentError):
        return 2
    if isinstance(exc, subprocess.CalledProcessError):
        if exc.returncode < 0:
            return min(255, 128 + abs(exc.returncode))
        if exc.returncode > 0:
            return min(255, exc.returncode)
    return 1


def _ensure_real_directory(path: Path, *, create: bool) -> None:
    """Reject symlinks/non-directories in one required PVC subPath chain."""
    pvc_root = PVC_ROOT
    try:
        relative = path.relative_to(pvc_root)
    except ValueError as exc:
        raise RuntimeError(f"PVC directory escapes {pvc_root}: {path}") from exc

    try:
        root_stat = pvc_root.lstat()
    except OSError as exc:
        raise RuntimeError(f"PVC root is unavailable: {pvc_root}: {exc}") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError(f"PVC root is not a real directory: {pvc_root}")

    current = pvc_root
    for part in relative.parts:
        current = current / part
        try:
            identity = current.lstat()
        except FileNotFoundError as exc:
            if not create:
                raise RuntimeError(
                    f"required PVC directory is missing: {current}"
                ) from exc
            try:
                current.mkdir()
                identity = current.lstat()
            except OSError as exc:
                raise RuntimeError(
                    f"cannot create required PVC directory {current}: {exc}"
                ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"cannot inspect PVC directory {current}: {exc}"
            ) from exc
        if not stat.S_ISDIR(identity.st_mode):
            raise RuntimeError(
                f"PVC subPath component is not a real directory: {current}"
            )


def _set_directory_mode(path: Path, mode: int) -> None:
    """Change only the verified directory inode, never a symlink target."""
    flags = os.O_RDONLY | os.O_DIRECTORY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open PVC directory {path}: {exc}") from exc
    try:
        identity = os.fstat(fd)
        if not stat.S_ISDIR(identity.st_mode):
            raise RuntimeError(f"PVC path is not a directory: {path}")
        os.fchmod(fd, mode)
    except OSError as exc:
        raise RuntimeError(f"cannot set PVC directory mode {path}: {exc}") from exc
    finally:
        os.close(fd)


def _prepare_private_work() -> None:
    """Prepare writable temp/home paths on the Pod-private emptyDir."""
    try:
        identity = WORK_ROOT.lstat()
    except OSError as exc:
        raise RuntimeError(f"private work root is unavailable: {exc}") from exc
    if not stat.S_ISDIR(identity.st_mode):
        raise RuntimeError(f"private work root is not a directory: {WORK_ROOT}")
    for name in ("home", "tmp"):
        path = WORK_ROOT / name
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            child = path.lstat()
            if not stat.S_ISDIR(child.st_mode):
                raise RuntimeError(f"private work path is not a directory: {path}")
        except OSError as exc:
            raise RuntimeError(
                f"cannot create private work path {path}: {exc}"
            ) from exc


def build_parser() -> argparse.ArgumentParser:
    return JobArgumentParser(description=__doc__)


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    environment = os.environ if environ is None else environ
    claims = runtime_claims(environment)
    job = LucasCropSplitJob(claims)
    # This is the job boundary: every unexpected runtime failure must reach
    # immutable terminal provenance rather than escape unrecorded.
    try:
        job.failure_stage = "parse-arguments"
        build_parser().parse_args(argv)
        job.failure_stage = "validate-runtime-environment"
        job.identity = runtime_identity(environment)
        job.source_access_plan_sha256 = require_source_access_sha256(
            environment.get("CROP_SOURCE_ACCESS_PLAN_SHA256", ""),
            "CROP_SOURCE_ACCESS_PLAN_SHA256",
        )
        job.source_access_plan_pod_uid = require_source_access_run_id(
            environment.get("CROP_SOURCE_ACCESS_PLAN_POD_UID", ""),
            "CROP_SOURCE_ACCESS_PLAN_POD_UID",
        )
        job.source_access_completion_sha256 = require_source_access_sha256(
            environment.get("CROP_SOURCE_ACCESS_COMPLETION_SHA256", ""),
            "CROP_SOURCE_ACCESS_COMPLETION_SHA256",
        )
        job.source_access_completion_pod_uid = require_source_access_run_id(
            environment.get("CROP_SOURCE_ACCESS_COMPLETION_POD_UID", ""),
            "CROP_SOURCE_ACCESS_COMPLETION_POD_UID",
        )
        raw_freeze_lease_path = environment.get(
            "CROP_SOURCE_FREEZE_LEASE_PATH", ""
        )
        if raw_freeze_lease_path != str(FREEZE_LEASE_PATH):
            raise JobArgumentError(
                "CROP_SOURCE_FREEZE_LEASE_PATH must equal the baked lease path"
            )
        job.freeze_lease_path = Path(raw_freeze_lease_path)
        job.failure_stage = "validate-process-identity"
        require_process_identity(
            STORAGE_UID,
            expected_gid=STORAGE_GID,
            role="crop split producer",
        )
        job.execute()
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        exit_code = _failure_exit_code(exc)
        print(
            f"FATAL [{job.failure_stage}] {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        try:
            job.publish_failure(exit_code)
        except Exception as publish_exc:  # noqa: BLE001
            print(
                f"FATAL: failed work also lacks publishable provenance: {publish_exc}",
                file=sys.stderr,
                flush=True,
            )
            return 97
        return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
