#!/usr/bin/env python3
"""Run one image-baked LUCAS crop-distill protocol column.

The only behavioural argument is ``--model``. All commands, paths, model
parameters, checkpoint identities, and scoring settings are fixed in
``crop_distill_protocol.py`` and therefore in the runtime image's source SHA.
The reviewed Kubernetes manifest supplies the off-PVC trust anchor for the
frozen split; crop workers never infer that trust anchor from the mounted PVC.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import secrets
import shlex
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crop_distill_protocol import (
    BASE_PYTHON,
    CROP_HEADS_DIR,
    CROP_INDEX,
    CROP_RECORD_DIR,
    CROP_SPLIT,
    CROP_SPLIT_MANIFEST,
    DEVICE,
    EXTRACT_SCRIPT,
    MODEL_KEYS,
    MODEL_PYTHON,
    OOF_FOLDS,
    OOF_GROUP_COLUMN,
    OOF_HEADS,
    PROVENANCE_SCRIPT,
    RUNTIME_MANIFEST,
    SCORE_SCRIPT,
    SCORING_PYTHON,
    SOURCE_ROOT,
    STORAGE_GID,
    SPLIT_SCRIPT,
    TRUTH_COLUMN,
    WORK_ROOT,
    CropModelProtocol,
    RuntimeIdentity,
    model_protocol,
    model_process_uid,
    require_process_identity,
    require_split_manifest_sha256,
    runtime_claims,
    runtime_identity,
    split_manifest_claim,
)

_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
_DIR_FLAGS = _READ_FLAGS | os.O_DIRECTORY
_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
_SHA256_HEX = frozenset("0123456789abcdef")
_COPY_BLOCK_SIZE = 1024 * 1024
_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
_MAX_DIAGNOSTIC_LENGTH = 1024
_SPLIT_FILES = {
    "index": CROP_INDEX.name,
    "split": CROP_SPLIT.name,
    "manifest": CROP_SPLIT_MANIFEST.name,
}
_VALIDATOR_FILENAME = "lucas_crop_validator_holdout_index.parquet"


@dataclass(frozen=True, slots=True)
class CropSplitSnapshot:
    """Private, authenticated crop-consumer copy of the frozen split."""

    root: Path
    index: Path
    split: Path
    manifest: Path
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    """Identity computed from the private source while publishing it."""

    path: Path
    size_bytes: int
    sha256: str


class JobArgumentError(ValueError):
    """Raised instead of exiting before failure provenance can be written."""


class JobArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise JobArgumentError(message)


def _bounded_claim(
    value: str,
    label: str,
    *,
    max_length: int = _MAX_DIAGNOSTIC_LENGTH,
) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        return f"<empty {label}>"
    return normalized[:max_length]


def _path_parts(path: Path) -> tuple[str, ...]:
    if not path.is_absolute():
        raise RuntimeError(f"path must be absolute: {path}")
    parts = path.parts[1:]
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise RuntimeError(f"path is not a normalized absolute path: {path}")
    return parts


def _open_directory(path: Path, *, create: bool = False, mode: int = 0o700) -> int:
    """Open *path* without following any symlinked path component."""
    parts = _path_parts(path)
    current_fd = os.open("/", _DIR_FLAGS)
    try:
        for part in parts:
            try:
                next_fd = os.open(part, _DIR_FLAGS, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise RuntimeError(f"required directory is missing: {path}")
                try:
                    os.mkdir(part, mode=mode, dir_fd=current_fd)
                except FileExistsError:
                    pass
                try:
                    next_fd = os.open(part, _DIR_FLAGS, dir_fd=current_fd)
                except OSError as exc:
                    raise RuntimeError(
                        f"cannot securely open created directory {path}: {exc}"
                    ) from exc
            except OSError as exc:
                raise RuntimeError(
                    f"directory path contains a symlink or non-directory: {path}: {exc}"
                ) from exc
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _ensure_directory(path: Path, *, mode: int = 0o700) -> bool:
    """Safely ensure one child directory; return whether it was created."""
    parent_fd = _open_directory(path.parent)
    created = False
    try:
        try:
            os.mkdir(path.name, mode=mode, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        child_fd = os.open(path.name, _DIR_FLAGS, dir_fd=parent_fd)
        try:
            identity = os.fstat(child_fd)
            if identity.st_uid != os.geteuid():
                raise RuntimeError(
                    f"private directory is not owned by runtime UID: {path}"
                )
            if stat.S_IMODE(identity.st_mode) & 0o077:
                raise RuntimeError(
                    f"private directory grants group/other access: {path}"
                )
        finally:
            os.close(child_fd)
    except OSError as exc:
        raise RuntimeError(f"cannot securely create directory {path}: {exc}") from exc
    finally:
        os.close(parent_fd)
    return created


def _ensure_publish_directory(path: Path) -> None:
    """Create/reopen one model-owned, group-readable publication root."""
    parent_fd = _open_directory(path.parent)
    child_fd: int | None = None
    created = False
    try:
        try:
            os.mkdir(path.name, mode=0o750, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        child_fd = os.open(path.name, _DIR_FLAGS, dir_fd=parent_fd)
        identity = os.fstat(child_fd)
        if identity.st_uid != os.geteuid():
            raise RuntimeError(
                f"publication directory is not owned by runtime UID: {path}"
            )
        if created:
            os.fchmod(child_fd, 0o750)
            identity = os.fstat(child_fd)
        actual_mode = stat.S_IMODE(identity.st_mode)
        if actual_mode != 0o750:
            raise RuntimeError(
                f"publication directory mode must be 0750, got "
                f"{actual_mode:04o}: {path}"
            )
    except OSError as exc:
        raise RuntimeError(
            f"cannot securely prepare publication directory {path}: {exc}"
        ) from exc
    finally:
        if child_fd is not None:
            os.close(child_fd)
        os.close(parent_fd)


def _create_directory(path: Path, *, mode: int = 0o700) -> None:
    """Create one safe child directory and refuse pre-existing state."""
    if not _ensure_directory(path, mode=mode):
        raise RuntimeError(f"refusing to reuse directory: {path}")


def _read_all(fd: int, *, max_bytes: int | None = None) -> bytes:
    blocks: list[bytes] = []
    total = 0
    while True:
        block = os.read(fd, _COPY_BLOCK_SIZE)
        if not block:
            break
        total += len(block)
        if max_bytes is not None and total > max_bytes:
            raise RuntimeError(f"file exceeds safe size limit of {max_bytes} bytes")
        blocks.append(block)
    return b"".join(blocks)


def _same_file(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _copy_authenticated_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    label: str,
) -> str:
    """Copy and hash one regular single-link source through the same FD."""
    source_parent_fd = _open_directory(source.parent)
    destination_parent_fd = _open_directory(destination.parent)
    source_fd: int | None = None
    destination_fd: int | None = None
    created = False
    try:
        try:
            source_fd = os.open(source.name, _READ_FLAGS, dir_fd=source_parent_fd)
        except OSError as exc:
            raise RuntimeError(f"cannot securely open {label} {source}: {exc}") from exc
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise RuntimeError(f"{label} is not a nonempty regular file: {source}")
        if before.st_nlink != 1:
            raise RuntimeError(f"{label} must have exactly one hard link: {source}")

        try:
            destination_fd = os.open(
                destination.name,
                _CREATE_FLAGS,
                0o600,
                dir_fd=destination_parent_fd,
            )
            created = True
        except OSError as exc:
            raise RuntimeError(
                f"cannot create private {label} snapshot {destination}: {exc}"
            ) from exc

        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(source_fd, _COPY_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
            total += len(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        after = os.fstat(source_fd)
        if not _same_file(before, after) or total != before.st_size:
            raise RuntimeError(f"{label} changed while it was being copied: {source}")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{label} SHA256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        os.fsync(destination_fd)
        os.fchmod(destination_fd, 0o400)
        return actual_sha256
    except BaseException:
        if created:
            try:
                os.unlink(destination.name, dir_fd=destination_parent_fd)
            except OSError:
                pass
        raise
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)
        os.close(destination_parent_fd)
        os.close(source_parent_fd)


def _load_authenticated_manifest(path: Path) -> dict[str, Any]:
    parent_fd = _open_directory(path.parent)
    file_fd: int | None = None
    try:
        file_fd = os.open(path.name, _READ_FLAGS, dir_fd=parent_fd)
        identity = os.fstat(file_fd)
        if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
            raise RuntimeError(
                f"private split manifest is not a single-link file: {path}"
            )
        raw = _read_all(file_fd, max_bytes=_MANIFEST_MAX_BYTES)
    except OSError as exc:
        raise RuntimeError(f"cannot read private split manifest {path}: {exc}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid private split manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError("frozen split manifest must be a JSON object")
    return value


def _manifest_artifact_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    artifacts = manifest.get("artifacts")
    expected_names = {
        _SPLIT_FILES["index"],
        _SPLIT_FILES["split"],
        _VALIDATOR_FILENAME,
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise RuntimeError(
            "frozen split manifest must declare exactly the distill index, "
            "validator holdout, and split document"
        )
    result: dict[str, str] = {}
    for name, value in artifacts.items():
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in _SHA256_HEX for character in value)
        ):
            raise RuntimeError(f"frozen split manifest has malformed SHA256 for {name}")
        result[name] = value
    return result


def snapshot_crop_inputs(
    source_dir: Path,
    snapshot_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> CropSplitSnapshot:
    """Authenticate and privately snapshot only crop-visible split inputs."""
    _create_directory(snapshot_dir)
    local_manifest = snapshot_dir / _SPLIT_FILES["manifest"]
    _copy_authenticated_file(
        source_dir / _SPLIT_FILES["manifest"],
        local_manifest,
        expected_sha256=expected_manifest_sha256,
        label="frozen split manifest",
    )
    manifest = _load_authenticated_manifest(local_manifest)
    artifact_hashes = _manifest_artifact_hashes(manifest)

    local_index = snapshot_dir / _SPLIT_FILES["index"]
    local_split = snapshot_dir / _SPLIT_FILES["split"]
    for source_name, destination in (
        (_SPLIT_FILES["index"], local_index),
        (_SPLIT_FILES["split"], local_split),
    ):
        _copy_authenticated_file(
            source_dir / source_name,
            destination,
            expected_sha256=artifact_hashes[source_name],
            label=f"frozen split artifact {source_name}",
        )

    entries = sorted(path.name for path in snapshot_dir.iterdir())
    if entries != sorted(_SPLIT_FILES.values()):
        raise RuntimeError(f"private split snapshot has unexpected entries: {entries}")
    return CropSplitSnapshot(
        root=snapshot_dir,
        index=local_index,
        split=local_split,
        manifest=local_manifest,
        manifest_sha256=expected_manifest_sha256,
    )


def publish_create_only(source: Path, destination: Path) -> PublishedArtifact:
    """Copy across filesystems and atomically publish without replacement."""
    source_parent_fd = _open_directory(source.parent)
    destination_parent_fd = _open_directory(destination.parent)
    source_fd: int | None = None
    temporary_fd: int | None = None
    temporary_name = f".{destination.name}.{os.getpid()}.{secrets.token_hex(16)}.create"
    temporary_created = False
    try:
        try:
            source_fd = os.open(source.name, _READ_FLAGS, dir_fd=source_parent_fd)
        except OSError as exc:
            raise RuntimeError(f"cannot securely open output {source}: {exc}") from exc
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise RuntimeError(f"output is not a nonempty regular file: {source}")
        if before.st_nlink != 1:
            raise RuntimeError(f"output must have exactly one hard link: {source}")

        try:
            temporary_fd = os.open(
                temporary_name,
                _CREATE_FLAGS,
                0o600,
                dir_fd=destination_parent_fd,
            )
            temporary_created = True
        except OSError as exc:
            raise RuntimeError(
                f"cannot create output publication temporary for {destination}: {exc}"
            ) from exc

        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(source_fd, _COPY_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
            total += len(block)
            view = memoryview(block)
            while view:
                written = os.write(temporary_fd, view)
                view = view[written:]
        after = os.fstat(source_fd)
        if not _same_file(before, after) or total != before.st_size:
            raise RuntimeError(f"output changed while publishing: {source}")
        os.fsync(temporary_fd)
        os.fchmod(temporary_fd, 0o444)
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=destination_parent_fd,
                dst_dir_fd=destination_parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise RuntimeError(f"refusing to overwrite output: {destination}") from exc
        except OSError as exc:
            raise RuntimeError(f"cannot publish output {destination}: {exc}") from exc
        os.unlink(temporary_name, dir_fd=destination_parent_fd)
        temporary_created = False
        try:
            os.fsync(destination_parent_fd)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise RuntimeError(
                    f"cannot sync output directory {destination.parent}: {exc}"
                ) from exc
        return PublishedArtifact(
            path=destination,
            size_bytes=total,
            sha256=digest.hexdigest(),
        )
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=destination_parent_fd)
            except OSError:
                pass
        if temporary_fd is not None:
            os.close(temporary_fd)
        if source_fd is not None:
            os.close(source_fd)
        os.close(destination_parent_fd)
        os.close(source_parent_fd)


class CropDistillJob:
    """One model column with fail-closed terminal provenance."""

    def __init__(
        self,
        model: str,
        identity: RuntimeIdentity,
        split_manifest_sha256: str,
    ) -> None:
        self.model = model
        self.protocol: CropModelProtocol = model_protocol(model)
        self.identity = identity
        self.split_manifest_sha256 = split_manifest_sha256
        self.job_name = f"ladder-crop-distill-{model}"
        self.failure_stage = "bootstrap"
        self.work_dir = WORK_ROOT / identity.pod_uid
        self.snapshot_dir = self.work_dir / "split"
        self.features_work = self.work_dir / f"{model}_r2_crop_features.parquet"
        self.oof_work = self.work_dir / f"{model}_r2_crop_distillability.json"
        # Kubernetes mounts only this model's pre-owned backing directory at
        # CROP_HEADS_DIR. No crop Pod can see or squat a sibling model path.
        self.output_parent = CROP_HEADS_DIR
        # Private intermediate bytes stay under /work/<pod-uid>. Published
        # artifacts leave that 0700 tree and use a collision-free Pod identity
        # directly in the model-owned, group-readable publication directory.
        self.features = (
            self.output_parent / f"{identity.pod_uid}--{self.features_work.name}"
        )
        self.oof = self.output_parent / f"{identity.pod_uid}--{self.oof_work.name}"

    def _run(self, stage: str, command: Sequence[object]) -> None:
        self.failure_stage = stage
        argv = [str(value) for value in command]
        print(f"[{stage}] {shlex.join(argv)}", flush=True)
        subprocess.run(argv, cwd=SOURCE_ROOT, check=True)

    def _runtime_args(self, *, diagnostic: bool = False) -> list[str]:
        source_git_sha = self.identity.source_git_sha
        image_ref = self.identity.image_ref
        if diagnostic:
            source_git_sha = _bounded_claim(source_git_sha, "source-git-sha")
            image_ref = _bounded_claim(image_ref, "image-ref")
            # Equals-form prevents a diagnostic beginning with '-' from being
            # reinterpreted as a provenance CLI option.
            return [
                f"--source-git-sha={source_git_sha}",
                f"--image-ref={image_ref}",
                f"--runtime-manifest={RUNTIME_MANIFEST}",
            ]
        return [
            "--source-git-sha",
            source_git_sha,
            "--image-ref",
            image_ref,
            "--runtime-manifest",
            str(RUNTIME_MANIFEST),
        ]

    def _provenance_base(self, *, status: str, exit_code: int) -> list[str]:
        diagnostic = status == "failed"
        split_sha256 = self.split_manifest_sha256
        if diagnostic:
            split_sha256 = _bounded_claim(split_sha256, "split-manifest-sha256")
            split_args = [f"--split-sha256={split_sha256}"]
        else:
            split_args = ["--split-sha256", split_sha256]
        return [
            str(BASE_PYTHON),
            str(PROVENANCE_SCRIPT),
            "finalize",
            "--kind",
            "crop",
            "--model",
            self.model,
            "--record-dir",
            str(CROP_RECORD_DIR),
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
            *split_args,
            *self._runtime_args(diagnostic=diagnostic),
        ]

    def _prepare_work_directories(self) -> None:
        self.failure_stage = "prepare-work"
        work_root_fd = _open_directory(WORK_ROOT)
        os.close(work_root_fd)
        for name in ("home", "tmp"):
            _ensure_directory(WORK_ROOT / name)
        _create_directory(self.work_dir)

    def _prepare_output_dir(self) -> None:
        self.failure_stage = "prepare-output"
        root_fd = _open_directory(CROP_HEADS_DIR)
        os.close(root_fd)
        _ensure_publish_directory(self.output_parent)

    def execute(self) -> None:
        print(f"=== LUCAS crop-distill stage — {self.model} ===", flush=True)
        self._prepare_work_directories()
        self._run(
            "verify-runtime",
            [
                BASE_PYTHON,
                PROVENANCE_SCRIPT,
                "verify-runtime",
                *self._runtime_args(),
            ],
        )

        self.failure_stage = "snapshot-split"
        snapshot = snapshot_crop_inputs(
            CROP_SPLIT_MANIFEST.parent,
            self.snapshot_dir,
            expected_manifest_sha256=self.split_manifest_sha256,
        )
        self._run(
            "verify-split",
            [
                SCORING_PYTHON,
                SPLIT_SCRIPT,
                "--verify-consumer",
                "--out-dir",
                snapshot.root,
                "--expected-git-sha",
                self.identity.source_git_sha,
            ],
        )
        self._prepare_output_dir()

        extract_command: list[object] = [
            MODEL_PYTHON,
            EXTRACT_SCRIPT,
            "--checkpoint",
            self.protocol.checkpoint_path,
            "--checkpoint-size",
            self.protocol.checkpoint_size,
            "--checkpoint-sha256",
            self.protocol.checkpoint_sha256,
            "--plot-index",
            snapshot.index,
            "--truth-col",
            TRUTH_COLUMN,
            "--img-size",
            self.protocol.img_size,
            "--backbone-name",
            self.protocol.backbone,
            "--enable-markfukt",
            "--tile-inventory",
            snapshot.split,
            "--tile-inventory-partition",
            "distill",
        ]
        for key in self.protocol.required_npz_keys:
            extract_command.extend(("--require-npz-key", key))
        extract_command.extend(("--out", self.features_work, "--device", DEVICE))
        self._run("extract-features", extract_command)

        self._run(
            "score-oof",
            [
                SCORING_PYTHON,
                SCORE_SCRIPT,
                "--features",
                self.features_work,
                "--folds",
                OOF_FOLDS,
                "--heads",
                OOF_HEADS,
                "--truth-col",
                TRUTH_COLUMN,
                "--group-col",
                OOF_GROUP_COLUMN,
                "--git-sha",
                self.identity.source_git_sha,
                "--pinned-plots",
                snapshot.split,
                "--out",
                self.oof_work,
            ],
        )

        self.failure_stage = "publish-artifacts"
        published = {
            "features": publish_create_only(self.features_work, self.features),
            "oof": publish_create_only(self.oof_work, self.oof),
        }
        artifact_args: list[object] = []
        for name in sorted(published):
            identity = published[name]
            artifact_args.extend(("--artifact", f"{name}={identity.path}"))
            artifact_args.extend(("--artifact-size", f"{name}={identity.size_bytes}"))
            artifact_args.extend(("--artifact-sha256", f"{name}={identity.sha256}"))
        self._run(
            "publish-completion",
            [
                *self._provenance_base(status="completed", exit_code=0),
                "--split-manifest",
                snapshot.manifest,
                "--checkpoint",
                self.protocol.checkpoint_path,
                "--checkpoint-sha256",
                self.protocol.checkpoint_sha256,
                "--checkpoint-size",
                self.protocol.checkpoint_size,
                *artifact_args,
            ],
        )
        print(f"=== crop-distill complete for {self.model} ===", flush=True)

    def publish_failure(self, exit_code: int) -> None:
        """Publish bounded identity claims without reading untrusted inputs."""
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
    if isinstance(exc, subprocess.CalledProcessError):
        if exc.returncode < 0:
            return min(255, 128 + abs(exc.returncode))
        if exc.returncode > 0:
            return min(255, exc.returncode)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = JobArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_KEYS)
    return parser


def _publish_argument_failure(
    identity: RuntimeIdentity,
    split_manifest_sha256: str,
    diagnostic: str,
) -> int:
    """Record a baked-argument failure once a safe Pod UID is available."""
    command = [
        str(BASE_PYTHON),
        str(PROVENANCE_SCRIPT),
        "finalize",
        "--kind",
        "crop",
        "--record-dir",
        str(CROP_RECORD_DIR),
        "--run-id",
        identity.pod_uid,
        "--job",
        "ladder-crop-distill-argument-error",
        "--pod-uid",
        identity.pod_uid,
        "--status",
        "failed",
        "--exit-code",
        "2",
        "--failure-stage="
        + _bounded_claim(
            f"parse-arguments: {diagnostic}",
            "argument-error",
            max_length=191,
        ),
        "--split-sha256="
        + _bounded_claim(split_manifest_sha256, "split-manifest-sha256"),
        "--source-git-sha=" + _bounded_claim(identity.source_git_sha, "source-git-sha"),
        "--image-ref=" + _bounded_claim(identity.image_ref, "image-ref"),
        f"--runtime-manifest={RUNTIME_MANIFEST}",
    ]
    argv = [str(value) for value in command]
    print(f"[publish-failure] {shlex.join(argv)}", file=sys.stderr, flush=True)
    try:
        subprocess.run(argv, cwd=SOURCE_ROOT, check=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"FATAL: argument failure also lacks publishable provenance: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 97
    return 2


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    environment = os.environ if environ is None else environ
    claims = runtime_claims(environment)
    raw_split_sha256 = split_manifest_claim(environment)
    try:
        args = build_parser().parse_args(argv)
    except JobArgumentError as exc:
        print(f"FATAL [parse-arguments] JobArgumentError: {exc}", file=sys.stderr)
        return _publish_argument_failure(claims, raw_split_sha256, str(exc))
    job = CropDistillJob(args.model, claims, raw_split_sha256)
    # This is the job boundary: every unexpected runtime failure must reach
    # immutable terminal provenance rather than escape unrecorded.
    try:
        job.failure_stage = "validate-runtime-environment"
        job.identity = runtime_identity(environment)
        job.failure_stage = "validate-split-manifest-environment"
        job.split_manifest_sha256 = require_split_manifest_sha256(raw_split_sha256)
        job.failure_stage = "validate-process-identity"
        require_process_identity(
            model_process_uid(args.model),
            expected_gid=STORAGE_GID,
            role=f"crop-distill model {args.model}",
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
