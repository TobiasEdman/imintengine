#!/usr/bin/env python3
"""Capture and verify durable crop-distill Kubernetes evidence.

The terminal completion record is emitted by the workload as a machine-readable
marker.  This helper extracts exactly one such marker from a captured Pod log,
binds it to the corresponding Kubernetes Pod, and creates a small immutable
evidence bundle outside the workload PVC.

This is an evidence-capture control, not an admission control.  It proves what
the Kubernetes API and workload log returned at capture time; it does not make
the scheduler or container runtime choose an image.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

if __package__:
    from .crop_distill_provenance import (
        COMPLETION_SCHEMA,
        TERMINAL_EVIDENCE_PREFIX,
        ProvenanceError,
        canonical_json_bytes,
        parse_terminal_evidence_line,
    )
else:
    from crop_distill_provenance import (  # type: ignore[no-redef]
        COMPLETION_SCHEMA,
        TERMINAL_EVIDENCE_PREFIX,
        ProvenanceError,
        canonical_json_bytes,
        parse_terminal_evidence_line,
    )

CAPTURE_SCHEMA = "imint-crop-distill-evidence-capture-v1"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")
_IMAGE_VALUE = re.compile(
    r"^(?:(?:containerd|docker-pullable)://)?"
    r"(?:[^\s@]+@)?sha256:([0-9a-f]{64})$"
)
_CAPTURE_FILES = frozenset({"completion.json", "completion.sha256", "capture.json"})
_MAX_POD_JSON_BYTES = 8 * 1024 * 1024
_MAX_POD_LOG_BYTES = 64 * 1024 * 1024
_MAX_BUNDLE_FILE_BYTES = 16 * 1024 * 1024


class EvidenceCaptureError(ValueError):
    """Raised when live or archived evidence cannot be trusted."""


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceCaptureError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvidenceCaptureError(f"{label} must be a JSON array")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceCaptureError(f"{label} must be a non-empty string")
    return value


def _require_safe_id(value: Any, label: str) -> str:
    value = _require_string(value, label)
    if _SAFE_ID.fullmatch(value) is None:
        raise EvidenceCaptureError(f"{label} is not a safe identifier")
    return value


def _read_regular_file(path: Path, label: str, *, limit: int) -> bytes:
    """Read one bounded regular file without following its final symlink."""
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise EvidenceCaptureError(f"cannot open {label} {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceCaptureError(f"{label} must be a regular file: {path}")
        if before.st_size > limit:
            raise EvidenceCaptureError(
                f"{label} exceeds the {limit}-byte capture limit: {path}"
            )
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > limit:
            raise EvidenceCaptureError(
                f"{label} exceeds the {limit}-byte capture limit: {path}"
            )
        after = os.fstat(fd)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity or len(payload) != after.st_size:
            raise EvidenceCaptureError(f"{label} changed while it was read: {path}")
        return payload
    except OSError as exc:
        raise EvidenceCaptureError(f"cannot read {label} {path}: {exc}") from exc
    finally:
        os.close(fd)


def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceCaptureError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    return _require_object(value, label)


def normalize_image_digest(value: Any, label: str) -> str:
    """Normalize accepted Kubernetes image forms to one lowercase hex digest."""
    value = _require_string(value, label)
    match = _IMAGE_VALUE.fullmatch(value)
    if match is None:
        raise EvidenceCaptureError(f"{label} must be an immutable sha256 image value")
    return match.group(1)


def extract_terminal_record(pod_log: bytes) -> tuple[bytes, str, dict[str, Any]]:
    """Extract and authenticate exactly one terminal marker from a Pod log."""
    try:
        text = pod_log.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceCaptureError("Pod log is not valid UTF-8") from exc

    marker_lines = [
        line for line in text.splitlines() if TERMINAL_EVIDENCE_PREFIX in line
    ]
    if len(marker_lines) != 1:
        raise EvidenceCaptureError(
            "Pod log must contain exactly one terminal evidence marker; "
            f"found {len(marker_lines)}"
        )
    line = marker_lines[0]
    try:
        actual_sha256, payload, record = parse_terminal_evidence_line(line)
    except ProvenanceError as exc:
        raise EvidenceCaptureError(
            f"terminal evidence marker is invalid: {exc}"
        ) from exc
    encoded = line.split(" ")[2]
    if base64.b64encode(payload).decode("ascii") != encoded:
        raise EvidenceCaptureError(
            "terminal evidence marker payload is not canonical base64"
        )
    return payload, actual_sha256, record


def _validate_completion_record(record: dict[str, Any]) -> dict[str, str]:
    if record.get("schema") != COMPLETION_SCHEMA:
        raise EvidenceCaptureError("unexpected terminal completion schema")
    kind = record.get("kind")
    if kind not in {"crop", "split"}:
        raise EvidenceCaptureError("completion kind must be crop or split")
    pod_uid = _require_safe_id(record.get("pod_uid"), "completion pod_uid")
    run_id = _require_safe_id(record.get("run_id"), "completion run_id")
    if run_id != pod_uid:
        raise EvidenceCaptureError("completion run_id must equal pod_uid")
    job = _require_safe_id(record.get("job"), "completion job")
    model = record.get("model")
    if kind == "crop":
        model = _require_safe_id(model, "completion model")
        if job != f"ladder-crop-distill-{model}":
            raise EvidenceCaptureError("crop completion job does not match model")
        container = "crop-distill"
    else:
        if model is not None:
            raise EvidenceCaptureError("split completion model must be null")
        if job != "ladder-lucas-crop-split":
            raise EvidenceCaptureError("split completion has an unexpected job")
        container = "split"

    terminal = _require_object(record.get("terminal"), "completion terminal")
    if terminal.get("status") != "completed":
        raise EvidenceCaptureError("only completed terminal records can be archived")
    if terminal.get("exit_code") != 0:
        raise EvidenceCaptureError("completed terminal record must have exit_code 0")
    if terminal.get("failure_stage") is not None:
        raise EvidenceCaptureError(
            "completed terminal record must not have a failure_stage"
        )

    runtime = _require_object(record.get("runtime"), "completion runtime")
    if runtime.get("verification") != "verified":
        raise EvidenceCaptureError("completion runtime must be verified")
    image = _require_object(runtime.get("image"), "completion runtime image")
    digest = _require_string(image.get("digest"), "completion image digest")
    if _HEX64.fullmatch(digest) is None:
        raise EvidenceCaptureError(
            "completion image digest must be 64 lowercase hex digits"
        )
    ref_digest = normalize_image_digest(
        image.get("ref"), "completion runtime image ref"
    )
    if ref_digest != digest:
        raise EvidenceCaptureError("completion runtime image ref and digest disagree")
    return {
        "container": container,
        "digest": digest,
        "job": job,
        "kind": kind,
        "pod_uid": pod_uid,
        "run_id": run_id,
    }


def _named_entry(entries: Any, target: str, label: str) -> dict[str, Any]:
    entries = _require_list(entries, label)
    objects = [_require_object(entry, f"{label} entry") for entry in entries]
    matches = [entry for entry in objects if entry.get("name") == target]
    if len(matches) != 1:
        raise EvidenceCaptureError(
            f"{label} must contain target {target!r} exactly once; found {len(matches)}"
        )
    return matches[0]


def _reject_target_in_other_container_classes(pod: dict[str, Any], target: str) -> None:
    spec = _require_object(pod.get("spec"), "Pod spec")
    status = _require_object(pod.get("status"), "Pod status")
    for owner, field in (
        (spec, "initContainers"),
        (spec, "ephemeralContainers"),
        (status, "initContainerStatuses"),
        (status, "ephemeralContainerStatuses"),
    ):
        entries = owner.get(field, [])
        entries = _require_list(entries, f"Pod {field}")
        entries = [_require_object(entry, f"Pod {field} entry") for entry in entries]
        if any(entry.get("name") == target for entry in entries):
            raise EvidenceCaptureError(
                f"target container {target!r} is ambiguous with Pod {field}"
            )


def build_capture(
    pod: dict[str, Any],
    record: dict[str, Any],
    *,
    record_sha256: str,
    container: str,
    expected_namespace: str,
    expected_pod: str,
    expected_job: str,
) -> dict[str, Any]:
    """Bind a completed record to the exact observed Pod and runtime image."""
    completion = _validate_completion_record(record)
    if container != completion["container"]:
        raise EvidenceCaptureError(
            f"container must be {completion['container']!r} for a "
            f"{completion['kind']} record"
        )
    if expected_job != completion["job"]:
        raise EvidenceCaptureError("expected job disagrees with completion record")
    if pod.get("apiVersion") != "v1" or pod.get("kind") != "Pod":
        raise EvidenceCaptureError("captured Kubernetes object must be a v1 Pod")

    metadata = _require_object(pod.get("metadata"), "Pod metadata")
    namespace = _require_safe_id(metadata.get("namespace"), "Pod namespace")
    pod_name = _require_safe_id(metadata.get("name"), "Pod name")
    pod_uid = _require_safe_id(metadata.get("uid"), "Pod UID")
    if namespace != expected_namespace:
        raise EvidenceCaptureError("Pod namespace does not match --expected-namespace")
    if pod_name != expected_pod:
        raise EvidenceCaptureError("Pod name does not match --expected-pod")
    if pod_uid != completion["pod_uid"]:
        raise EvidenceCaptureError("Pod UID does not match completion record")

    labels = _require_object(metadata.get("labels"), "Pod labels")
    job_labels: dict[str, str] = {}
    for key in ("batch.kubernetes.io/job-name", "job-name"):
        if key in labels:
            value = _require_string(labels[key], f"Pod label {key}")
            if value != completion["job"]:
                raise EvidenceCaptureError(
                    f"Pod label {key} does not match completion job"
                )
            job_labels[key] = value
    if "batch.kubernetes.io/job-name" not in job_labels:
        raise EvidenceCaptureError(
            "Pod lacks the batch.kubernetes.io/job-name binding label"
        )

    owner_references = _require_list(
        metadata.get("ownerReferences"), "Pod ownerReferences"
    )
    owner_references = [
        _require_object(owner, "Pod ownerReference") for owner in owner_references
    ]
    controllers = [
        owner for owner in owner_references if owner.get("controller") is True
    ]
    if len(controllers) != 1:
        raise EvidenceCaptureError(
            "Pod must have exactly one controller ownerReference"
        )
    controller = controllers[0]
    if (
        controller.get("apiVersion") != "batch/v1"
        or controller.get("kind") != "Job"
        or controller.get("name") != completion["job"]
    ):
        raise EvidenceCaptureError(
            "Pod controller ownerReference does not match completion Job"
        )
    job_uid = _require_safe_id(controller.get("uid"), "Job controller UID")

    _reject_target_in_other_container_classes(pod, container)
    spec = _require_object(pod.get("spec"), "Pod spec")
    status = _require_object(pod.get("status"), "Pod status")
    spec_container = _named_entry(spec.get("containers"), container, "Pod containers")
    status_container = _named_entry(
        status.get("containerStatuses"), container, "Pod containerStatuses"
    )
    spec_image = _require_string(spec_container.get("image"), "Pod spec image")
    status_image_id = _require_string(
        status_container.get("imageID"), "Pod status imageID"
    )
    spec_digest = normalize_image_digest(spec_image, "Pod spec image")
    status_digest = normalize_image_digest(status_image_id, "Pod status imageID")
    expected_digest = completion["digest"]
    if spec_digest != expected_digest:
        raise EvidenceCaptureError(
            "Pod spec image digest does not match completion image digest"
        )
    if status_digest != expected_digest:
        raise EvidenceCaptureError(
            "Pod status imageID digest does not match completion image digest"
        )

    if status.get("phase") != "Succeeded":
        raise EvidenceCaptureError("Pod phase must be Succeeded")
    state = _require_object(status_container.get("state"), "container state")
    if set(state) != {"terminated"}:
        raise EvidenceCaptureError(
            "target container must have exactly one terminated state"
        )
    terminated = _require_object(state["terminated"], "terminated state")
    if terminated.get("exitCode") != 0:
        raise EvidenceCaptureError("target container exitCode must be 0")
    terminated_reason = terminated.get("reason")
    if terminated_reason is not None:
        terminated_reason = _require_string(
            terminated_reason, "terminated state reason"
        )

    return {
        "schema": CAPTURE_SCHEMA,
        "completion": {
            "image_digest": expected_digest,
            "job": completion["job"],
            "kind": completion["kind"],
            "pod_uid": completion["pod_uid"],
            "record_sha256": record_sha256,
            "run_id": completion["run_id"],
            "schema": COMPLETION_SCHEMA,
        },
        "kubernetes": {
            "api_version": "v1",
            "container": {
                "name": container,
                "spec_image": spec_image,
                "spec_image_digest": spec_digest,
                "status_image_id": status_image_id,
                "status_image_digest": status_digest,
                "terminated_exit_code": 0,
                "terminated_reason": terminated_reason,
            },
            "job": {
                "labels": dict(sorted(job_labels.items())),
                "name": completion["job"],
                "uid": job_uid,
            },
            "namespace": namespace,
            "pod": {
                "name": pod_name,
                "phase": "Succeeded",
                "uid": pod_uid,
            },
        },
        "source": {
            "completion": "pod-log-terminal-marker-v1",
            "pod": "kubernetes-api-v1",
        },
    }


def _write_new_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(path, flags, 0o644)
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    except OSError as exc:
        raise EvidenceCaptureError(
            f"cannot create evidence file {path}: {exc}"
        ) from exc
    finally:
        if fd is not None:
            os.close(fd)


def write_bundle(
    out_dir: Path,
    completion_payload: bytes,
    completion_sha256: str,
    capture: dict[str, Any],
) -> None:
    """Create one evidence directory without replacing any existing evidence."""
    try:
        out_dir.mkdir(mode=0o755)
    except OSError as exc:
        raise EvidenceCaptureError(
            f"evidence output directory must be new: {out_dir}: {exc}"
        ) from exc
    created: list[Path] = []
    try:
        for name, payload in (
            ("completion.json", completion_payload),
            ("completion.sha256", f"{completion_sha256}\n".encode("ascii")),
            ("capture.json", canonical_json_bytes(capture)),
        ):
            path = out_dir / name
            _write_new_file(path, payload)
            created.append(path)
        directory_fd = os.open(out_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass
        try:
            out_dir.rmdir()
        except OSError:
            pass
        raise


def _verify_capture_document(
    capture: dict[str, Any],
    record: dict[str, Any],
    record_sha256: str,
) -> None:
    if capture.get("schema") != CAPTURE_SCHEMA:
        raise EvidenceCaptureError("unexpected capture.json schema")
    completion = _validate_completion_record(record)
    archived_completion = _require_object(
        capture.get("completion"), "capture completion"
    )
    expected_completion = {
        "image_digest": completion["digest"],
        "job": completion["job"],
        "kind": completion["kind"],
        "pod_uid": completion["pod_uid"],
        "record_sha256": record_sha256,
        "run_id": completion["run_id"],
        "schema": COMPLETION_SCHEMA,
    }
    if archived_completion != expected_completion:
        raise EvidenceCaptureError(
            "capture completion binding disagrees with completion.json"
        )
    if capture.get("source") != {
        "completion": "pod-log-terminal-marker-v1",
        "pod": "kubernetes-api-v1",
    }:
        raise EvidenceCaptureError("capture source description is unexpected")

    kubernetes = _require_object(capture.get("kubernetes"), "capture kubernetes")
    if kubernetes.get("api_version") != "v1":
        raise EvidenceCaptureError("capture Kubernetes API version is unexpected")
    _require_safe_id(kubernetes.get("namespace"), "capture namespace")
    pod = _require_object(kubernetes.get("pod"), "capture pod")
    _require_safe_id(pod.get("name"), "capture Pod name")
    if pod.get("uid") != completion["pod_uid"] or pod.get("phase") != "Succeeded":
        raise EvidenceCaptureError("capture Pod identity or phase is inconsistent")
    job = _require_object(kubernetes.get("job"), "capture job")
    if job.get("name") != completion["job"]:
        raise EvidenceCaptureError("capture Job name is inconsistent")
    _require_safe_id(job.get("uid"), "capture Job UID")
    labels = _require_object(job.get("labels"), "capture Job labels")
    if labels.get("batch.kubernetes.io/job-name") != completion["job"]:
        raise EvidenceCaptureError("capture Job binding label is inconsistent")
    if any(value != completion["job"] for value in labels.values()):
        raise EvidenceCaptureError("capture Job labels disagree")

    container = _require_object(kubernetes.get("container"), "capture container")
    if container.get("name") != completion["container"]:
        raise EvidenceCaptureError("capture container name is inconsistent")
    spec_digest = normalize_image_digest(
        container.get("spec_image"), "capture Pod spec image"
    )
    status_digest = normalize_image_digest(
        container.get("status_image_id"), "capture Pod status imageID"
    )
    if (
        spec_digest != completion["digest"]
        or status_digest != completion["digest"]
        or container.get("spec_image_digest") != spec_digest
        or container.get("status_image_digest") != status_digest
    ):
        raise EvidenceCaptureError("capture image digest binding is inconsistent")
    if container.get("terminated_exit_code") != 0:
        raise EvidenceCaptureError("capture target container did not exit successfully")
    reason = container.get("terminated_reason")
    if reason is not None:
        _require_string(reason, "capture terminated reason")


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Verify an archived bundle without needing the live cluster or PVC."""
    try:
        entries = {entry.name for entry in bundle_dir.iterdir()}
    except OSError as exc:
        raise EvidenceCaptureError(
            f"cannot inspect evidence bundle {bundle_dir}: {exc}"
        ) from exc
    if entries != _CAPTURE_FILES:
        raise EvidenceCaptureError(
            "evidence bundle must contain exactly completion.json, "
            "completion.sha256, and capture.json"
        )
    completion_payload = _read_regular_file(
        bundle_dir / "completion.json",
        "archived completion record",
        limit=_MAX_BUNDLE_FILE_BYTES,
    )
    record = _load_json_bytes(completion_payload, "archived completion record")
    if canonical_json_bytes(record) != completion_payload:
        raise EvidenceCaptureError(
            "archived completion record does not use canonical JSON bytes"
        )
    record_sha256 = hashlib.sha256(completion_payload).hexdigest()
    sha_payload = _read_regular_file(
        bundle_dir / "completion.sha256",
        "archived completion digest",
        limit=1024,
    )
    if sha_payload != f"{record_sha256}\n".encode("ascii"):
        raise EvidenceCaptureError("completion.sha256 does not match completion.json")
    capture_payload = _read_regular_file(
        bundle_dir / "capture.json",
        "archived capture document",
        limit=_MAX_BUNDLE_FILE_BYTES,
    )
    capture = _load_json_bytes(capture_payload, "archived capture document")
    if canonical_json_bytes(capture) != capture_payload:
        raise EvidenceCaptureError("capture.json does not use canonical JSON bytes")
    _verify_capture_document(capture, record, record_sha256)
    return capture


def capture_from_files(args: argparse.Namespace) -> dict[str, Any]:
    pod_payload = _read_regular_file(
        args.pod_json, "captured Pod JSON", limit=_MAX_POD_JSON_BYTES
    )
    log_payload = _read_regular_file(
        args.pod_log, "captured Pod log", limit=_MAX_POD_LOG_BYTES
    )
    pod = _load_json_bytes(pod_payload, "captured Pod JSON")
    completion_payload, completion_sha256, record = extract_terminal_record(log_payload)
    capture = build_capture(
        pod,
        record,
        record_sha256=completion_sha256,
        container=args.container,
        expected_namespace=args.expected_namespace,
        expected_pod=args.expected_pod,
        expected_job=args.expected_job,
    )
    write_bundle(args.out_dir, completion_payload, completion_sha256, capture)
    return verify_bundle(args.out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture", help="create and immediately verify a new evidence bundle"
    )
    capture.add_argument("--pod-json", type=Path, required=True)
    capture.add_argument("--pod-log", type=Path, required=True)
    capture.add_argument("--container", required=True)
    capture.add_argument("--expected-namespace", required=True)
    capture.add_argument("--expected-pod", required=True)
    capture.add_argument("--expected-job", required=True)
    capture.add_argument("--out-dir", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify", help="verify one archived evidence bundle offline"
    )
    verify.add_argument("--bundle-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = (
            capture_from_files(args)
            if args.command == "capture"
            else verify_bundle(args.bundle_dir)
        )
    except EvidenceCaptureError as exc:
        raise SystemExit(f"crop-distill evidence capture refused: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
