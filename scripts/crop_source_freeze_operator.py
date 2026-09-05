#!/usr/bin/env python3
"""Run the crop-source freeze watchdog independently inside Kubernetes."""

from __future__ import annotations

import argparse
import errno
import hashlib
import os
import signal
import stat
import sys
import time
from pathlib import Path

if __package__:
    from . import crop_source_freeze as freeze
else:
    import crop_source_freeze as freeze

OPERATOR_UID = 2000
OPERATOR_GID = 2000
STATE_MODE = 0o700
STATE_SUBDIR = "crop-source-freeze"


class OperatorError(RuntimeError):
    """The in-cluster operator cannot preserve the freeze contract."""


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise OperatorError(f"state parent is not a real directory: {path}") from exc


def prepare_state_root(
    parent: Path,
    *,
    uid: int = OPERATOR_UID,
    gid: int = OPERATOR_GID,
) -> Path:
    """Create only the operator's restricted, pre-owned state directory."""
    parent_fd = _open_directory(parent)
    child_fd: int | None = None
    try:
        try:
            before = os.stat(STATE_SUBDIR, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            os.mkdir(STATE_SUBDIR, mode=STATE_MODE, dir_fd=parent_fd)
            before = os.stat(STATE_SUBDIR, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode) or before.st_uid not in {0, uid}:
            raise OperatorError("state root has an unexpected type or owner")

        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        child_fd = os.open(STATE_SUBDIR, flags, dir_fd=parent_fd)
        opened = os.fstat(child_fd)
        if opened.st_dev != before.st_dev or opened.st_ino != before.st_ino:
            raise OperatorError("state root changed while opening")
        os.fchown(child_fd, uid, gid)
        os.fchmod(child_fd, STATE_MODE)
        os.fsync(child_fd)
        after = os.fstat(child_fd)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_uid != uid
            or after.st_gid != gid
            or stat.S_IMODE(after.st_mode) != STATE_MODE
        ):
            raise OperatorError("state root identity or permissions did not settle")
        try:
            os.fsync(parent_fd)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise OperatorError("cannot sync state parent") from exc
    finally:
        if child_fd is not None:
            os.close(child_fd)
        os.close(parent_fd)
    return parent / STATE_SUBDIR


def _require_state_root(
    path: Path,
    *,
    uid: int = OPERATOR_UID,
    gid: int = OPERATOR_GID,
) -> None:
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise OperatorError(f"state root is unavailable: {path}") from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or current.st_uid != uid
        or current.st_gid != gid
        or stat.S_IMODE(current.st_mode) != STATE_MODE
    ):
        raise OperatorError("state root must be a real 0700 operator directory")


def _wait_for_exact_restore(
    client: freeze.Kubectl,
    *,
    run_dir: Path,
    poll_seconds: float,
) -> None:
    """Remain exec-capable until the durable run proves exact release."""
    while True:
        if (run_dir / "controllers-restored.json").exists():
            if not (run_dir / "restore-in-progress.json").exists():
                raise OperatorError(
                    "controllers-restored exists without restore ownership"
                )
            lease = freeze._live_lease(client, run_id=run_dir.name)
            if lease.get("status") == "released":
                print(f"in-cluster freeze restored exactly: run={run_dir.name}")
                return
        time.sleep(poll_seconds)


def _validate_incomplete_hold_for_restore(
    client: freeze.Kubectl,
    *,
    run_dir: Path,
) -> None:
    """Prove an interrupted hold reached state that core restore accepts."""
    before_path = run_dir / "controllers-before.json"
    try:
        before = freeze._read_json(before_path)
    except (OSError, freeze.FreezeError) as exc:
        raise OperatorError(
            "incomplete run lacks controller-before authority"
        ) from exc
    entries = before.get("controllers")
    if not isinstance(entries, list):
        raise OperatorError("incomplete controller-before authority is malformed")
    names = {
        entry.get("name")
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    if (
        before.get("schema") != freeze.FREEZE_SCHEMA
        or before.get("run_id") != run_dir.name
        or names != set(freeze.ALL_CONTROLLERS)
    ):
        raise OperatorError("incomplete controller-before authority is malformed")

    try:
        lease = freeze._live_lease(client, run_id=run_dir.name)
    except freeze.FreezeError as exc:
        raise OperatorError("incomplete run lacks its live lease") from exc
    if lease.get("status") not in {
        "initializing",
        "failed",
        "closed",
        "held",
        "released",
    }:
        raise OperatorError("incomplete run has an unsafe live lease state")
    controller_hashes = {hashlib.sha256(before_path.read_bytes()).hexdigest()}
    held_path = run_dir / "controllers-held.json"
    if held_path.exists():
        controller_hashes.add(hashlib.sha256(held_path.read_bytes()).hexdigest())
    if lease.get("controller_snapshot_sha256") not in controller_hashes:
        raise OperatorError("incomplete run controller authority differs")


def serve(
    client: freeze.Kubectl,
    *,
    state_dir: Path,
    run_id: str,
    interval_seconds: float = freeze.DEFAULT_INTERVAL_SECONDS,
    poll_seconds: float = 1.0,
    uid: int = OPERATOR_UID,
    gid: int = OPERATOR_GID,
) -> None:
    """Acquire, watch, and remain exec-capable until exact restore completes."""
    _require_state_root(state_dir, uid=uid, gid=gid)
    run_dir = state_dir / run_id
    if run_dir.exists():
        if not (run_dir / "hold-complete.json").exists():
            _validate_incomplete_hold_for_restore(client, run_dir=run_dir)
            print(
                f"in-cluster hold incomplete; awaiting exact restore: "
                f"run={run_id}",
                file=sys.stderr,
            )
            _wait_for_exact_restore(
                client,
                run_dir=run_dir,
                poll_seconds=poll_seconds,
            )
            return
        try:
            freeze._verify_hold_record_hashes(run_dir)
        except (OSError, freeze.FreezeError) as exc:
            raise OperatorError(
                f"existing run lacks valid completed-hold authority: {run_dir}"
            ) from exc
        if (
            (run_dir / "restore-in-progress.json").exists()
            or (run_dir / "watchdog-stopped.json").exists()
            or (run_dir / "controllers-restored.json").exists()
        ):
            _wait_for_exact_restore(
                client,
                run_dir=run_dir,
                poll_seconds=poll_seconds,
            )
            return
        acquired = run_dir
    else:
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        previous_sigint = signal.getsignal(signal.SIGINT)

        def interrupt_hold(signum: int, _frame: object) -> None:
            raise freeze.FreezeError(
                f"operator hold interrupted by signal {signum}"
            )

        signal.signal(signal.SIGTERM, interrupt_hold)
        signal.signal(signal.SIGINT, interrupt_hold)
        hold_error: Exception | None = None
        try:
            acquired = freeze.hold(client, state_dir=state_dir, run_id=run_id)
        except Exception as exc:
            hold_error = exc
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            signal.signal(signal.SIGINT, previous_sigint)
        if hold_error is not None:
            if run_dir.exists():
                _validate_incomplete_hold_for_restore(client, run_dir=run_dir)
                print(
                    f"in-cluster hold failed closed; awaiting exact restore: "
                    f"run={run_id}",
                    file=sys.stderr,
                )
                _wait_for_exact_restore(
                    client,
                    run_dir=run_dir,
                    poll_seconds=poll_seconds,
                )
                return
            raise hold_error

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)
    watch_failed_closed = False
    try:
        freeze.watch(
            client,
            run_dir=acquired,
            interval_seconds=interval_seconds,
            fail_on_signal=True,
        )
    except freeze.FreezeError:
        lease = freeze._live_lease(client, run_id=run_id)
        if lease.get("status") != "failed":
            raise
        watch_failed_closed = True
        print(
            f"in-cluster watchdog failed closed; awaiting exact restore: "
            f"run={run_id}",
            file=sys.stderr,
        )
    finally:
        # watch owns cooperative handlers only for its critical section.
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)

    if not watch_failed_closed and not (
        acquired / "watchdog-stopped.json"
    ).exists():
        raise OperatorError("watchdog stopped without restore authority")
    _wait_for_exact_restore(
        client,
        run_dir=acquired,
        poll_seconds=poll_seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--state-parent", type=Path, required=True)

    run = subparsers.add_parser("serve")
    run.add_argument("--state-dir", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--namespace", default=freeze.NAMESPACE)
    run.add_argument(
        "--interval-seconds",
        type=float,
        default=freeze.DEFAULT_INTERVAL_SECONDS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            if os.geteuid() != 0 or os.getegid() != OPERATOR_GID:
                raise OperatorError("state preparation requires UID 0:GID 2000")
            path = prepare_state_root(args.state_parent)
            print(f"operator state root ready: {path}")
        else:
            if os.geteuid() != OPERATOR_UID or os.getegid() != OPERATOR_GID:
                raise OperatorError("operator requires UID 2000:GID 2000")
            serve(
                freeze.Kubectl(context="", namespace=args.namespace),
                state_dir=args.state_dir,
                run_id=args.run_id,
                interval_seconds=args.interval_seconds,
            )
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        print(
            f"crop source freeze operator refused: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
