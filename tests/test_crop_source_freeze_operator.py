"""The in-cluster freeze operator owns restricted durable state."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from scripts import crop_source_freeze_operator as operator


def test_prepare_state_root_is_exact_and_idempotent(tmp_path):
    target = operator.prepare_state_root(
        tmp_path,
        uid=os.geteuid(),
        gid=os.getegid(),
    )
    first = target.stat(follow_symlinks=False)

    assert target == tmp_path / operator.STATE_SUBDIR
    assert stat.S_ISDIR(first.st_mode)
    assert stat.S_IMODE(first.st_mode) == operator.STATE_MODE

    assert operator.prepare_state_root(
        tmp_path,
        uid=os.geteuid(),
        gid=os.getegid(),
    ) == target
    second = target.stat(follow_symlinks=False)
    assert (second.st_dev, second.st_ino) == (first.st_dev, first.st_ino)


def test_prepare_state_root_rejects_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / operator.STATE_SUBDIR).symlink_to(real, target_is_directory=True)

    with pytest.raises(operator.OperatorError, match="type or owner"):
        operator.prepare_state_root(
            tmp_path,
            uid=os.geteuid(),
            gid=os.getegid(),
        )


def test_serve_holds_and_waits_for_exact_restore(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=operator.STATE_MODE)
    run_id = "attempt-14"
    client = object()
    calls = []

    def fake_hold(actual_client, *, state_dir, run_id):
        calls.append(("hold", actual_client, state_dir, run_id))
        run_dir = state_dir / run_id
        run_dir.mkdir(mode=operator.STATE_MODE)
        return run_dir

    def fake_watch(
        actual_client,
        *,
        run_dir,
        interval_seconds,
        fail_on_signal,
    ):
        calls.append(
            (
                "watch",
                actual_client,
                run_dir,
                interval_seconds,
                fail_on_signal,
            )
        )
        (run_dir / "watchdog-stopped.json").write_text("{}")
        (run_dir / "restore-in-progress.json").write_text("{}")
        (run_dir / "controllers-restored.json").write_text("{}")
        return 0

    monkeypatch.setattr(operator.freeze, "hold", fake_hold)
    monkeypatch.setattr(operator.freeze, "watch", fake_watch)
    monkeypatch.setattr(
        operator.freeze,
        "_live_lease",
        lambda actual_client, *, run_id: {"status": "released"},
    )

    operator.serve(
        client,
        state_dir=state_dir,
        run_id=run_id,
        uid=os.geteuid(),
        gid=os.getegid(),
    )

    assert [call[0] for call in calls] == ["hold", "watch"]
    assert calls[1][3] == operator.freeze.DEFAULT_INTERVAL_SECONDS
    assert calls[1][4] is True


def test_serve_refuses_watch_stop_without_restore_authority(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=operator.STATE_MODE)
    run_dir = state_dir / "interrupted"
    client = object()

    def fake_hold(*_args, **_kwargs):
        run_dir.mkdir(mode=operator.STATE_MODE)
        return run_dir

    monkeypatch.setattr(operator.freeze, "hold", fake_hold)
    monkeypatch.setattr(operator.freeze, "watch", lambda *_args, **_kwargs: 0)

    with pytest.raises(operator.OperatorError, match="without restore authority"):
        operator.serve(
            client,
            state_dir=state_dir,
            run_id=run_dir.name,
            uid=os.geteuid(),
            gid=os.getegid(),
        )


def test_serve_resumes_existing_completed_hold(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=operator.STATE_MODE)
    run_dir = state_dir / "existing"
    run_dir.mkdir(mode=operator.STATE_MODE)
    (run_dir / "hold-complete.json").write_text("{}")
    client = object()
    calls = []

    monkeypatch.setattr(
        operator.freeze,
        "_verify_hold_record_hashes",
        lambda actual_run_dir: calls.append(("verify", actual_run_dir)),
    )
    monkeypatch.setattr(
        operator.freeze,
        "hold",
        lambda *_args, **_kwargs: pytest.fail("existing run must not hold again"),
    )

    def fake_watch(
        actual_client,
        *,
        run_dir,
        interval_seconds,
        fail_on_signal,
    ):
        calls.append(
            (
                "watch",
                actual_client,
                run_dir,
                interval_seconds,
                fail_on_signal,
            )
        )
        (run_dir / "watchdog-stopped.json").write_text("{}")
        (run_dir / "restore-in-progress.json").write_text("{}")
        (run_dir / "controllers-restored.json").write_text("{}")

    monkeypatch.setattr(operator.freeze, "watch", fake_watch)
    monkeypatch.setattr(
        operator.freeze,
        "_live_lease",
        lambda actual_client, *, run_id: {"status": "released"},
    )

    operator.serve(
        client,
        state_dir=state_dir,
        run_id=run_dir.name,
        uid=os.geteuid(),
        gid=os.getegid(),
    )

    assert calls[0] == ("verify", run_dir)
    assert calls[1][0] == "watch"
    assert calls[1][2] == run_dir
    assert calls[1][4] is True


def test_serve_keeps_failed_watchdog_exec_capable_for_restore(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=operator.STATE_MODE)
    run_dir = state_dir / "failed"
    client = object()
    lease_status = "failed"

    def fake_hold(*_args, **_kwargs):
        run_dir.mkdir(mode=operator.STATE_MODE)
        return run_dir

    def fake_watch(*_args, **_kwargs):
        raise operator.freeze.FreezeError("simulated eviction")

    def fake_live_lease(_client, *, run_id):
        assert run_id == run_dir.name
        return {"status": lease_status}

    def finish_restore(_seconds):
        nonlocal lease_status
        (run_dir / "restore-in-progress.json").write_text("{}")
        (run_dir / "controllers-restored.json").write_text("{}")
        lease_status = "released"

    monkeypatch.setattr(operator.freeze, "hold", fake_hold)
    monkeypatch.setattr(operator.freeze, "watch", fake_watch)
    monkeypatch.setattr(operator.freeze, "_live_lease", fake_live_lease)
    monkeypatch.setattr(operator.time, "sleep", finish_restore)

    operator.serve(
        client,
        state_dir=state_dir,
        run_id=run_dir.name,
        uid=os.geteuid(),
        gid=os.getegid(),
    )

    assert lease_status == "released"


def test_wait_keeps_restore_alive_between_evidence_and_release(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=operator.STATE_MODE)
    run_dir = state_dir / "restoring"
    run_dir.mkdir(mode=operator.STATE_MODE)
    (run_dir / "restore-in-progress.json").write_text("{}")
    (run_dir / "controllers-restored.json").write_text("{}")
    statuses = iter(("closed", "released"))
    sleeps = []

    monkeypatch.setattr(
        operator.freeze,
        "_live_lease",
        lambda _client, *, run_id: {"status": next(statuses)},
    )
    monkeypatch.setattr(operator.time, "sleep", sleeps.append)

    operator._wait_for_exact_restore(
        object(),
        run_dir=run_dir,
        poll_seconds=0.25,
    )

    assert sleeps == [0.25]


def test_serve_keeps_incomplete_existing_hold_available_for_restore(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=operator.STATE_MODE)
    run_dir = state_dir / "incomplete"
    run_dir.mkdir(mode=operator.STATE_MODE)
    waited = []

    monkeypatch.setattr(
        operator,
        "_validate_incomplete_hold_for_restore",
        lambda client, *, run_dir: None,
    )
    monkeypatch.setattr(
        operator,
        "_wait_for_exact_restore",
        lambda client, *, run_dir, poll_seconds: waited.append(run_dir),
    )

    operator.serve(
        object(),
        state_dir=state_dir,
        run_id=run_dir.name,
        uid=os.geteuid(),
        gid=os.getegid(),
    )

    assert waited == [run_dir]


def test_serve_fails_closed_when_signal_interrupts_hold(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=operator.STATE_MODE)
    run_dir = state_dir / "interrupted-hold"
    handlers = {}
    waited = []

    monkeypatch.setattr(
        operator.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )

    def fake_hold(*_args, **_kwargs):
        run_dir.mkdir(mode=operator.STATE_MODE)
        handlers[operator.signal.SIGTERM](operator.signal.SIGTERM, None)

    monkeypatch.setattr(operator.freeze, "hold", fake_hold)
    monkeypatch.setattr(
        operator,
        "_validate_incomplete_hold_for_restore",
        lambda client, *, run_dir: None,
    )
    monkeypatch.setattr(
        operator,
        "_wait_for_exact_restore",
        lambda client, *, run_dir, poll_seconds: waited.append(run_dir),
    )

    operator.serve(
        object(),
        state_dir=state_dir,
        run_id=run_dir.name,
        uid=os.geteuid(),
        gid=os.getegid(),
    )

    assert waited == [run_dir]


def test_serve_refuses_unrecoverable_incomplete_existing_run(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=operator.STATE_MODE)
    run_dir = state_dir / "unrecoverable"
    run_dir.mkdir(mode=operator.STATE_MODE)

    with pytest.raises(operator.OperatorError, match="controller-before authority"):
        operator.serve(
            object(),
            state_dir=state_dir,
            run_id=run_dir.name,
            uid=os.geteuid(),
            gid=os.getegid(),
        )


def test_crop_image_pins_official_kubectl_binary():
    dockerfile = (
        Path(__file__).resolve().parents[1]
        / "docker"
        / "ladder-crop-distill"
        / "Dockerfile"
    ).read_text()

    assert (
        "ADD https://dl.k8s.io/release/v1.28.15/bin/linux/amd64/kubectl "
        "/tmp/kubectl"
    ) in dockerfile
    assert (
        "1f7651ad0b50ef4561aa82e77f3ad06599b5e6b0b2a5fb6c4f474d95a77e41c5"
        "  /tmp/kubectl"
    ) in dockerfile
    assert "install -o root -g root -m 0755 /tmp/kubectl" in dockerfile


def test_pr_and_publish_images_smoke_operator_runtime():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "build-pipeline-images.yml"
    ).read_text()

    assert workflow.count("version --client=true --output=json") == 2
    assert workflow.count(
        "/opt/imintengine/scripts/crop_source_freeze_operator.py"
    ) == 2
    assert workflow.count("prepare --state-parent /state-parent") == 2
    assert workflow.count(
        "--tmpfs /state-parent:rw,nosuid,nodev,noexec,size=1m"
    ) == 2
