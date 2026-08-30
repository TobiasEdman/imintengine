"""The ladder queue must never over-submit, skip ahead, or double-submit.

Its whole job is to keep the GPU fed without creating a Job the quota will
reject — such a Job gets no pod and reads 0/1 for ever, which looks exactly
like a healthy run to anything polling .status.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import ladder_queue as lq  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def test_quantity_parsing():
    assert lq._to_gi("250Gi") == 250
    assert lq._to_gi("252800Mi") == pytest.approx(246.875)
    assert lq._to_gi("1Ti") == 1024
    assert lq._to_gi(str(80 * 1024 ** 3)) == 80


def test_reads_request_from_the_real_manifests():
    """Requests come from the manifests, never a constant in the script."""
    big = lq.requested_gi(lq.manifest_path(REPO, 1, "prithvi600m"))
    small = lq.requested_gi(lq.manifest_path(REPO, 1, "clay"))
    assert big == 80 and small == 48


def test_order_is_largest_first_within_a_rung():
    gis = [lq.requested_gi(lq.manifest_path(REPO, 1, m)) for m in lq.MODEL_ORDER]
    assert gis == sorted(gis, reverse=True), (
        "a 48Gi job submitted before an 80Gi one can strand a slot")


def test_only_rungs_1_and_2(monkeypatch, tmp_path, capsys):
    """Rungs 3-4 need distillation sidecars that do not exist yet."""
    assert lq.RUNGS == (1, 2)
    _run(monkeypatch, tmp_path, free=1000, existing=set())
    out = capsys.readouterr().out
    assert "-r3-" not in out and "-r4-" not in out


def _run(monkeypatch, tmp_path, *, free, existing, dry=True):
    created: list[str] = []

    def fake_kubectl(args, namespace):
        if args[:2] == ["get", "job"]:
            if args[2] in existing:
                return f"job.batch/{args[2]}\n"
            raise RuntimeError("NotFound")
        if args[0] == "create":
            created.append(Path(args[2]).name)
            return "created\n"
        if args[0] == "patch":
            created.append("SUSPEND")
            return "patched\n"
        raise AssertionError(args)

    monkeypatch.setattr(lq, "_kubectl", fake_kubectl)
    monkeypatch.setattr(lq, "free_gi", lambda ns, q: free)
    argv = ["ladder_queue.py", "--repo", str(REPO), "--log", str(tmp_path / "l.log")]
    if dry:
        argv.append("--dry-run")
    monkeypatch.setattr("sys.argv", argv)
    assert lq.main() == 0
    return created


def test_stops_at_first_job_that_does_not_fit(monkeypatch, tmp_path, capsys):
    """With room for one 80Gi job it must not skip ahead to a 48Gi one."""
    _run(monkeypatch, tmp_path, free=90, existing=set())
    out = capsys.readouterr().out
    assert "would submit ladder-r1-prithvi600m" in out
    assert "would submit ladder-r1-prithvi300m" not in out
    assert "clay" not in out, "skipped ahead to a smaller job"
    assert "stop:" in out


def test_never_exceeds_quota(monkeypatch, tmp_path, capsys):
    _run(monkeypatch, tmp_path, free=3, existing=set())
    out = capsys.readouterr().out
    assert "would submit" not in out


def test_does_not_resubmit_existing_jobs(monkeypatch, tmp_path, capsys):
    already = {f"ladder-r1-{m}" for m in lq.MODEL_ORDER}
    _run(monkeypatch, tmp_path, free=1000, existing=already)
    out = capsys.readouterr().out
    assert "ladder-r1-" not in out.replace("pending=", "")
    assert "would submit ladder-r2-prithvi600m" in out


def test_self_suspends_when_drained(monkeypatch, tmp_path, capsys):
    everything = {f"ladder-r{r}-{m}" for r in lq.RUNGS for m in lq.MODEL_ORDER}
    created = _run(monkeypatch, tmp_path, free=1000, existing=everything, dry=False)
    assert "SUSPEND" in created
    assert "DONE" in capsys.readouterr().out


def test_log_is_written_to_the_pvc(monkeypatch, tmp_path):
    log = tmp_path / "ops" / "ladder_queue.log"
    monkeypatch.setattr(lq, "_kubectl", lambda a, n: (_ for _ in ()).throw(RuntimeError("NotFound"))
                        if a[:2] == ["get", "job"] else "ok")
    monkeypatch.setattr(lq, "free_gi", lambda ns, q: 0)
    monkeypatch.setattr("sys.argv",
                        ["ladder_queue.py", "--repo", str(REPO), "--log", str(log)])
    assert lq.main() == 0
    assert log.exists() and "pending=12" in log.read_text()
