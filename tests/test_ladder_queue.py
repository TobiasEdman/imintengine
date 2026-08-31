"""The ladder queue must never over-submit, skip ahead, or double-submit.

Its whole job is to keep the GPU fed without creating a Job the quota will
reject — such a Job gets no pod and reads 0/1 for ever, which looks exactly
like a healthy run to anything polling .status.
"""
from __future__ import annotations

import json
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


def test_rungs_3_4_locked_without_gate(monkeypatch, tmp_path, capsys):
    """A column's rungs 3/4 stay OUT of the queue until its distill job
    has written _GATE_OK — sidecars that do not exist cannot be trained
    on. With no gates open, behaviour equals the original rung-1/2 queue."""
    assert lq.RUNGS == (1, 2) and lq.GATED_RUNGS == (3, 4)
    monkeypatch.setattr(lq, "DISTILL_ROOT", tmp_path / "distill")
    _run(monkeypatch, tmp_path, free=1000, existing=set())
    out = capsys.readouterr().out
    assert "-r3-" not in out and "-r4-" not in out
    assert "await their distill gate" in out


def test_gate_unlocks_both_rungs_for_that_column_only(monkeypatch, tmp_path,
                                                      capsys):
    """_GATE_OK for one column queues ITS r3+r4 (they share sidecars and
    neither warm-starts from the other); every other column stays locked.
    This is what lets fast columns run two-by-two while croma trails in
    its own lane [user-stated 2026-08-31]."""
    droot = tmp_path / "distill"
    (droot / "tessera_r2").mkdir(parents=True)
    (droot / "tessera_r2" / "_GATE_OK").touch()
    monkeypatch.setattr(lq, "DISTILL_ROOT", droot)
    created = _run_with_distill(monkeypatch, tmp_path, droot,
                                free=1000, existing=set())
    r34 = [c for c in created if "-r3-" in c or "-r4-" in c]
    assert r34 == ["ladder-r3-tessera-job.yaml", "ladder-r4-tessera-job.yaml"]


def test_gated_done_column_not_resubmitted(monkeypatch, tmp_path, capsys):
    """An open gate + an already-trained r3 (checkpoint dir, reaper-proof)
    must not resubmit r3."""
    droot = tmp_path / "distill"
    (droot / "tessera_r2").mkdir(parents=True)
    (droot / "tessera_r2" / "_GATE_OK").touch()
    lroot = tmp_path / "ladder"
    _log(lroot, "tessera", 3, status="completed", last_epoch=30)
    created = _run_with_distill(monkeypatch, tmp_path, droot,
                                free=1000, existing=set(), ladder_root=lroot)
    assert "ladder-r3-tessera-job.yaml" not in created
    assert "ladder-r4-tessera-job.yaml" in created


def _run_with_distill(monkeypatch, tmp_path, droot, *, free, existing,
                      ladder_root=None, dry=False):
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
    argv = ["ladder_queue.py", "--repo", str(REPO),
            "--log", str(tmp_path / "l.log"),
            "--distill-root", str(droot)]
    if ladder_root is not None:
        argv += ["--ladder-root", str(ladder_root)]
    monkeypatch.setattr("sys.argv", argv)
    assert lq.main() == 0
    return created


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


def test_no_suspend_while_gates_are_locked(monkeypatch, tmp_path, capsys):
    """Rungs 1/2 fully submitted but distill gates still closed: the queue
    must KEEP TICKING — locked rung-3/4 work is future work, not done
    work. Self-suspending here would orphan every gated rung."""
    monkeypatch.setattr(lq, "DISTILL_ROOT", tmp_path / "distill")
    everything = {f"ladder-r{r}-{m}" for r in lq.RUNGS for m in lq.MODEL_ORDER}
    created = _run(monkeypatch, tmp_path, free=1000, existing=everything, dry=False)
    assert "SUSPEND" not in created
    assert "DONE" not in capsys.readouterr().out


def test_self_suspends_when_all_24_are_drained(monkeypatch, tmp_path, capsys):
    """Every rung-1..4 job existing (or trained) + every gate open = the
    campaign is fully fed; only then does the queue retire itself."""
    droot = tmp_path / "distill"
    for m in lq.MODEL_ORDER:
        (droot / f"{m}_r2").mkdir(parents=True)
        (droot / f"{m}_r2" / "_GATE_OK").touch()
    everything = {f"ladder-r{r}-{m}"
                  for r in (*lq.RUNGS, *lq.GATED_RUNGS)
                  for m in lq.MODEL_ORDER}
    created = _run_with_distill(monkeypatch, tmp_path, droot,
                                free=1000, existing=everything)
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


def _log(tmp_path, model, rung, *, status, last_epoch, target=30):
    d = tmp_path / f"{model}_r{rung}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "training_log.json").write_text(json.dumps({
        "status": status,
        "config": {"epochs": target},
        "epochs": [{"epoch": e} for e in range(1, last_epoch + 1)],
    }))
    return tmp_path


def test_reaped_completed_run_is_not_resubmitted(tmp_path):
    """gpu-reaper deletes finished Jobs — the checkpoint dir is the record.

    tessera_r1 finished 30/30 with status "stopped" (the early-stop label)
    and was reaped; a job-existence check alone had it queued for a redo.
    """
    _log(tmp_path, "tessera", 1, status="stopped", last_epoch=30)
    assert lq.already_trained(tmp_path, 1, "tessera") is True


def test_completed_status_counts_as_done(tmp_path):
    _log(tmp_path, "clay", 2, status="completed", last_epoch=12)
    assert lq.already_trained(tmp_path, 2, "clay") is True


def test_partial_run_is_not_done(tmp_path):
    """A run killed halfway must still be resubmittable."""
    _log(tmp_path, "croma", 1, status="stopped", last_epoch=7)
    assert lq.already_trained(tmp_path, 1, "croma") is False


def test_missing_dir_is_not_done(tmp_path):
    assert lq.already_trained(tmp_path, 1, "terramind") is False


def test_margin_allows_an_80gi_job_into_83gi(monkeypatch, tmp_path, capsys):
    """A 3Gi margin stranded an arithmetically-free 80Gi slot for an hour."""
    monkeypatch.setattr(lq, "already_trained", lambda *a: False)
    _run(monkeypatch, tmp_path, free=83.1, existing=set())
    assert "would submit ladder-r1-prithvi600m" in capsys.readouterr().out
