"""Evidence-before-deletion gates for the GPU reaper.

The reaper destroyed its own diagnostic trail twice (ERA5 control 2026-08-27,
wave-1 retrain 2026-08-29): deleting a finished Job cascades to its pods and
their logs. These tests pin the contract that makes that impossible:

1. No deletion ever happens without a successfully written evidence archive.
2. A successful sweep leaves the job spec, pod specs, pod logs and events on
   disk, plus a run report that survives the reaper pod's own TTL.
3. Dry-run mode touches nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.reap_gpu_jobs as reap

DUE = [{"job": "train-x", "pod": "train-x-abc12", "phase": "Failed",
        "gpus": 1, "age_min": 120.0}]


def _fake_kubectl_factory(calls: list[list[str]], *, fail_on_logs: bool = False):
    def fake(args: list[str], context: str, namespace: str) -> str:
        calls.append(args)
        if args[:2] == ["get", "job"]:
            return "kind: Job\nmetadata:\n  name: train-x\n"
        if args[:2] == ["get", "pods"]:
            return json.dumps({"items": [{"metadata": {"name": "train-x-abc12"}}]})
        if args[0] == "logs":
            if "--previous" in args:
                raise RuntimeError("previous terminated container not found")
            if fail_on_logs:
                raise RuntimeError("simulated: pods/log forbidden")
            return "Traceback: the actual failure evidence\n"
        if args[:2] == ["get", "events"]:
            return "LAST SEEN   REASON   MESSAGE\n"
        if args[0] == "delete":
            return "job.batch/train-x deleted\n"
        raise AssertionError(f"unexpected kubectl call: {args}")
    return fake


def _run_main(monkeypatch, tmp_path: Path, *, apply: bool, fail_on_logs: bool = False):
    calls: list[list[str]] = []
    monkeypatch.setattr(reap, "_kubectl", _fake_kubectl_factory(calls, fail_on_logs=fail_on_logs))
    monkeypatch.setattr(reap, "collect", lambda *a: (list(DUE), [], 1))
    argv = ["reap_gpu_jobs.py", "--grace-minutes", "60",
            "--archive-dir", str(tmp_path / "archive")]
    if apply:
        argv.append("--apply")
    monkeypatch.setattr("sys.argv", argv)
    assert reap.main() == 0
    return calls


def _deletes(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c and c[0] == "delete"]


def test_archive_failure_forbids_delete(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, apply=True, fail_on_logs=True)
    assert _deletes(calls) == [], "deleted a job whose evidence could not be archived"


def test_unwritable_archive_root_forbids_delete(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(reap, "_kubectl", _fake_kubectl_factory(calls))
    monkeypatch.setattr(reap, "collect", lambda *a: (list(DUE), [], 1))
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, not a directory — mkdir must fail")
    monkeypatch.setattr("sys.argv", ["reap_gpu_jobs.py", "--apply",
                                     "--archive-dir", str(blocked)])
    assert reap.main() == 0
    assert _deletes(calls) == []


def test_successful_archive_then_delete_and_run_report(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, apply=True)
    assert _deletes(calls) == [["delete", "job", "train-x", "--wait=false"]]

    root = tmp_path / "archive"
    job_dirs = [d for d in root.iterdir() if d.name.startswith("train-x-")]
    assert len(job_dirs) == 1
    d = job_dirs[0]
    assert (d / "job.yaml").read_text().startswith("kind: Job")
    assert json.loads((d / "pods.json").read_text())[0]["metadata"]["name"] == "train-x-abc12"
    assert "actual failure evidence" in (d / "train-x-abc12.log").read_text()
    assert (d / "events.txt").exists()
    # --previous had no restarted container: absent is correct, not an error.
    assert not (d / "train-x-abc12.previous.log").exists()

    reports = list((root / "runs").iterdir())
    assert len(reports) == 1
    assert "deleted train-x" in reports[0].read_text()


def test_job_scope_excludes_other_due_jobs(monkeypatch, tmp_path):
    """--job restricts deletion to exactly the named Jobs."""
    calls: list[list[str]] = []
    monkeypatch.setattr(reap, "_kubectl", _fake_kubectl_factory(calls))
    other = {"job": "train-other", "pod": "train-other-zzz99", "phase": "Failed",
             "gpus": 1, "age_min": 500.0}
    monkeypatch.setattr(reap, "collect", lambda *a: ([dict(DUE[0]), other], [], 2))
    monkeypatch.setattr("sys.argv", ["reap_gpu_jobs.py", "--apply",
                                     "--grace-minutes", "0",
                                     "--job", "train-x",
                                     "--archive-dir", str(tmp_path / "archive")])
    assert reap.main() == 0
    assert _deletes(calls) == [["delete", "job", "train-x", "--wait=false"]]


def test_dry_run_touches_nothing(monkeypatch, tmp_path):
    calls = _run_main(monkeypatch, tmp_path, apply=False)
    assert _deletes(calls) == []
    assert not [c for c in calls if c[0] == "logs"], "dry run must not read logs"
    assert not (tmp_path / "archive").exists()
